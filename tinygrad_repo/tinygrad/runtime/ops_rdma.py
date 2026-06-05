from __future__ import annotations
import mmap, struct, functools
from typing import cast
from tinygrad.uop.ops import sint
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocatorBase, HCQAllocator, HWQueue, HCQBuffer, FileIOInterface
from tinygrad.runtime.support.system import System, PCIIfaceBase, PCIAllocationMeta
from tinygrad.runtime.support.memory import VirtMapping, AddrSpace
from tinygrad.runtime.support.mlx.mlxdev import MLXDev, MLXQP
from tinygrad.helpers import unwrap, to_be32, to_be64

class RDMACopyQueue(HWQueue):
  def __init__(self, dev:RDMADevice):
    self.dev = dev
    super().__init__()

  def _wqe_data(self, buf:HCQBuffer, sz:int, nic:RDMADevice) -> bytes:
    cast(HCQAllocatorBase, nic.allocator).map(buf)
    return struct.pack('>IIQ', sz, buf.mappings[nic].meta, buf.mappings[nic].va_addr + (buf.va_addr - buf.base.va_addr))

  def encode_ring(self, hwq:HWQueue, dev:HCQCompiled, iface:MLXIface, qp:MLXQP, cq_buf:HCQBuffer, head:sint, ring_uar:bool=False):
    for buf in [iface.dbr_buf, cq_buf] + ([iface.uar_buf] if ring_uar else []): cast(HCQAllocator, dev.allocator).map(buf)
    hwq.write(iface.dbr_buf.offset(qp.qp_dbr + (4 if ring_uar else 0)), to_be32(head + 1))
    if ring_uar: hwq.write(iface.uar_buf.offset(0x800), to_be64(((head << 8) | 0x0a) << 32 | ((qp.qp_info['qpn'] << 8) | 2)), b64=True)
    hwq.poll_bit(cq_buf.offset((head & (qp.cq_size - 1)) * 64 + 60, 4), ((head >> (qp.cq_size.bit_length() - 1)) & 1) << 24, mask=0x01000000)
    hwq.write(iface.dbr_buf.offset(qp.cq_dbr), to_be32((head + 1) & 0xFFFFFF))
    return self

  def copy(self, dest:HCQBuffer, src:HCQBuffer, sz:int):
    src_qp, dest_qp, _, _ = self.dev.iface.connect(remote_nic:=unwrap(dest.owner).rdma_dev())

    sq_wqe = bytearray(64)
    sq_wqe[4:8] = struct.pack('>I', (src_qp.qp_info['qpn'] << 8) | 2)
    sq_wqe[11] = 0x08 # CE: signal completion
    sq_wqe[16:32] = self._wqe_data(src, sz, self.dev)

    self.q(remote_nic, bytes(sq_wqe), self._wqe_data(dest, sz, remote_nic))
    return self

  def _submit(self, dev:RDMADevice):
    for remote_nic, sq_wqe, rq_wqe in zip(self._q[0::3], self._q[1::3], self._q[2::3]):
      src_qp, dest_qp, _, _ = dev.iface.connect(remote_nic)
      assert src_qp.head + 1 - to_be32(src_qp.dev.dbr[src_qp.qp_dbr // 4 + 1]) <= (1 << src_qp.log_sq_size), "SQ ring full"
      assert src_qp.head + 1 - to_be32(dest_qp.dev.dbr[dest_qp.qp_dbr // 4]) <= (1 << dest_qp.log_rq_size), "RQ ring full"
      dest_qp.qp_buf.view((src_qp.head & ((1 << dest_qp.log_rq_size) - 1)) * 16, 16)[:] = rq_wqe
      sq_view = src_qp.qp_buf.view(src_qp.sq_offset + (src_qp.head & ((1 << src_qp.log_sq_size) - 1)) * 64, 64)
      sq_view[:] = struct.pack('>I', (src_qp.head << 8) | 0x0a) + sq_wqe[4:]
      src_qp.head += 1

class MLXIface(PCIIfaceBase):
  def __init__(self, dev:RDMADevice, dev_id:int):
    cl, pcibus = System.list_devices(vendor=0x15b3, devices=((0xffff, (0x101b,)),))[dev_id]
    self.dev = dev
    self.pci_dev = cl("mlx", pcibus)
    self.mlx_dev = MLXDev(self.pci_dev, ip=f"10.0.0.{dev_id}")
    self.uar_buf = self._buf([self.mlx_dev.pci_dev.bar_info(0)[0] + self.mlx_dev.uar * 0x1000])
    self.dbr_buf = self._buf(self.mlx_dev.dbr_paddrs)

  def is_bar_small(self) -> bool: return False

  def _buf(self, paddrs:list[int]) -> HCQBuffer:
    va = FileIOInterface.anon_mmap(0, size:=len(paddrs) * 0x1000, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, 0)
    mapping = VirtMapping(va, size, [(p, 0x1000) for p in paddrs], AddrSpace.SYS, uncached=True, snooped=True)
    return HCQBuffer(va, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=False), owner=self.dev)

  @functools.cache
  def connect(self, remote_nic:RDMADevice) -> tuple[MLXQP, MLXQP, HCQBuffer, HCQBuffer]:
    src_qp, dest_qp = MLXQP(self.mlx_dev, log_sq_size=7, log_rq_size=7), MLXQP(remote_nic.iface.mlx_dev, log_sq_size=7, log_rq_size=7)
    src_qp.connect(dest_qp)
    dest_qp.connect(src_qp)
    return src_qp, dest_qp, self._buf(src_qp.cq_paddrs), remote_nic.iface._buf(dest_qp.cq_paddrs)

class RDMAAllocator(HCQAllocatorBase):
  def __init__(self, dev:RDMADevice): super().__init__(dev, batch_cnt=0)

  def _map(self, buf:HCQBuffer) -> HCQBuffer:
    owner = unwrap(buf.base.owner)
    bar, paddrs = owner.iface.pci_dev.bar_info(owner.iface.vram_bar)[0], buf.base.meta.mapping.paddrs  # type: ignore[attr-defined]
    page_sz = (2 << 20) if min(sz for _, sz in paddrs) >= (2 << 20) else (4 << 10)
    pages = [bar + p + off for p, sz in paddrs for off in range(0, sz, page_sz)]
    return HCQBuffer(bar + paddrs[0][0], buf.base.size, owner=owner,
                     meta=self.dev.iface.mlx_dev.register_mem(pages, len(pages) * page_sz, page_sz.bit_length() - 1))

  def _do_free(self, buf:HCQBuffer, options): self.dev.iface.mlx_dev.unregister_mem(buf.meta)
  def _unmap(self, mb): self.dev.iface.mlx_dev.unregister_mem(mb.meta)

  def _transfer(self, dest:HCQBuffer, src:HCQBuffer, sz:int, src_dev:HCQCompiled, dest_dev:HCQCompiled):
    # sync device
    src_q = unwrap(dest_dev.hw_compute_queue_t)().wait(src_dev.timeline_signal, src_dev.timeline_value - 1)
    dest_q = unwrap(dest_dev.hw_compute_queue_t)().wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1)

    # rdma body + encode doorbell rings
    src_qp, dest_qp, src_cq_buf, dest_cq_buf = self.dev.iface.connect(remote_nic:=dest_dev.rdma_dev())
    RDMACopyQueue(self.dev).copy(dest, src, sz) \
                           .encode_ring(src_q, src_dev, self.dev.iface, src_qp, src_cq_buf, src_qp.head, ring_uar=True) \
                           .encode_ring(dest_q, dest_dev, remote_nic.iface, dest_qp, dest_cq_buf, src_qp.head) \
                           .submit(self.dev)

    # signal completion
    src_q.signal(src_dev.timeline_signal, src_dev.next_timeline()).submit(src_dev)
    dest_q.signal(dest_dev.timeline_signal, dest_dev.next_timeline()).submit(dest_dev)

class RDMADevice(HCQCompiled):
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = MLXIface(self, self.device_id)
    super().__init__(device, RDMAAllocator(self), [], None, signal_t=None)
