from __future__ import annotations
from typing import cast
import functools, struct, operator
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import round_up, ceildiv, unwrap, to_tuple, flatten
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import BNXTDev, BNXTQP, db_value, send_wqe, recv_wqe, WQE_SIZE, RING_ENTRIES, CQ_ENTRIES, MTU
from tinygrad.runtime.support.hcq2 import unwrap_view, rt_addr, to_name
from tinygrad.runtime.support.memory import AddrSpace, MMIOInterface, VirtMapping, MemoryManager
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta, System
from tinygrad.runtime.support.hcq import hcq_filter_visible_devices
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

RDMA_CHUNK = 1 << 30 # a wqe length is 32 bits

BNXT_IDS = (0x14e4, ((0xffff, (0x1760,)),), 0x02) # vendor, (mask, device ids), base class

class BNXTIface(PCIIfaceBase):
  def __init__(self, dev:RDMADevice, index:int):
    super().__init__(dev, index, *BNXT_IDS[:2], vram_bar=2, va_start=MemoryManager.va_allocator.base, va_size=MemoryManager.va_allocator.size,
                     dev_impl_t=BNXTDev, base_class=BNXT_IDS[2])
  def device_fini(self): self.dev_impl.fini()

  def storage(self, mem:MMIOInterface, paddrs:list[int], snooped:bool=True) -> BufferStorage: # nic memory any gpu of the node maps
    va = MemoryManager.alloc_vaddr(size:=round_up(mem.nbytes, 0x1000), 0x1000)
    mapping = VirtMapping(va, size, [(p, 0x1000) for p in paddrs], AddrSpace.SYS, uncached=True, snooped=snooped)
    return BufferStorage(va, PCIAllocationMeta(mapping, True), mem)
  def buffer(self, mem:MMIOInterface, paddrs:list[int], snooped:bool=True) -> Buffer:
    return Buffer(self.dev.device, mem.nbytes, dtypes.uint8, opaque=self.storage(mem, paddrs, snooped))

  @functools.cached_property
  def doorbell(self) -> Buffer:
    off = self.dev_impl.db_off & ~0xfff
    return self.buffer(self.pci_dev.map_bar(2, off=off, size=0x1000), [self.pci_dev.bar_info(2)[0] + off], snooped=False)

class BNXTAllocator(Allocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: # sysmem: the counters, the batch slots, the timeline
    return self.dev.iface.storage(*self.dev.iface.pci_dev.alloc_sysmem(round_up(size, 0x1000)))
  def _copyin(self, dest:BufferStorage, src:memoryview): unwrap(dest.host)[:len(src)] = src
  def _map(self, buf:Buffer) -> BufferStorage: # a memory region over the buffer's pages, keyed at the gpu virtual address
    iface = getattr(Device[buf.device], "iface", None)
    if not isinstance(iface, PCIIfaceBase) or iface.peer_group != self.dev.peer_group: raise RuntimeError("RDMA requires memory on its node")
    mapping = buf.meta.mapping # the nic reaches vram over pcie: the bar, even where gpus reach each other over xgmi
    paddrs = mapping.paddrs if mapping.aspace is AddrSpace.SYS else PCIIfaceBase.p2p_paddrs(iface, mapping.paddrs)[0]
    align = buf._buf | functools.reduce(operator.or_, (p | s for p, s in paddrs)) # every address a multiple of the page: fewer pbl entries
    log_page = max(l for l in (12, 13, 16, 18, 20, 21, 22, 30) if not align & ((1 << l) - 1)) # the page sizes the nic has
    key = self.dev.iface.dev_impl.register_mem([p + off for p, size in paddrs for off in range(0, size, 1 << log_page)],
                                              mapping.size, log_page, va=buf._buf)
    return BufferStorage(key, key)
  def _offset(self, buf, size:int, offset:int): return buf
  def _unmap(self, storage:BufferStorage): self.dev.iface.dev_impl.unregister_mem(storage.meta)

def rdma_nic_for(dev) -> RDMADevice|None:
  try: count = len(hcq_filter_visible_devices(System.list_devices(*BNXT_IDS), "RDMA"))
  except RuntimeError: return None
  return next((cast(RDMADevice, n) for i in range(count) if (n:=Device[f"RDMA:{i}"]).peer_group == dev.peer_group), None)

class RDMADevice(Compiled):
  ifaces = [BNXTIface]

  def __init__(self, device:str):
    self.iface = self._select_iface(device)
    super().__init__(device, BNXTAllocator(self), [], None)

# *****************
# UOps implementation

@functools.cache
def rdma_qp(pair:tuple[str, str]) -> dict[str, BNXTQP]:
  # one qp per gpu pair
  nics = [unwrap(rdma_nic_for(Device[d])) for d in pair]
  qps = {nic.device: BNXTQP(nic.iface.dev_impl) for nic in nics}
  for nic, q in zip(nics, qps.values()):
    bufs = {name: nic.iface.buffer(getattr(q, name).ring, getattr(q, name).paddrs) for name in ("sq", "rq", "scq", "rcq")}
    bufs |= {name: Buffer(nic.device, 1, dtypes.uint64, initial_value=bytes(8)) for name in ("sq_seq", "rq_seq", "psn")} | {"db": nic.iface.doorbell}
    rules = [(UPat(Ops.PARAM, tag=to_name("rdma", *pair, n)), lambda ctx, b=b: b) for n, b in bufs.items()]
    nic.pm_bufferize = PatternMatcher(rules) + nic.pm_bufferize
  for a, b in (nics, nics[::-1]): qps[a.device].connect(qps[b.device].qpn, b.iface.dev_impl.local_gid, b.iface.dev_impl.mac)
  return qps

def rdma_mem(nic:str, pair:tuple[str, str], name:str, size:int, dtype:DType=dtypes.uint8) -> UOp:
  return UOp.placeholder((size,), dtype, 0, device=nic, volatile=True, tag=to_name("rdma", *pair, name))
def rdma_ring(nic:str, pair:tuple[str, str], is_recv:bool) -> UOp:
  return rdma_mem(nic, pair, "rq" if is_recv else "sq", RING_ENTRIES * (WQE_SIZE if is_recv else WQE_SIZE + 8))
def rdma_cq(nic:str, pair:tuple[str, str], is_recv:bool) -> UOp: return rdma_mem(nic, pair, "rcq" if is_recv else "scq", CQ_ENTRIES * 32)
def rdma_seq(nic:str, pair:tuple[str, str], is_recv:bool) -> UOp: return rdma_mem(nic, pair, "rq_seq" if is_recv else "sq_seq", 1, dtypes.uint64)
def rdma_psn(nic:str, pair:tuple[str, str]) -> UOp: return rdma_mem(nic, pair, "psn", 1, dtypes.uint64) # the next psn of the sends
def rdma_db(nic:str, pair:tuple[str, str]) -> UOp: return rdma_mem(nic, pair, "db", 0x1000)

def rdma_wire(call:UOp) -> UOp|None:
  if call.op is not Ops.CALL or call.src[0].op is not Ops.COPY: return None
  return next((b for b in get_call_arg_uops(call) if to_tuple(b.device)[0].startswith("RDMA")), None)
def is_rdma(call:UOp) -> bool: return rdma_wire(call) is not None

def queue_of(call:UOp) -> tuple[tuple[str, str], bool]:
  (dst, src), wire = get_call_arg_uops(call), unwrap(rdma_wire(call))
  gpu = to_tuple((dst if wire is src else src).device)[0]
  return (min(gpu, wire.tag), max(gpu, wire.tag)), wire is src

def ins(name:str, *src:UOp|int) -> UOp:
  return UOp(Ops.INS, arg=(name, dtypes.void), src=tuple(UOp.const(s, dtypes.uint32) if isinstance(s, int) else s for s in src))

def rdma_copies(devs:tuple[str, ...], calls:list[UOp]) -> list[list[UOp]]: # the ops of each copy of a submit on one queue
  (pair, is_recv), nic = queue_of(calls[0]), cast(RDMADevice, Device[unwrap(rdma_wire(calls[0])).device])
  qp = rdma_qp(pair)[nic.device]
  ring, cq = rdma_ring(nic.device, pair, is_recv), rdma_cq(nic.device, pair, is_recv)
  seq, psn = rdma_seq(nic.device, pair, is_recv), rdma_psn(nic.device, pair)
  bufs = [get_call_arg_uops(c)[0 if is_recv else 1] for c in calls]
  wqes, packets = sum(ceildiv(b.nbytes(), RDMA_CHUNK) for b in bufs), sum(ceildiv(b.nbytes(), MTU) for b in bufs)

  assert wqes <= min(RING_ENTRIES, CQ_ENTRIES), "a batch posts at most a ring of wqes per pair"

  # next slot and psn persist in nic memory. read once per submit and own it
  n, p = seq.index(0).load(), psn.index(0).load()
  advances = [seq.index(0).store(n + wqes)] + ([] if is_recv else [psn.index(0).store(p + packets)])

  # gpu addresses are rt patches
  ring_addr, cq_addr = rt_addr(ring, devs, *advances), rt_addr(cq, devs)
  db = rdma_db(nic.device, pair).getaddr(devs) + (nic.iface.dev_impl.db_off & 0xfff)
  ring_db = db_value(qp.qpn, bnxt.DBC_DBC_TYPE_RQ if is_recv else bnxt.DBC_DBC_TYPE_SQ, 0, 0)
  cq_db = db_value(qp.rcq_id if is_recv else qp.scq_id, bnxt.DBC_DBC_TYPE_CQ, 0, 0)

  copies = []
  for buf in bufs:
    ops:list[UOp] = []
    for off in range(0, buf.nbytes(), RDMA_CHUNK): # a wqe per chunk, each completed
      size = min(RDMA_CHUNK, buf.nbytes() - off)

      # sdma fills in the wqe
      hdr, key = struct.unpack("<8I", (recv_wqe if is_recv else send_wqe)(0, 0, size)[:32]), unwrap_view(buf)[0].getaddr(nic.device)
      ops += [ins("write", ring_addr + (n % RING_ENTRIES) * WQE_SIZE, *hdr, buf.getaddr(devs) + off, key.cast(dtypes.uint32), size)]

      # a send also fills in its msn entry: the slot, the psn after it (a psn per packet), its first psn
      if not is_recv: ops += [ins("write", ring_addr + RING_ENTRIES * WQE_SIZE + (n % RING_ENTRIES) * 8,
                                  ((n % RING_ENTRIES) << 48) | (((p + ceildiv(size, MTU)) & 0xffffff) << 24) | (p & 0xffffff))]

      # rings the doorbell: the slot after the wqe and the epoch of its pass
      ops += [ins("write", db, ((n + 1) % RING_ENTRIES | ((n + 1) // RING_ENTRIES & 1) << bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT) | ring_db)]

      # waits for the cqe, acks the cq
      ops += [ins("wait_eq", cq_addr + (n % CQ_ENTRIES) * 32 + 24, (n // CQ_ENTRIES & 1 ^ 1 | (2 if is_recv else 0)).cast(dtypes.uint16)),
              ins("write", db, ((n + 1) % CQ_ENTRIES | ((n + 1) // CQ_ENTRIES & 1) << bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT) | cq_db)]
      n, p = n + 1, p + ceildiv(size, MTU)

    # and invalidate the gpu caches on recv
    copies.append(ops + ([ins("barrier")] if is_recv else []))
  return copies

# *****************
# encode rewrite

def rdma_submit(ctx, submit:UOp, lin:UOp) -> UOp|None:
  if not (queues:={i: queue_of(u) for i, u in enumerate(lin.src) if is_rdma(u)}): return None

  ops = [[u] for u in lin.src]
  for q in dict.fromkeys(queues.values()):
    positions = [i for i in queues if queues[i] == q]
    for i, copy_ops in zip(positions, rdma_copies(lin.arg[0], [lin.src[i] for i in positions])): ops[i] = copy_ops
  return submit.replace(src=(lin.replace(src=tuple(flatten(ops))),))
pm_rdma_encode = PatternMatcher([(UPat(Ops.CUSTOM_FUNCTION, src=(UPat(Ops.LINEAR, name="lin"),), name="submit"), rdma_submit)])
