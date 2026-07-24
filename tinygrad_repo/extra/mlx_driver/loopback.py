#!/usr/bin/env python3
# GMMU=0 MLX_PCI=0000:41:00.0 PYTHONPATH=. python3 extra/mlx_driver/loopback.py
import struct
from tinygrad.helpers import getenv, round_up
from tinygrad.device import Device, BufferSpec
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.ops_amd import AMDComputeQueue
from tinygrad.helpers import to_be32, to_be64
from extra.mlx_driver.mlxdev import MLXDev, MLXQP

BUF_SIZE = 0x1000
MLX_PCI = getenv("MLX_PCI", "0000:41:00.0")
MLX_IP = getenv("MLX_IP", "10.0.0.1")

def map_phys_to_gpu(gpu, paddr, size):
  size = round_up(size, 0x1000)
  va = gpu.iface.dev_impl.mm.alloc_vaddr(size, align=0x1000)
  gpu.iface.dev_impl.mm.map_range(va, size, [(paddr, size)], aspace=AddrSpace.SYS, snooped=True, uncached=True)
  return va

print("[init] AMD GPU...")
gpu = Device["AMD"]

print(f"[init] MLX5 at {MLX_PCI}")
dev = MLXDev(PCIDevice("mlx5", MLX_PCI), ip=MLX_IP)
qp = MLXQP(dev)

print(f"[init] loopback connect QP 0x{qp.qp_info['qpn']:x}")
qp.connect(qp.qp_info['qpn'], dev.mac, int.from_bytes(dev.local_gid, 'big'))

# allocate src/dst via AMD GPU allocator
buf_src = gpu.allocator.alloc(BUF_SIZE, BufferSpec(nolru=True))
buf_dst = gpu.allocator.alloc(BUF_SIZE, BufferSpec(nolru=True))

bar_base = gpu.iface.pci_dev.bar_info(gpu.iface.vram_bar)[0]
src_paddr = buf_src.meta.mapping.paddrs[0][0] + bar_base
dst_paddr = buf_dst.meta.mapping.paddrs[0][0] + bar_base
print(f"src paddr=0x{src_paddr:x} dst paddr=0x{dst_paddr:x}")

# fill src, zero dst
test_msg = b"Hello from loopback send/recv!"
gpu.allocator._copyin(buf_src, memoryview(bytearray(test_msg.ljust(BUF_SIZE, b'\x00'))))
gpu.allocator._copyin(buf_dst, memoryview(bytearray(BUF_SIZE)))
gpu.synchronize()

# post recv WQE on RQ from CPU (scatter entry: byte_count, lkey, addr)
rq_mask = (1 << 4) - 1  # log_rq_size=4
rq_wqe = qp.qp_buf.view((qp.rq_head & rq_mask) * 16, 16)
rq_wqe[:] = struct.pack('>IIQ', len(test_msg), dev.mkey, dst_paddr)
qp.rq_head += 1
# ring recv doorbell from CPU (DBR offset 0 = recv counter)
dev.dbr[qp.qp_dbr // 4] = to_be32(qp.rq_head)

# build send WQE in SQ from CPU (opcode 0x0a = SEND, ds_count=2)
sq_head = qp.sq_head
sq_mask = (1 << qp.log_sq_size) - 1
wqe = qp.qp_buf.view(qp.sq_offset + (sq_head & sq_mask) * 64, 64)
wqe[:] = bytes(64)
wqe[0:8] = struct.pack('>II', (sq_head << 8) | 0x0a, (qp.qp_info['qpn'] << 8) | 2)
wqe[11] = 0x08  # CE: signal completion
wqe[16:32] = struct.pack('>IIQ', len(test_msg), dev.mkey, src_paddr)
qp.sq_head += 1
doorbell_val = to_be64(int.from_bytes(bytes(wqe[0:8]), 'big'))

# map MLX5 UAR and DBR into GPU VA
uar_paddr = dev.pci_dev.bar_info(0)[0] + dev.uar * 0x1000
uar_gpu_va = map_phys_to_gpu(gpu, uar_paddr, 0x1000)
dbr_gpu_va = map_phys_to_gpu(gpu, dev.dbr_paddrs[0], 0x1000)
print(f"UAR gpu_va=0x{uar_gpu_va:x} DBR gpu_va=0x{dbr_gpu_va:x}")

# GPU rings send doorbell via compute queue release_mem
q = AMDComputeQueue(gpu)
q.wait(gpu.timeline_signal, gpu.timeline_value - 1)
# write DBR (32-bit sq_head) - send doorbell at qp_dbr + 4
q.release_mem(dbr_gpu_va + qp.qp_dbr + 4, to_be32(qp.sq_head), q.pm4.data_sel__mec_release_mem__send_32_bit_low,
              q.pm4.int_sel__mec_release_mem__none)
# write UAR doorbell (64-bit)
q.release_mem(uar_gpu_va + 0x800, doorbell_val, q.pm4.data_sel__mec_release_mem__send_64_bit_data,
              q.pm4.int_sel__mec_release_mem__none)
q.signal(gpu.timeline_signal, gpu.next_timeline())
q.submit(gpu)

print("GPU kicked doorbell, waiting...")
gpu.synchronize()

# poll CQ from CPU (send + recv completions)
qp.poll_cq()
qp.poll_cq()

# read back
result = bytearray(BUF_SIZE)
gpu.allocator._copyout(memoryview(result), buf_dst)
gpu.synchronize()

got = bytes(result[:len(test_msg)])
print(f"result: {got}")
assert got == test_msg, f"MISMATCH: {got} != {test_msg}"
print("RDMA loopback send/recv test passed (GPU-kicked)")
