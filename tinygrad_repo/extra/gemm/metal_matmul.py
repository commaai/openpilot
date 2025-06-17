import os
os.environ["METAL"] = "1"
import time
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.helpers import getenv, flat_mv
from tinygrad.runtime.ops_metal import MetalAllocator, MetalDevice, MetalProgram, MetalCompiler

N = getenv("N", 2048)
LID = 2

device = MetalDevice("METAL")
metalalloc = MetalAllocator(device)

a = metalalloc.alloc(N*N*4)
b = metalalloc.alloc(N*N*4)
c = metalalloc.alloc(N*N*4)

na = np.zeros((N,N),dtype=np.float32)
nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32) #.astype(np.int32).astype(np.float32)N
nc = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32) #.astype(np.int32).astype(np.float32)

metalalloc._copyin(b,nb.tobytes())
metalalloc._copyin(c,nc.tobytes())

FLOPS = N*N*N*2
BW = N*N*3*4

prog = MetalProgram(device, "test", MetalCompiler(device).compile(f"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+
using namespace metal;
kernel void test(device float *a, device const float *data1, device const float *data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  a += gid.x * 32 * {N} + (gid.y * {LID} + lid.y) * 32;
  data1 += gid.x * 32 * {N};
  data2 += (gid.y * {LID} + lid.y) * 32;

  simdgroup_float8x8 acc[4][4];
  for (uint i = 0; i < 4; i++) {{
    for (uint j = 0; j < 4; j++) {{
      acc[i][j] = simdgroup_float8x8(0);
    }}
  }}

  simdgroup_float8x8 A[4];
  simdgroup_float8x8 B[4];
  for (uint k = 0; k < {N}; k+=8) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(A[0], data1+k+{0*N}, {N}, ulong2(0, 0));
    simdgroup_load(A[1], data1+k+{8*N}, {N}, ulong2(0, 0));
    simdgroup_load(A[2], data1+k+{16*N}, {N}, ulong2(0, 0));
    simdgroup_load(A[3], data1+k+{24*N}, {N}, ulong2(0, 0));
    simdgroup_load(B[0], data2+0+k*{N}, {N}, ulong2(0, 0));
    simdgroup_load(B[1], data2+8+k*{N}, {N}, ulong2(0, 0));
    simdgroup_load(B[2], data2+16+k*{N}, {N}, ulong2(0, 0));
    simdgroup_load(B[3], data2+24+k*{N}, {N}, ulong2(0, 0));

    simdgroup_multiply_accumulate(acc[0][0], A[0], B[0], acc[0][0]);
    simdgroup_multiply_accumulate(acc[0][1], A[1], B[0], acc[0][1]);
    simdgroup_multiply_accumulate(acc[0][2], A[2], B[0], acc[0][2]);
    simdgroup_multiply_accumulate(acc[0][3], A[3], B[0], acc[0][3]);
    simdgroup_multiply_accumulate(acc[1][0], A[0], B[1], acc[1][0]);
    simdgroup_multiply_accumulate(acc[1][1], A[1], B[1], acc[1][1]);
    simdgroup_multiply_accumulate(acc[1][2], A[2], B[1], acc[1][2]);
    simdgroup_multiply_accumulate(acc[1][3], A[3], B[1], acc[1][3]);
    simdgroup_multiply_accumulate(acc[2][0], A[0], B[2], acc[2][0]);
    simdgroup_multiply_accumulate(acc[2][1], A[1], B[2], acc[2][1]);
    simdgroup_multiply_accumulate(acc[2][2], A[2], B[2], acc[2][2]);
    simdgroup_multiply_accumulate(acc[2][3], A[3], B[2], acc[2][3]);
    simdgroup_multiply_accumulate(acc[3][0], A[0], B[3], acc[3][0]);
    simdgroup_multiply_accumulate(acc[3][1], A[1], B[3], acc[3][1]);
    simdgroup_multiply_accumulate(acc[3][2], A[2], B[3], acc[3][2]);
    simdgroup_multiply_accumulate(acc[3][3], A[3], B[3], acc[3][3]);
  }}
  simdgroup_store(acc[0][0], a+{0+0*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[1][0], a+{8+0*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[2][0], a+{16+0*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[3][0], a+{24+0*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[0][1], a+{0+8*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[1][1], a+{8+8*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[2][1], a+{16+8*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[3][1], a+{24+8*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[0][2], a+{0+16*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[1][2], a+{8+16*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[2][2], a+{16+16*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[3][2], a+{24+16*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[0][3], a+{0+24*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[1][3], a+{8+24*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[2][3], a+{16+24*N}, {N}, ulong2(0, 0));
  simdgroup_store(acc[3][3], a+{24+24*N}, {N}, ulong2(0, 0));
}}"""))
def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  # NOTE: et doesn't contain the launch overhead
  return time.perf_counter() - st
tm = min([timeit(lambda: prog(a, b, c, global_size=[N//(8*4), N//(8*4*LID), 1], local_size=[32, LID, 1], wait=True)) for _ in range(20)])
comp = nb@nc
metalalloc._copyout(flat_mv(na.data), a)
if N <= 32:
  print(na)
  print(comp)
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
np.testing.assert_allclose(na, comp, atol=1e-3)

import torch, torch.mps
b = torch.from_numpy(nb).to('mps')
c = torch.from_numpy(nc).to('mps')

def torch_prog(b, c):
  st = time.perf_counter()
  a = b@c
  torch.mps.synchronize()
  return time.perf_counter() - st
tm = min([torch_prog(b, c) for _ in range(20)])
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul in torch")

from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
b = Tensor(nb)
c = Tensor(nc)
# TODO: slowness without the JIT I suspect comes from a lack of a caching allocator
@TinyJit
def tiny_jit(b, c):
  return (b@c).realize()
def tiny_prog(b, c):
  st = time.perf_counter()
  a = tiny_jit(b, c)
  Device["METAL"].synchronize()
  return time.perf_counter() - st
tm = min([tiny_prog(b, c) for _ in range(20)])
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul in tinygrad")
