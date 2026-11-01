import numpy as np
import time, torch, torch.mps

from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import flat_mv
from tinygrad.runtime.ops_metal import MetalAllocator, MetalDevice, MetalProgram, MetalCompiler

N = 16384
M = 4096
FLOPS = N*M*2

nb = np.random.default_rng().standard_normal(size=(N), dtype=np.float32) #.astype(np.int32).astype(np.float32)
nc = np.random.default_rng().standard_normal(size=(N,M), dtype=np.float32) #.astype(np.int32).astype(np.float32)

b = torch.from_numpy(nb).to('mps')
c = torch.from_numpy(nc).to('mps')

def torch_prog(b, c):
  st = time.perf_counter()
  a = b@c
  torch.mps.synchronize()
  return time.perf_counter() - st
tm = min([torch_prog(b, c) for _ in range(200)])
print(f"{N:d}x{M:d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matvec in torch")
torch_a = (b@c).cpu()

device = MetalDevice("METAL")
metalalloc = MetalAllocator(device)
WORKSIZE_ROW = 16
WORKSIZE_COL = 1
LOCAL_SIZE = [32, WORKSIZE_COL, WORKSIZE_ROW]
GLOBAL_SIZE = [M//(LOCAL_SIZE[0]*LOCAL_SIZE[1]*4), 1, 1]
prog = MetalProgram(device, "test", MetalCompiler().compile(f"""
#include <metal_stdlib>
using namespace metal;
kernel void test(device float* data0, const device float* data1, const device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  int gidx0 = gid.x;  /* {GLOBAL_SIZE[0]} */
  int lidx1 = lid.x;  /* {LOCAL_SIZE[0]} */
  int lidx2 = lid.y;  /* {LOCAL_SIZE[1]} */
  int lidx3 = lid.z;  /* {LOCAL_SIZE[2]} */

  // 4 rows per thread
  threadgroup float4 acc0[{LOCAL_SIZE[0]*LOCAL_SIZE[1]*LOCAL_SIZE[2]}];
  int acc0_index = ((lidx1*{LOCAL_SIZE[1]})+lidx2)+({LOCAL_SIZE[0]*LOCAL_SIZE[1]}*lidx3);
  acc0[acc0_index] = float4(0.0f,0.0f,0.0f,0.0f);

  threadgroup float4 val1[{LOCAL_SIZE[0]*LOCAL_SIZE[1]*LOCAL_SIZE[2]}];

  // iterate over the columns
  for (int ridx2 = 0; ridx2 < {N//(4*LOCAL_SIZE[0]*LOCAL_SIZE[1]*(LOCAL_SIZE[2]))}; ++ridx2) {{
    // load 4*threadgroup_size columns into shared memory
    int col_1 = (((lidx3*{N//(4*LOCAL_SIZE[0]*LOCAL_SIZE[1]*(LOCAL_SIZE[2]))})+ridx2)*{LOCAL_SIZE[0]*LOCAL_SIZE[1]})+(lidx1*{LOCAL_SIZE[1]})+lidx2;
    val1[(lidx3*{LOCAL_SIZE[1]*LOCAL_SIZE[0]})+((lidx1*{LOCAL_SIZE[1]})+lidx2)] = *((device float4*)(data1+(col_1*4)));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ridx3 = 0; ridx3 < {LOCAL_SIZE[0]*LOCAL_SIZE[1]}; ++ridx3) {{
      int col = ((((lidx3*{N//(4*LOCAL_SIZE[0]*LOCAL_SIZE[1]*(LOCAL_SIZE[2]))})+ridx2)*{LOCAL_SIZE[0]*LOCAL_SIZE[1]})+ridx3);
      float4 val1_0 = val1[(lidx3*{LOCAL_SIZE[1]*LOCAL_SIZE[0]})+ridx3];
      float4 val2_0 = (float4)(*((device float4*)(data2+(gidx0*{M//GLOBAL_SIZE[0]})+(((lidx1*{LOCAL_SIZE[1]})+lidx2)*4)+(col*{M*4})+{M*0})));
      float4 val2_1 = (float4)(*((device float4*)(data2+(gidx0*{M//GLOBAL_SIZE[0]})+(((lidx1*{LOCAL_SIZE[1]})+lidx2)*4)+(col*{M*4})+{M*1})));
      float4 val2_2 = (float4)(*((device float4*)(data2+(gidx0*{M//GLOBAL_SIZE[0]})+(((lidx1*{LOCAL_SIZE[1]})+lidx2)*4)+(col*{M*4})+{M*2})));
      float4 val2_3 = (float4)(*((device float4*)(data2+(gidx0*{M//GLOBAL_SIZE[0]})+(((lidx1*{LOCAL_SIZE[1]})+lidx2)*4)+(col*{M*4})+{M*3})));
      acc0[acc0_index] = ((val1_0.x*val2_0)+acc0[acc0_index]);
      acc0[acc0_index] = ((val1_0.y*val2_1)+acc0[acc0_index]);
      acc0[acc0_index] = ((val1_0.z*val2_2)+acc0[acc0_index]);
      acc0[acc0_index] = ((val1_0.w*val2_3)+acc0[acc0_index]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }} /* reduce */

  if (lidx3 == 0) {{
    float4 out = float4(0.0f,0.0f,0.0f,0.0f);
    for (int n = 0; n < {LOCAL_SIZE[2]}; n++) {{
      out += acc0[((lidx1*{LOCAL_SIZE[1]})+lidx2)+({LOCAL_SIZE[0]*LOCAL_SIZE[1]}*n)];
    }}
    *( (device float4 *) (data0 + (gidx0*{M//GLOBAL_SIZE[0]}) + ( ( (lidx1*{LOCAL_SIZE[1]})+lidx2 ) * 4 ) ) ) = out;
  }}
}}
"""))
a = metalalloc.alloc(M*4)
b = metalalloc.alloc(N*4)
c = metalalloc.alloc(N*M*4)
metalalloc._copyin(b,nb.tobytes())
metalalloc._copyin(c,nc.tobytes())
def metalrun():
  prog(a, b, c, global_size=GLOBAL_SIZE, local_size=LOCAL_SIZE, wait=True)
  return a
def timeit(fxn):
  st = time.perf_counter()
  et = fxn()
  # NOTE: et doesn't contain the launch overhead
  return time.perf_counter() - st
tm = min([timeit(metalrun) for _ in range(200)])
print(f"{N:d}x{M:d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matvec in metal")
metal_a = np.zeros(M, dtype=np.float32)
metalalloc._copyout(flat_mv(metal_a.data), a)
np.testing.assert_allclose(metal_a, torch_a, atol=5e-3)

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
tm = min([tiny_prog(b, c) for _ in range(200)])
print(f"{N:d}x{M:d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matvec in tinygrad")
tiny_a = tiny_jit(b, c).numpy()
np.testing.assert_allclose(tiny_a, torch_a, atol=5e-3)