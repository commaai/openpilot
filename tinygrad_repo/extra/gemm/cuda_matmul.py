import os
import numpy as np
os.environ["CUDA"] = "1"
from tinygrad.runtime.ops_cuda import CUDAAllocator, CUDADevice, CUDAProgram, CUDACompiler
from tinygrad.helpers import flat_mv

FLOAT16 = True
ACC_FLOAT16 = False
N = 4096

na = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32)
nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32)
nc = np.empty(N*N, np.float32)

if FLOAT16:
  na = na.astype(np.float16)
  nb = nb.astype(np.float16)

device = CUDADevice("cuda:0")
cudaalloc = CUDAAllocator(device)

a = cudaalloc.alloc(N*N*2 if FLOAT16 else N*N*4)
b = cudaalloc.alloc(N*N*2 if FLOAT16 else N*N*4)
c = cudaalloc.alloc(N*N*4)

cudaalloc._copyin(a, bytearray(na))
cudaalloc._copyin(b, bytearray(nb))

FLOPS = N*N*N*2
BW = N*N*3*4

print(device.arch)
compiler = CUDACompiler(device.arch)
prog = CUDAProgram(device, "wmma_example", compiler.compile(f"""
#include <mma.h>
using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = {'16' if FLOAT16 else '8'};

extern "C" __global__ void wmma_example({'half' if FLOAT16 else 'float'} *a, {'half' if FLOAT16 else 'float'} *b, float *c)
{{
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  warpM *= 4;
  warpN *= 4;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, {'half' if FLOAT16 else 'wmma::precision::tf32'}, wmma::col_major> a_frag[4];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, {'half' if FLOAT16 else 'wmma::precision::tf32'}, wmma::col_major> b_frag[4];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, {'half' if ACC_FLOAT16 else 'float'}> acc_frag[4][4];
  for (int j = 0; j < 4; j++) {{
    for (int i = 0; i < 4; i++) {{
      wmma::fill_fragment(acc_frag[i][j], 0.0f);
    }}
  }}

  for (int k = 0; k < {N}; k += WMMA_K) {{
    int aRow = warpM * WMMA_M;
    int aCol = k;
    int bRow = k;
    int bCol = warpN * WMMA_N;

    wmma::load_matrix_sync(a_frag[0], a + aRow + 0 * WMMA_M + aCol * {N}, {N});
    wmma::load_matrix_sync(a_frag[1], a + aRow + 1 * WMMA_M + aCol * {N}, {N});
    wmma::load_matrix_sync(a_frag[2], a + aRow + 2 * WMMA_M + aCol * {N}, {N});
    wmma::load_matrix_sync(a_frag[3], a + aRow + 3 * WMMA_M + aCol * {N}, {N});

    wmma::load_matrix_sync(b_frag[0], b + bRow + (0 * WMMA_N + bCol) * {N}, {N});
    wmma::load_matrix_sync(b_frag[1], b + bRow + (1 * WMMA_N + bCol) * {N}, {N});
    wmma::load_matrix_sync(b_frag[2], b + bRow + (2 * WMMA_N + bCol) * {N}, {N});
    wmma::load_matrix_sync(b_frag[3], b + bRow + (3 * WMMA_N + bCol) * {N}, {N});

    #pragma unroll
    for (int i = 0; i < {'0' if FLOAT16 else '4'}; i++) {{
      #pragma unroll
      for (int t = 0; t < a_frag[i].num_elements; t++) {{ a_frag[i].x[t] =  wmma::__float_to_tf32(a_frag[i].x[t]); }}
      #pragma unroll
      for (int t = 0; t < b_frag[i].num_elements; t++) {{ b_frag[i].x[t] =  wmma::__float_to_tf32(b_frag[i].x[t]); }}
    }}

    #pragma unroll
    for (int j = 0; j < 4; j++) {{
      #pragma unroll
      for (int i = 0; i < 4; i++) {{
        wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
      }}
    }}
  }}

  for (int j = 0; j < 4; j++) {{
    for (int i = 0; i < 4; i++) {{
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_store;
      for (int t = 0; t < acc_frag[i][j].num_elements; t++) acc_store.x[t] = acc_frag[i][j].x[t];
      int cRow = (warpM + i) * WMMA_M;
      int cCol = (warpN + j) * WMMA_N;
      wmma::store_matrix_sync(c + cRow + cCol * {N}, acc_store, {N}, wmma::mem_col_major);
    }}
  }}
}}
"""))

global_size, local_size = [(N//16)//4, (N//16)//4, 1], [32, 1, 1]
tm = min([prog(a, b, c, global_size=global_size, local_size=local_size, wait=True) for _ in range(20)])
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s")
cudaalloc._copyout(flat_mv(nc.data), c)
np.testing.assert_allclose(na.T.astype(np.float32) @ nb.T.astype(np.float32), nc.reshape(N,N).T, atol=1e-2)