# include directory copied from https://github.com/HazyResearch/ThunderMittens
# https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx

gemm = """
#include <metal_stdlib>
#include "include/tk.metal"
using namespace mittens;

#define GEMM_PARAMS_DEF(T) \
    device T* D [[buffer(0)]], \
    device T* A [[buffer(1)]], \
    device T* B [[buffer(2)]], \
    const constant int &N [[buffer(3)]], \
    const constant int &K [[buffer(4)]], \
    const constant int &M [[buffer(5)]], \
    uint3 tg_id [[threadgroup_position_in_grid]], \
    uint simd_lane_id [[thread_index_in_simdgroup]]

template<typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
kernel void matmul_naive(GEMM_PARAMS_DEF(T)) {
  using global_layout = gl<T, 1, 1, -1, -1>;
  global_layout gl_a(A, nullptr, nullptr, N, K);
  global_layout gl_b(B, nullptr, nullptr, K, M);
  global_layout gl_d(D, nullptr, nullptr, N, M);
  rt<T,     N_BLOCK * TILE_DIM, K_BLOCK * TILE_DIM> a_reg;
  rt<T,     K_BLOCK * TILE_DIM, M_BLOCK * TILE_DIM> b_reg;
  rt<float, N_BLOCK * TILE_DIM, M_BLOCK * TILE_DIM> d_reg;
  zero(d_reg);
  #pragma clang loop unroll(full)
  for (int k = 0; k < K / (K_BLOCK * TILE_DIM); k++) {
    load(a_reg, gl_a, {0, 0, (int)tg_id.y, k}, simd_lane_id);
    load(b_reg, gl_b, {0, 0, k, (int)tg_id.x}, simd_lane_id);
    mma_AB(d_reg, a_reg, b_reg, d_reg);
  }
  store(gl_d, d_reg, {0, 0, (int)tg_id.y, (int)tg_id.x}, simd_lane_id);
}

#define instantiate_matmul_custom(type_name, T) \
   template [[host_name("matmul_custom_" #type_name)]] [[kernel]] \
   void matmul_naive<T, 4, 2, 4>(GEMM_PARAMS_DEF(T)); \

instantiate_matmul_custom(float32, float);
"""

from tinygrad import Device, Tensor, Context

if __name__ == "__main__":
  device = Device["METAL"]
  lib = device.compiler.compile(gemm)
  prg = device.runtime("matmul_custom_float32", lib)

  N = 4096
  a = Tensor.randn(N, N)
  b = Tensor.randn(N, N)
  c = Tensor.empty(N, N)
  Tensor.realize(a, b, c)

  TILE_DIM = 8
  N_BLOCK = 4
  M_BLOCK = 4

  gsz = (N // (M_BLOCK * TILE_DIM), N // (N_BLOCK * TILE_DIM), 1)
  for _ in range(5):
    et = prg(c.uop.buffer.ensure_allocated()._buf, a.uop.buffer._buf, b.uop.buffer._buf,
            global_size=gsz, local_size=(32,1,1), vals=(N, N, N), wait=True)
    print(f"{N*N*N*2/(et*1e9):2f} GFLOPS")

  for _ in range(5):
    with Context(DEBUG=2):
      ref = (a@b).realize()

  print((ref-c).mean().item())


