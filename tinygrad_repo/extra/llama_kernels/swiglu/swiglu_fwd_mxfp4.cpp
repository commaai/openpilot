#include "kittens.cuh"
#include "quantize_mxfp4_device.h"

using namespace kittens;

#if !defined(KERNEL_NAME) || !defined(M_DIM) || !defined(N_DIM)
#error kernel dimensions and name must be defined
#endif

namespace {
constexpr int M = M_DIM, N = N_DIM, INPUT_N = 2 * N;
constexpr int BLOCK = 32, TILE_M = 128, NUM_WARPS = 4, THREADS_PER_ROW = 8, VALUES_PER_THREAD = 4;
using Tile = st_bf<BLOCK, BLOCK, st_32x32_s>;
static_assert(M % TILE_M == 0 && N % BLOCK == 0);
__device__ __forceinline__ float sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }
__device__ __forceinline__ float bf16_to_float(uint16_t x) { return __uint_as_float(static_cast<uint32_t>(x) << 16); }
__device__ __forceinline__ uint16_t* tile_at(Tile& tile, int row, int col) {
  return reinterpret_cast<uint16_t*>(tile.data) + Tile::swizzle(make_int2(row, col)) / sizeof(bf16);
}
__device__ __forceinline__ float4 load_col4(Tile& tile, int row, int col) {
  return make_float4(bf16_to_float(*tile_at(tile, row + 0, col)), bf16_to_float(*tile_at(tile, row + 1, col)),
                     bf16_to_float(*tile_at(tile, row + 2, col)), bf16_to_float(*tile_at(tile, row + 3, col)));
}
}

extern "C" __global__ __launch_bounds__(256, 8)
void KERNEL_NAME(__hip_bfloat16* __restrict__ out,
                 uint8_t* __restrict__ row_fp4, uint8_t* __restrict__ row_scale,
                 uint8_t* __restrict__ col_fp4, uint8_t* __restrict__ col_scale,
                 const __hip_bfloat16* __restrict__ packed) {
  __shared__ Tile tiles[NUM_WARPS];
  const int warp = warpid(), lane = laneid(), line = lane / THREADS_PER_ROW, quant_lane = lane % THREADS_PER_ROW;
  const int block_m = blockIdx.x * TILE_M + warp * BLOCK, block_n = blockIdx.y * BLOCK;
  Tile& tile = tiles[warp];
  #pragma unroll
  for (int row_chunk = 0; row_chunk < BLOCK / THREADS_PER_ROW; row_chunk++) {
    const int local_row = row_chunk * THREADS_PER_ROW + line, row = block_m + local_row;
    const int local_col = quant_lane * VALUES_PER_THREAD, col = block_n + local_col;
    const uint64_t acts = *reinterpret_cast<const uint64_t*>(packed + row * INPUT_N + col);
    const uint64_t gates = *reinterpret_cast<const uint64_t*>(packed + row * INPUT_N + N + col);
    __hip_bfloat16 values[VALUES_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < VALUES_PER_THREAD; j++) {
      const float act = bf16_to_float(static_cast<uint16_t>(acts >> (16 * j)));
      const float gate = bf16_to_float(static_cast<uint16_t>(gates >> (16 * j)));
      values[j] = __hip_bfloat16((act * sigmoidf(act)) * gate);
    }
    *reinterpret_cast<uint64_t*>(tile_at(tile, local_row, local_col)) = *reinterpret_cast<uint64_t*>(values);
    const mxfp4::Quantized4 result = mxfp4::quantize_pipelined(make_float4(
      static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2]), static_cast<float>(values[3])), quant_lane);
    mxfp4::store_fp4<false>(row_fp4, row, col / 2, N / 2, result.fp4);
    if (quant_lane == 0) mxfp4::store_scale(row_scale, row, col / BLOCK, N / BLOCK, result.scale);
  }
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  #pragma unroll
  for (int col_chunk = 0; col_chunk < BLOCK / THREADS_PER_ROW; col_chunk++) {
    const int local_col = col_chunk * THREADS_PER_ROW + line, col = block_n + local_col;
    const int local_row = quant_lane * VALUES_PER_THREAD;
    const mxfp4::Quantized4 result = mxfp4::quantize_pipelined(load_col4(tile, local_row, local_col), quant_lane);
    mxfp4::store_fp4<true>(col_fp4, col, (block_m + local_row) / 2, M / 2, result.fp4);
    if (quant_lane == 0) mxfp4::store_scale(col_scale, col, block_m / BLOCK, M / BLOCK, result.scale);
  }
}

