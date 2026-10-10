// Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "quantize_mxfp4_device.h"

#if !defined(KERNEL_NAME) || !defined(M_DIM) || !defined(N_DIM) || !defined(SHUFFLE_ROWWISE_FP4_VALUE) || \
    !defined(SHUFFLE_COLWISE_FP4_VALUE)
#error kernel dimensions and layouts must be defined
#endif

namespace {

constexpr int BLOCK = 32;
constexpr int TILE_M = 128;
constexpr int TILE_N = 64;
constexpr int THREADS = 256;
constexpr int THREADS_PER_ROW = 8;
constexpr int VALUES_PER_THREAD = 4;
constexpr int SMEM_STRIDE = BLOCK + 2;
constexpr int M = M_DIM;
constexpr int N = N_DIM;
constexpr int M_PACKED = M / 2;
constexpr int N_PACKED = N / 2;
constexpr int M_SCALES = M / BLOCK;
constexpr int N_SCALES = N / BLOCK;
constexpr bool SHUFFLE_ROWWISE_FP4 = SHUFFLE_ROWWISE_FP4_VALUE;
constexpr bool SHUFFLE_COLWISE_FP4 = SHUFFLE_COLWISE_FP4_VALUE;

static_assert(M % 256 == 0 && N % 256 == 0);

__device__ __forceinline__ void load_tile(uint16_t* tile, const uint16_t* input, int tile_m, int tile_n) {
  const int row = threadIdx.x / THREADS_PER_ROW;
  const int col = threadIdx.x % THREADS_PER_ROW * VALUES_PER_THREAD;
  const uint64_t packed = *reinterpret_cast<const uint64_t*>(input + (tile_m + row) * N + tile_n + col);
  *reinterpret_cast<uint32_t*>(tile + row * SMEM_STRIDE + col) = static_cast<uint32_t>(packed);
  *reinterpret_cast<uint32_t*>(tile + row * SMEM_STRIDE + col + 2) = static_cast<uint32_t>(packed >> 32);
}

__device__ __forceinline__ void quantize_row(uint16_t* tile, uint8_t* fp4_output, uint8_t* scale_output,
                                             int tile_m, int tile_n, int local_row, int lane) {
  const int row = tile_m + local_row;
  const int col = lane * VALUES_PER_THREAD;
  const mxfp4::Quantized4 result = mxfp4::quantize(mxfp4::load_bf16x4(tile + local_row * SMEM_STRIDE + col), lane);
  mxfp4::store_fp4<SHUFFLE_ROWWISE_FP4>(fp4_output, row, (tile_n + col) / 2, N_PACKED, result.fp4);
  if (lane == 0) mxfp4::store_scale(scale_output, row, tile_n / BLOCK, N_SCALES, result.scale);
}

__device__ __forceinline__ mxfp4::Quantized4 quantize_col(uint16_t* tile, int col, int lane) {
  const int row = lane * VALUES_PER_THREAD;
  return mxfp4::quantize(make_float4(
    __uint_as_float(static_cast<uint32_t>(tile[(row + 0) * SMEM_STRIDE + col]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row + 1) * SMEM_STRIDE + col]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row + 2) * SMEM_STRIDE + col]) << 16),
    __uint_as_float(static_cast<uint32_t>(tile[(row + 3) * SMEM_STRIDE + col]) << 16)), lane);
}

} // namespace

extern "C" __global__ __launch_bounds__(THREADS, 8)
void KERNEL_NAME(uint8_t* __restrict__ rowwise_fp4, uint8_t* __restrict__ rowwise_scale,
                 uint8_t* __restrict__ colwise_fp4, uint8_t* __restrict__ colwise_scale,
                 const uint16_t* __restrict__ input) {
  __shared__ uint16_t tile[BLOCK * SMEM_STRIDE];
  const int tid = threadIdx.x;
  const int line = tid / THREADS_PER_ROW;
  const int lane = tid % THREADS_PER_ROW;
  const int block_m = blockIdx.x * TILE_M;
  const int block_n = blockIdx.y * TILE_N;

  if constexpr (!SHUFFLE_COLWISE_FP4) {
    uint16_t col_fp4[TILE_N / BLOCK][TILE_M / BLOCK];
    uint8_t col_scale[TILE_N / BLOCK][TILE_M / BLOCK];

    for (int chunk_m = 0; chunk_m < TILE_M / BLOCK; chunk_m++) {
      for (int chunk_n = 0; chunk_n < TILE_N / BLOCK; chunk_n++) {
        const int tile_m = block_m + chunk_m * BLOCK;
        const int tile_n = block_n + chunk_n * BLOCK;
        load_tile(tile, input, tile_m, tile_n);
        __syncthreads();

        quantize_row(tile, rowwise_fp4, rowwise_scale, tile_m, tile_n, line, lane);
        const mxfp4::Quantized4 result = quantize_col(tile, line, lane);
        col_fp4[chunk_n][chunk_m] = result.fp4;
        col_scale[chunk_n][chunk_m] = result.scale;
        __syncthreads();
      }
    }

    for (int chunk_n = 0; chunk_n < TILE_N / BLOCK; chunk_n++) {
      for (int chunk_m = 0; chunk_m < TILE_M / BLOCK; chunk_m++)
        tile[line * BLOCK + chunk_m * THREADS_PER_ROW + lane] = col_fp4[chunk_n][chunk_m];
      __syncthreads();

      for (int round = 0; round < BLOCK / THREADS_PER_ROW; round++) {
        const int col = round * THREADS_PER_ROW + tid / BLOCK;
        const int row_pair = tid % BLOCK;
        *reinterpret_cast<uint16_t*>(colwise_fp4 + (block_n + chunk_n * BLOCK + col) * M_PACKED + block_m / 2 + row_pair * 2) =
          tile[col * BLOCK + row_pair];
      }

      if (lane == 0) {
        const int col = block_n + chunk_n * BLOCK + line;
        for (int chunk_m = 0; chunk_m < TILE_M / BLOCK; chunk_m++)
          mxfp4::store_scale(colwise_scale, col, block_m / BLOCK + chunk_m, M_SCALES, col_scale[chunk_n][chunk_m]);
      }
      __syncthreads();
    }
  } else {
    for (int chunk_m = 0; chunk_m < TILE_M / BLOCK; chunk_m++) {
      for (int chunk_n = 0; chunk_n < TILE_N / BLOCK; chunk_n++) {
        const int tile_m = block_m + chunk_m * BLOCK;
        const int tile_n = block_n + chunk_n * BLOCK;
        load_tile(tile, input, tile_m, tile_n);
        __syncthreads();

        quantize_row(tile, rowwise_fp4, rowwise_scale, tile_m, tile_n, line, lane);
        const int row = lane * VALUES_PER_THREAD;
        const int col = tile_n + line;
        const mxfp4::Quantized4 result = quantize_col(tile, line, lane);
        mxfp4::store_fp4<true>(colwise_fp4, col, (tile_m + row) / 2, M_PACKED, result.fp4);
        if (lane == 0) mxfp4::store_scale(colwise_scale, col, tile_m / BLOCK, M_SCALES, result.scale);
        __syncthreads();
      }
    }
  }
}
