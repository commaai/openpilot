// Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdint>

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

struct Quantized4 {
  uint16_t fp4;
  uint8_t scale;
};

__device__ __forceinline__ float swizzle_xor1(float value) {
  float result;
  asm volatile("ds_swizzle_b32 %0, %1 offset:0x041f\n\ts_waitcnt lgkmcnt(0)" : "=v"(result) : "v"(value));
  return result;
}

__device__ __forceinline__ float swizzle_xor2(float value) {
  float result;
  asm volatile("ds_swizzle_b32 %0, %1 offset:0x081f\n\ts_waitcnt lgkmcnt(0)" : "=v"(result) : "v"(value));
  return result;
}

__device__ __forceinline__ float swizzle_xor4(float value) {
  float result;
  asm volatile("ds_swizzle_b32 %0, %1 offset:0x101f\n\ts_waitcnt lgkmcnt(0)" : "=v"(result) : "v"(value));
  return result;
}

__device__ __forceinline__ float max8(float value) {
  value = fmaxf(value, swizzle_xor4(value));
  value = fmaxf(value, swizzle_xor2(value));
  return fmaxf(value, swizzle_xor1(value));
}

__device__ __forceinline__ float4 load_bf16x4(const uint16_t* values) {
  const uint32_t lo = *reinterpret_cast<const uint32_t*>(values);
  const uint32_t hi = *reinterpret_cast<const uint32_t*>(values + 2);
  return make_float4(__uint_as_float(lo << 16), __uint_as_float(lo & 0xffff0000u),
                     __uint_as_float(hi << 16), __uint_as_float(hi & 0xffff0000u));
}

__device__ __forceinline__ void hadamard16(float4& value, int lane) {
  const float a0 = value.x + value.y, a1 = value.x - value.y;
  const float a2 = value.z + value.w, a3 = value.z - value.w;
  value = make_float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);

  const float4 xor1 = make_float4(swizzle_xor1(value.x), swizzle_xor1(value.y), swizzle_xor1(value.z), swizzle_xor1(value.w));
  value = lane & 1 ? make_float4(xor1.x - value.x, xor1.y - value.y, xor1.z - value.z, xor1.w - value.w)
                   : make_float4(xor1.x + value.x, xor1.y + value.y, xor1.z + value.z, xor1.w + value.w);

  const float4 xor2 = make_float4(swizzle_xor2(value.x), swizzle_xor2(value.y), swizzle_xor2(value.z), swizzle_xor2(value.w));
  value = lane & 2 ? make_float4(xor2.x - value.x, xor2.y - value.y, xor2.z - value.z, xor2.w - value.w)
                   : make_float4(xor2.x + value.x, xor2.y + value.y, xor2.z + value.z, xor2.w + value.w);
  value.x *= 0.25f;
  value.y *= 0.25f;
  value.z *= 0.25f;
  value.w *= 0.25f;
}

__device__ __forceinline__ uint8_t e8m0_scale(float amax, float& scale) {
  if (amax == 0.0f) {
    scale = 1.0f;
    return 127;
  }

  const uint32_t rounded = (__float_as_uint(amax) + 0x200000u) & 0xff800000u;
  int exponent = static_cast<int>((rounded >> 23) & 0xff) - 129;
  exponent = exponent < -127 ? -127 : exponent > 127 ? 127 : exponent;
  scale = exponent == -127 ? __uint_as_float(0x00400000u) : __uint_as_float(static_cast<uint32_t>(exponent + 127) << 23);
  return static_cast<uint8_t>(exponent + 127);
}

__device__ __forceinline__ uint16_t pack_fp4(float4 value, float scale) {
  uint32_t lo = 0, hi = 0;
  asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3" : "+v"(lo) : "v"(value.x), "v"(value.y), "v"(scale));
  asm volatile("v_cvt_scalef32_pk_fp4_f32 %0, %1, %2, %3" : "+v"(hi) : "v"(value.z), "v"(value.w), "v"(scale));
  return static_cast<uint16_t>(lo | (hi << 8));
}

__device__ __forceinline__ Quantized4 quantize(float4 value, int lane) {
  hadamard16(value, lane);
  const float local_max = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w)));
  float scale;
  const uint8_t e8m0 = e8m0_scale(max8(local_max), scale);
  return {pack_fp4(value, scale), e8m0};
}

__device__ __forceinline__ void store_scale(uint8_t* output, int row, int col, int cols, uint8_t value) {
  const int tile = ((row >> 5) * (cols >> 3) + (col >> 3)) << 8;
  const int offset = ((col & 3) << 6) + ((row & 15) << 2) + (((col >> 2) & 1) << 1) + ((row >> 4) & 1);
  output[tile + offset] = value;
}

template<bool Shuffled>
__device__ __forceinline__ void store_fp4(uint8_t* output, int row, int col, int packed_cols, uint16_t value) {
  int index = row * packed_cols + col;
  if constexpr (Shuffled) {
    const int tile = (row >> 4) * (packed_cols << 4) + (col >> 5) * 512;
    const int offset = ((col >> 4) & 1) * 256 + (row & 15) * 16 + (col & 15);
    index = tile + offset;
  }
  *reinterpret_cast<uint16_t*>(output + index) = value;
}

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
  const Quantized4 result = quantize(load_bf16x4(tile + local_row * SMEM_STRIDE + col), lane);
  store_fp4<SHUFFLE_ROWWISE_FP4>(fp4_output, row, (tile_n + col) / 2, N_PACKED, result.fp4);
  if (lane == 0) store_scale(scale_output, row, tile_n / BLOCK, N_SCALES, result.scale);
}

__device__ __forceinline__ Quantized4 quantize_col(uint16_t* tile, int col, int lane) {
  const int row = lane * VALUES_PER_THREAD;
  return quantize(make_float4(
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
        const Quantized4 result = quantize_col(tile, line, lane);
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
          store_scale(colwise_scale, col, block_m / BLOCK + chunk_m, M_SCALES, col_scale[chunk_n][chunk_m]);
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
        const Quantized4 result = quantize_col(tile, line, lane);
        store_fp4<true>(colwise_fp4, col, (tile_m + row) / 2, M_PACKED, result.fp4);
        if (lane == 0) store_scale(colwise_scale, col, tile_m / BLOCK, M_SCALES, result.scale);
        __syncthreads();
      }
    }
  }
}
