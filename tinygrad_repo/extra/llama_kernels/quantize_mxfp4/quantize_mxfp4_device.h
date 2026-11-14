#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace mxfp4 {

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

// Issue all four independent lane swizzles together so the LDS pipeline latency is paid once.
__device__ __forceinline__ float4 swizzle4_xor1(float4 value) {
  float4 result;
  asm volatile("ds_swizzle_b32 %0, %4 offset:0x041f\n\t"
               "ds_swizzle_b32 %1, %5 offset:0x041f\n\t"
               "ds_swizzle_b32 %2, %6 offset:0x041f\n\t"
               "ds_swizzle_b32 %3, %7 offset:0x041f\n\t"
               "s_waitcnt lgkmcnt(0)"
               : "=v"(result.x), "=v"(result.y), "=v"(result.z), "=v"(result.w)
               : "v"(value.x), "v"(value.y), "v"(value.z), "v"(value.w));
  return result;
}

__device__ __forceinline__ float4 swizzle4_xor2(float4 value) {
  float4 result;
  asm volatile("ds_swizzle_b32 %0, %4 offset:0x081f\n\t"
               "ds_swizzle_b32 %1, %5 offset:0x081f\n\t"
               "ds_swizzle_b32 %2, %6 offset:0x081f\n\t"
               "ds_swizzle_b32 %3, %7 offset:0x081f\n\t"
               "s_waitcnt lgkmcnt(0)"
               : "=v"(result.x), "=v"(result.y), "=v"(result.z), "=v"(result.w)
               : "v"(value.x), "v"(value.y), "v"(value.z), "v"(value.w));
  return result;
}

template<int Offset>
__device__ __forceinline__ void swizzle4_pair(float4 a, float4 b, float4& a_out, float4& b_out) {
  asm volatile("ds_swizzle_b32 %0, %8 offset:%16\n\t"
               "ds_swizzle_b32 %1, %9 offset:%16\n\t"
               "ds_swizzle_b32 %2, %10 offset:%16\n\t"
               "ds_swizzle_b32 %3, %11 offset:%16\n\t"
               "ds_swizzle_b32 %4, %12 offset:%16\n\t"
               "ds_swizzle_b32 %5, %13 offset:%16\n\t"
               "ds_swizzle_b32 %6, %14 offset:%16\n\t"
               "ds_swizzle_b32 %7, %15 offset:%16\n\t"
               "s_waitcnt lgkmcnt(0)"
               : "=v"(a_out.x), "=v"(a_out.y), "=v"(a_out.z), "=v"(a_out.w),
                 "=v"(b_out.x), "=v"(b_out.y), "=v"(b_out.z), "=v"(b_out.w)
               : "v"(a.x), "v"(a.y), "v"(a.z), "v"(a.w), "v"(b.x), "v"(b.y), "v"(b.z), "v"(b.w), "n"(Offset));
}

template<int Offset>
__device__ __forceinline__ void swizzle_pair(float a, float b, float& a_out, float& b_out) {
  asm volatile("ds_swizzle_b32 %0, %2 offset:%4\n\t"
               "ds_swizzle_b32 %1, %3 offset:%4\n\t"
               "s_waitcnt lgkmcnt(0)"
               : "=v"(a_out), "=v"(b_out) : "v"(a), "v"(b), "n"(Offset));
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

__device__ __forceinline__ void hadamard16_pipelined(float4& value, int lane) {
  const float a0 = value.x + value.y, a1 = value.x - value.y;
  const float a2 = value.z + value.w, a3 = value.z - value.w;
  value = make_float4(a0 + a2, a1 + a3, a0 - a2, a1 - a3);

  const float4 xor1 = swizzle4_xor1(value);
  value = lane & 1 ? make_float4(xor1.x - value.x, xor1.y - value.y, xor1.z - value.z, xor1.w - value.w)
                   : make_float4(xor1.x + value.x, xor1.y + value.y, xor1.z + value.z, xor1.w + value.w);

  const float4 xor2 = swizzle4_xor2(value);
  value = lane & 2 ? make_float4(xor2.x - value.x, xor2.y - value.y, xor2.z - value.z, xor2.w - value.w)
                   : make_float4(xor2.x + value.x, xor2.y + value.y, xor2.z + value.z, xor2.w + value.w);
  value.x *= 0.25f;
  value.y *= 0.25f;
  value.z *= 0.25f;
  value.w *= 0.25f;
}

__device__ __forceinline__ void hadamard16_pair(float4& a, float4& b, int lane) {
  const float aa0 = a.x + a.y, aa1 = a.x - a.y, aa2 = a.z + a.w, aa3 = a.z - a.w;
  const float ba0 = b.x + b.y, ba1 = b.x - b.y, ba2 = b.z + b.w, ba3 = b.z - b.w;
  a = make_float4(aa0 + aa2, aa1 + aa3, aa0 - aa2, aa1 - aa3);
  b = make_float4(ba0 + ba2, ba1 + ba3, ba0 - ba2, ba1 - ba3);

  float4 a_xor, b_xor;
  swizzle4_pair<0x041f>(a, b, a_xor, b_xor);
  a = lane & 1 ? make_float4(a_xor.x - a.x, a_xor.y - a.y, a_xor.z - a.z, a_xor.w - a.w)
               : make_float4(a_xor.x + a.x, a_xor.y + a.y, a_xor.z + a.z, a_xor.w + a.w);
  b = lane & 1 ? make_float4(b_xor.x - b.x, b_xor.y - b.y, b_xor.z - b.z, b_xor.w - b.w)
               : make_float4(b_xor.x + b.x, b_xor.y + b.y, b_xor.z + b.z, b_xor.w + b.w);

  swizzle4_pair<0x081f>(a, b, a_xor, b_xor);
  a = lane & 2 ? make_float4(a_xor.x - a.x, a_xor.y - a.y, a_xor.z - a.z, a_xor.w - a.w)
               : make_float4(a_xor.x + a.x, a_xor.y + a.y, a_xor.z + a.z, a_xor.w + a.w);
  b = lane & 2 ? make_float4(b_xor.x - b.x, b_xor.y - b.y, b_xor.z - b.z, b_xor.w - b.w)
               : make_float4(b_xor.x + b.x, b_xor.y + b.y, b_xor.z + b.z, b_xor.w + b.w);
  a.x *= 0.25f; a.y *= 0.25f; a.z *= 0.25f; a.w *= 0.25f;
  b.x *= 0.25f; b.y *= 0.25f; b.z *= 0.25f; b.w *= 0.25f;
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

__device__ __forceinline__ Quantized4 quantize_pipelined(float4 value, int lane) {
  hadamard16_pipelined(value, lane);
  const float local_max = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w)));
  float scale;
  const uint8_t e8m0 = e8m0_scale(max8(local_max), scale);
  return {pack_fp4(value, scale), e8m0};
}

__device__ __forceinline__ void quantize_pair(float4 a, float4 b, int lane, Quantized4& a_result, Quantized4& b_result) {
  hadamard16_pair(a, b, lane);
  float amax = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w)));
  float bmax = fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w)));
  float a_other, b_other;
  swizzle_pair<0x101f>(amax, bmax, a_other, b_other); amax = fmaxf(amax, a_other); bmax = fmaxf(bmax, b_other);
  swizzle_pair<0x081f>(amax, bmax, a_other, b_other); amax = fmaxf(amax, a_other); bmax = fmaxf(bmax, b_other);
  swizzle_pair<0x041f>(amax, bmax, a_other, b_other); amax = fmaxf(amax, a_other); bmax = fmaxf(bmax, b_other);
  float a_scale, b_scale;
  const uint8_t a_e8m0 = e8m0_scale(amax, a_scale);
  const uint8_t b_e8m0 = e8m0_scale(bmax, b_scale);
  a_result = {pack_fp4(a, a_scale), a_e8m0};
  b_result = {pack_fp4(b, b_scale), b_e8m0};
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

} // namespace mxfp4
