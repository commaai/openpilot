#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

typedef short short2v __attribute__((ext_vector_type(2)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));
typedef unsigned short ushort4v __attribute__((ext_vector_type(4)));

extern "C" __global__ __launch_bounds__(256) void gptoss_combine_input_quantize_backward(
    __hip_bfloat16 *__restrict__ out, unsigned char *__restrict__ q, unsigned char *__restrict__ scales,
    const __hip_bfloat16 *__restrict__ dout, const int *__restrict__ src_row, const float *__restrict__ weights) {
  const int row = blockIdx.x, group = row / ROWS, tid = threadIdx.x;
  // Four lanes own one MX block, with aligned eight-value loads and stores per lane.
  const int src = src_row[row], lane = tid % 4;
  for (int block = tid / 4; block < 96; block += 64) {
    const int d = block * 32 + lane * 8;
    const long long index = (long long)row * 3072 + d;
    // Neither padding nor unused rows may read dout or weights.
    if (src < 0 || d >= 2880) {
      *reinterpret_cast<uint4v*>(&out[index]) = uint4v{};
      *reinterpret_cast<ushort4v*>(&q[index]) = ushort4v{};
      if (lane == 0) scales[(long long)row * 96 + block] = 0;
      continue;
    }
    const int token = group * TOKENS + src / 4;
    const float weight = weights[token * 4 + src % 4];
    const uint4v input = *reinterpret_cast<const uint4v*>(&dout[(long long)token * 2880 + d]);
    uint4v output;
    float2 values[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const unsigned bits = input[i];
      const float2 dy = make_float2(__builtin_bit_cast(float, bits << 16), __builtin_bit_cast(float, bits & 0xffff0000u));
      const float2 value = __bfloat1622float2(__float22bfloat162_rn(make_float2(dy.x * weight, dy.y * weight)));
      values[i] = value;
      const __hip_bfloat162 rounded_pair = __float22bfloat162_rn(value);
      output[i] = (unsigned)__bfloat16_as_ushort(rounded_pair.x) | ((unsigned)__bfloat16_as_ushort(rounded_pair.y) << 16);
      amax = fmaxf(amax, fmaxf(fabsf(value.x), fabsf(value.y)));
    }
    *reinterpret_cast<uint4v*>(&out[index]) = output;
#pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_down(amax, offset, 4));
    amax = __shfl(amax, 0, 4);
    const unsigned e8 = min((__builtin_bit_cast(unsigned, amax) >> 23) & 255u, 254u);
    if (lane == 0) scales[(long long)row * 96 + block] = e8;
    ushort4v quantized;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      short2v packed = {0, 0};
      packed = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(packed, values[i].x, values[i].y, __builtin_bit_cast(float, e8 << 23), false);
      quantized[i] = __builtin_bit_cast(unsigned short, packed[0]);
    }
    *reinterpret_cast<ushort4v*>(&q[index]) = quantized;
  }
}
