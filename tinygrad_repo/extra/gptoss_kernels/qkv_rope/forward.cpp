#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef ATTN_B
#define ATTN_B 2
#endif
#ifndef ATTN_N
#define ATTN_N 8192
#endif
#ifndef ATTN_H
#define ATTN_H 64
#endif
#ifndef ATTN_H_KV
#define ATTN_H_KV 8
#endif
#ifndef ATTN_D
#define ATTN_D 64
#endif
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
constexpr int HALF_D = ATTN_D / 2;
constexpr int PACKED_D = (GROUP_SIZE + 2) * ATTN_D;
using float2v = float __attribute__((ext_vector_type(2)));
static_assert(ATTN_B == 2 && ATTN_N == 8192 && ATTN_H == 64 && ATTN_H_KV == 8 && ATTN_D == 64,
              "this kernel is the exact per-device GPTOSS QKV/RoPE shape");
static_assert(THREADS_PER_BLOCK == ATTN_H_KV * HALF_D, "one thread must own each KV-head/RoPE pair");

extern "C" __global__ __launch_bounds__(THREADS_PER_BLOCK) void
fused_qkv_rope_forward(
    __hip_bfloat16*       __restrict__ q,
    __hip_bfloat16*       __restrict__ k,
    __hip_bfloat16*       __restrict__ v,
    const __hip_bfloat16* __restrict__ xqkv,
    const __hip_bfloat16* __restrict__ freqs_cis) {
  const int b = blockIdx.x;
  const int n = blockIdx.y;
  const int bn = b * ATTN_N + n;
  const int kvh = threadIdx.x / HALF_D;
  const int pair = threadIdx.x % HALF_D;
  const int even = pair * 2;
  const int packed_head = bn * ATTN_H_KV * PACKED_D + kvh * PACKED_D;
  const int q_head = bn * ATTN_H * ATTN_D + kvh * GROUP_SIZE * ATTN_D;
  const int kv_head = (bn * ATTN_H_KV + kvh) * ATTN_D;

  // Cos/sin is shared by every Q/K slot owned by this (KV head, pair) thread.
  const __hip_bfloat16 *freq_pair = freqs_cis + (n * HALF_D + pair) * 2;
  const uint32_t cs_raw = *reinterpret_cast<const uint32_t*>(freq_pair);
  const float2 cs = __bfloat1622float2(*reinterpret_cast<const __hip_bfloat162*>(freq_pair));
  uint32_t ab_raw[GROUP_SIZE + 1];
  bool exceptional = (cs_raw & 0x7f80u) == 0x7f80u || (cs_raw & 0x7f800000u) == 0x7f800000u;
  #pragma unroll
  for (int slot = 0; slot < GROUP_SIZE + 1; slot++) {
    ab_raw[slot] = *reinterpret_cast<const uint32_t*>(xqkv + packed_head + slot * ATTN_D + even);
    exceptional |= (ab_raw[slot] & 0x7f80u) == 0x7f80u || (ab_raw[slot] & 0x7f800000u) == 0x7f800000u;
  }
  if (__builtin_expect(exceptional, 0)) {
    #pragma unroll
    for (int slot = 0; slot < GROUP_SIZE + 1; slot++) {
      const float2 ab = __bfloat1622float2(*reinterpret_cast<const __hip_bfloat162*>(&ab_raw[slot]));
      __hip_bfloat16 *dst = slot < GROUP_SIZE ? q : k;
      const int out = slot < GROUP_SIZE ? q_head + slot * ATTN_D + even : kv_head + even;
      // Match the original kernel's packed instruction association exactly for NaN/Inf values. The low-lane
      // instruction modifier, rather than pre-negating an operand, preserves its NaN sign/payload behavior.
      const float2v base = float2v{ab.x, ab.y} * float2v{cs.x, cs.x};
      const float2v cross = {ab.y, ab.x}, sine = {cs.y, cs.y};
      float2v rotated;
      asm volatile("v_pk_fma_f32 %0, %1, %2, %3 neg_lo:[1,0,0]"
                   : "=v"(rotated) : "v"(sine), "v"(cross), "v"(base));
      *reinterpret_cast<__hip_bfloat162*>(dst + out) =
          __float22bfloat162_rn(make_float2(rotated[0], rotated[1]));
    }
  } else {
    #pragma unroll
    for (int slot = 0; slot < GROUP_SIZE + 1; slot++) {
      const float2 ab = __bfloat1622float2(*reinterpret_cast<const __hip_bfloat162*>(&ab_raw[slot]));
      __hip_bfloat16 *dst = slot < GROUP_SIZE ? q : k;
      const int out = slot < GROUP_SIZE ? q_head + slot * ATTN_D + even : kv_head + even;
      *reinterpret_cast<__hip_bfloat162*>(dst + out) = __float22bfloat162_rn(
          make_float2(ab.x * cs.x - ab.y * cs.y, ab.x * cs.y + ab.y * cs.x));
    }
  }

  // V is already in output order; copy the aligned BF16 pair without conversion.
  *reinterpret_cast<__hip_bfloat162*>(v + kv_head + even) =
      *reinterpret_cast<const __hip_bfloat162*>(xqkv + packed_head + (GROUP_SIZE + 1) * ATTN_D + even);
}
