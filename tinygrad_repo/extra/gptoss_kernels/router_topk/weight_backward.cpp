#include "kittens.cuh"

using namespace kittens;

constexpr int NUM_WARPS = 16;
constexpr int THREADS = NUM_WARPS * 64;
constexpr int TILE = 16;
constexpr int PREFETCH = 8;
using float4_t = float __attribute__((ext_vector_type(4)));

static_assert(ROUTER_K % TILE == 0);
static_assert(ROUTER_M % (4 * NUM_WARPS * PREFETCH) == 0);

// Each block computes 16 features for all 32 experts, retaining FP32 logit gradients and accumulation.
extern "C" __global__ __launch_bounds__(THREADS, 2) void moe_router_fp32_wgrad(
    bf16 *__restrict__ out, const bf16 *__restrict__ x, const float *__restrict__ gradient) {
  const int lane = laneid();
  const int warp = threadIdx.x / 64;
  const int feature_base = blockIdx.x * TILE;
  const int k_lane = lane / TILE;
  const int mn_lane = lane % TILE;

  float4_t accum[2] = {};
  #pragma unroll 1
  for (int mt = warp; mt < ROUTER_M / 4; mt += NUM_WARPS * PREFETCH) {
    float av[PREFETCH], bv0[PREFETCH], bv1[PREFETCH];
    #pragma unroll
    for (int p = 0; p < PREFETCH; p++) {
      const int token = (mt + p * NUM_WARPS) * 4 + k_lane;
      av[p] = (float)x[(long long)token * ROUTER_K + feature_base + mn_lane];
      bv0[p] = gradient[(long long)token * 32 + mn_lane];
      bv1[p] = gradient[(long long)token * 32 + TILE + mn_lane];
    }
    // Keep the prefetched operands live together instead of serializing load/MFMA pairs.
    asm volatile("" :: "v"(av[0]), "v"(bv0[0]), "v"(bv1[0]), "v"(av[1]), "v"(bv0[1]), "v"(bv1[1]),
                       "v"(av[2]), "v"(bv0[2]), "v"(bv1[2]), "v"(av[3]), "v"(bv0[3]), "v"(bv1[3]),
                       "v"(av[4]), "v"(bv0[4]), "v"(bv1[4]), "v"(av[5]), "v"(bv0[5]), "v"(bv1[5]),
                       "v"(av[6]), "v"(bv0[6]), "v"(bv1[6]), "v"(av[7]), "v"(bv0[7]), "v"(bv1[7]) : "memory");
    #pragma unroll
    for (int p = 0; p < PREFETCH; p++) {
      accum[0] = __builtin_amdgcn_mfma_f32_16x16x4f32(av[p], bv0[p], accum[0], 0, 0, 0);
      accum[1] = __builtin_amdgcn_mfma_f32_16x16x4f32(av[p], bv1[p], accum[1], 0, 0, 0);
    }
  }

  __shared__ float partial[2][NUM_WARPS][64][4];
  #pragma unroll
  for (int et = 0; et < 2; et++) {
    #pragma unroll
    for (int i = 0; i < 4; i++) partial[et][warp][lane][i] = accum[et][i];
  }
  __syncthreads();

  if (warp == 0) {
    #pragma unroll
    for (int et = 0; et < 2; et++) {
      float4_t total = {};
      #pragma unroll
      for (int w = 0; w < NUM_WARPS; w++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) total[i] += partial[et][w][lane][i];
      }
      const int expert = et * TILE + lane % 16;
      const int feature0 = feature_base + 4 * (lane / 16);
      #pragma unroll
      for (int i = 0; i < 4; i++) out[(long long)expert * ROUTER_K + feature0 + i] = (bf16)total[i];
    }
  }
}
