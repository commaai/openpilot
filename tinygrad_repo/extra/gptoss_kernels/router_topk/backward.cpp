#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(256) void moe_router_topk_bwd(
    float *__restrict__ grad_logits, float *__restrict__ bias_partials, const float *__restrict__ grad_weights,
    const float *__restrict__ weights, const int *__restrict__ indices) {
  __shared__ float partial[4][32];
  if (threadIdx.x < 128) partial[threadIdx.x / 32][threadIdx.x % 32] = 0.0f;
  __syncthreads();
  const int token = blockIdx.x * 256 + threadIdx.x;
  if (token < TOKENS) {
    const long long in = (long long)token * 4;
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) dot += grad_weights[in + i] * weights[in + i];
    const long long out = (long long)token * 32;
    #pragma unroll
    for (int e = 0; e < 32; e++) grad_logits[out + e] = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      const int e = indices[in + i];
      const float dg = weights[in + i] * (grad_weights[in + i] - dot);
      grad_logits[out + e] = dg;
      atomicAdd(&partial[threadIdx.x / 64][e], dg);
    }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float sum = 0.0f;
    #pragma unroll
    for (int w = 0; w < 4; w++) sum += partial[w][threadIdx.x];
    bias_partials[(long long)blockIdx.x * 32 + threadIdx.x] = sum;
  }
}
