#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(256) void moe_router_topk(
    float *__restrict__ weights, int *__restrict__ indices, const float *__restrict__ logits) {
  const int token = blockIdx.x * 256 + threadIdx.x;
  if (token >= TOKENS) return;
  float topv[4];
  int topi[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) { topv[i] = -3.402823466e+38f; topi[i] = -1; }

  const long long base = (long long)token * 32;
  #pragma unroll
  for (int e = 0; e < 32; e++) {
    float v = logits[base + e];
    int pos = 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) if (pos == 4 && v > topv[i]) pos = i;
    if (pos < 4) {
      #pragma unroll
      for (int i = 3; i > 0; i--) if (i > pos) { topv[i] = topv[i-1]; topi[i] = topi[i-1]; }
      topv[pos] = v;
      topi[pos] = e;
    }
  }

  float denom = 0.0f;
  float ex[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) { ex[i] = expf(topv[i] - topv[0]); denom += ex[i]; }
  const long long out = (long long)token * 4;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    weights[out + i] = ex[i] / denom;
    indices[out + i] = topi[i];
  }
}
