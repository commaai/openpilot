#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// One workgroup computes all four routing-weight gradients, reusing dout across expert slots.
extern "C" __global__ __launch_bounds__(256) void gptoss_combine_weights_backward(
    float *__restrict__ dw, const __hip_bfloat16 *__restrict__ z,
    const __hip_bfloat16 *__restrict__ dout, const int *__restrict__ dest_row) {
  const int token = blockIdx.x, group = token / TOKENS, tid = threadIdx.x;
  float sums[4] = {};
  int rows[4];
#pragma unroll
  for (int j = 0; j < 4; j++) rows[j] = group * ROWS + dest_row[token * 4 + j];

  for (int d = tid; d < 2880; d += 256) {
    const float dy = (float)dout[(long long)token * 2880 + d];
#pragma unroll
    for (int j = 0; j < 4; j++)
      sums[j] = fmaf(dy, (float)z[(long long)rows[j] * 2880 + d], sums[j]);
  }

  // Reduce each slot across the four wavefronts.
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1)
#pragma unroll
    for (int j = 0; j < 4; j++) sums[j] += __shfl_down(sums[j], offset, 64);

  __shared__ float warp_sums[4][4];
  const int lane = tid % 64, warp = tid / 64;
#pragma unroll
  for (int j = 0; j < 4; j++) if (lane == 0) warp_sums[warp][j] = sums[j];
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int j = 0; j < 4; j++) sums[j] = lane < 4 ? warp_sums[lane][j] : 0.0f;
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
#pragma unroll
      for (int j = 0; j < 4; j++) sums[j] += __shfl_down(sums[j], offset, 64);
    if (lane == 0)
#pragma unroll
      for (int j = 0; j < 4; j++) dw[token * 4 + j] = sums[j];
  }
}
