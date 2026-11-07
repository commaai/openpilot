#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// One workgroup gathers and combines the four expert rows for one token.
extern "C" __global__ __launch_bounds__(256) void gptoss_combine_forward(
    __hip_bfloat16 *__restrict__ out, const __hip_bfloat16 *__restrict__ z,
    const int *__restrict__ dest_row, const float *__restrict__ weights) {
  const int token = blockIdx.x, group = token / TOKENS, tid = threadIdx.x;
  __shared__ int rows[4];
  __shared__ float ws[4];
  if (tid < 4) {
    rows[tid] = group * ROWS + dest_row[token * 4 + tid];
    ws[tid] = weights[token * 4 + tid];
  }
  __syncthreads();

  for (int d = tid; d < 2880; d += 256) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      acc = fmaf(ws[j], (float)z[(long long)rows[j] * 2880 + d], acc);
    }
    out[(long long)token * 2880 + d] = (__hip_bfloat16)acc;
  }
}
