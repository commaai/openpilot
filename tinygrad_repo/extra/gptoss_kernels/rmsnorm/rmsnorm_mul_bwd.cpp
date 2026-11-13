#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef ROWS
#define ROWS 16384
#endif
#ifndef HIDDEN
#define HIDDEN 2880
#endif
#ifndef NUM_WG
#define NUM_WG 1024
#endif
#ifndef THREADS
#define THREADS 256
#endif

static_assert(THREADS % 64 == 0, "THREADS must contain whole AMD waves");
static_assert(ROWS % (2 * NUM_WG) == 0, "ROWS must contain complete row pairs per partial");
constexpr int COLS_PER_THREAD = (HIDDEN + THREADS - 1) / THREADS;

extern "C" __global__ __launch_bounds__(THREADS) void rmsnorm_mul_bwd(
    __hip_bfloat16 *__restrict__ grad_x,
    float *__restrict__ grad_weight_partial,
    const __hip_bfloat16 *__restrict__ grad,
    const __hip_bfloat16 *__restrict__ x,
    const __hip_bfloat16 *__restrict__ weight,
    const float *__restrict__ rrms) {
  const int tid = threadIdx.x, part = blockIdx.x;
  float dw[COLS_PER_THREAD];
#pragma unroll
  for (int j = 0; j < COLS_PER_THREAD; j++) dw[j] = 0.0f;

  // Reduce two rows through one pair of workgroup barriers, retaining each row's reduction tree.
  __shared__ float pair_wave_sums[2][THREADS / 64];
  const int lane = tid & 63, wave = tid >> 6;
  for (int row = part; row < ROWS; row += 2 * NUM_WG) {
    const int row1 = row + NUM_WG;
    const float r0 = rrms[row], r1 = rrms[row1];
    float xn0[COLS_PER_THREAD], xn1[COLS_PER_THREAD];
    float dxn0[COLS_PER_THREAD], dxn1[COLS_PER_THREAD];
    float dot0 = 0.0f, dot1 = 0.0f;
#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++) {
      const int col = tid + j * THREADS;
      if (col < HIDDEN) {
        const long long idx0 = (long long)row * HIDDEN + col;
        const long long idx1 = (long long)row1 * HIDDEN + col;
        xn0[j] = (float)x[idx0] * r0;
        xn1[j] = (float)x[idx1] * r1;
        const float dy0 = (float)grad[idx0], dy1 = (float)grad[idx1];
        const float w = (float)weight[col];
        dxn0[j] = dy0 * w;
        dxn1[j] = dy1 * w;
        dw[j] = fmaf(dy0, xn0[j], dw[j]);
        dw[j] = fmaf(dy1, xn1[j], dw[j]);
        dot0 = fmaf(dxn0[j], xn0[j], dot0);
        dot1 = fmaf(dxn1[j], xn1[j], dot1);
      }
    }
#pragma unroll
    for (int off = 32; off; off >>= 1) {
      dot0 += __shfl_down(dot0, off, 64);
      dot1 += __shfl_down(dot1, off, 64);
    }
    if (lane == 0) {
      pair_wave_sums[0][wave] = dot0;
      pair_wave_sums[1][wave] = dot1;
    }
    __syncthreads();
    if (wave == 0) {
      dot0 = lane < THREADS / 64 ? pair_wave_sums[0][lane] : 0.0f;
      dot1 = lane < THREADS / 64 ? pair_wave_sums[1][lane] : 0.0f;
#pragma unroll
      for (int off = 32; off; off >>= 1) {
        dot0 += __shfl_down(dot0, off, 64);
        dot1 += __shfl_down(dot1, off, 64);
      }
      if (lane == 0) {
        pair_wave_sums[0][0] = dot0;
        pair_wave_sums[1][0] = dot1;
      }
    }
    __syncthreads();
    const float mean0 = pair_wave_sums[0][0] * (1.0f / (float)HIDDEN);
    const float mean1 = pair_wave_sums[1][0] * (1.0f / (float)HIDDEN);
#pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; j++) {
      const int col = tid + j * THREADS;
      if (col < HIDDEN) {
        const long long idx0 = (long long)row * HIDDEN + col;
        const long long idx1 = (long long)row1 * HIDDEN + col;
        grad_x[idx0] = (__hip_bfloat16)(r0 * (dxn0[j] - xn0[j] * mean0));
        grad_x[idx1] = (__hip_bfloat16)(r1 * (dxn1[j] - xn1[j] * mean1));
      }
    }
  }

#pragma unroll
  for (int j = 0; j < COLS_PER_THREAD; j++) {
    const int col = tid + j * THREADS;
    if (col < HIDDEN) grad_weight_partial[(long long)part * HIDDEN + col] = dw[j];
  }
}
