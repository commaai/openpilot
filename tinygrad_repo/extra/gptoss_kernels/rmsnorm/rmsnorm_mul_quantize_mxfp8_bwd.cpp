#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef N_ELEMS
#define N_ELEMS 47185920
#endif
#ifndef HIDDEN
#define HIDDEN 2880
#endif
#ifndef PADDED
#define PADDED 3072
#endif
#ifndef NUM_WG
#define NUM_WG 1024
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif

constexpr int ROWS = N_ELEMS / HIDDEN;
constexpr int BLOCK = 32;
constexpr int SCALE_BLOCKS = PADDED / BLOCK;
constexpr int ELEMS_PER_THREAD = (HIDDEN + THREADS_PER_WG - 1) / THREADS_PER_WG;

static_assert(N_ELEMS % HIDDEN == 0, "N_ELEMS must be divisible by HIDDEN");
static_assert(HIDDEN % BLOCK == 0 && PADDED % BLOCK == 0 && PADDED >= HIDDEN,
              "HIDDEN and PADDED must be block aligned");

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void rmsnorm_mul_quantize_mxfp8_bwd(
    __hip_bfloat16 *__restrict__ grad_x,
    float *__restrict__ grad_weight_partial,
    const __hip_bfloat16 *__restrict__ grad_q,
    const __hip_bfloat16 *__restrict__ x,
    const __hip_bfloat16 *__restrict__ weight,
    const uint8_t *__restrict__ e8,
    const float *__restrict__ rrms) {
  __shared__ float reduce[THREADS_PER_WG];
  const int tid = threadIdx.x;
  const int wg = blockIdx.x;

  float w[ELEMS_PER_THREAD];
  float gw[ELEMS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    int col = tid + i * THREADS_PER_WG;
    w[i] = col < HIDDEN ? (float)weight[col] : 0.0f;
    gw[i] = 0.0f;
  }

  for (int row = wg; row < ROWS; row += NUM_WG) {
    const long long xbase = (long long)row * HIDDEN;
    const long long qbase = (long long)row * PADDED;
    const long long ebase = (long long)row * SCALE_BLOCKS;
    const float r = rrms[row];
    float xn[ELEMS_PER_THREAD];
    float gxn[ELEMS_PER_THREAD];
    float local_dot = 0.0f;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      const int col = tid + i * THREADS_PER_WG;
      if (col < HIDDEN) {
        const float xnf = (float)x[xbase + col] * r;
        const unsigned se = (unsigned)(254 - (int)e8[ebase + col / BLOCK]) << 23;
        const float qscale = __builtin_bit_cast(float, se);
        const float gy = (float)grad_q[qbase + col] * qscale;
        const float gxnf = gy * w[i];
        xn[i] = xnf;
        gxn[i] = gxnf;
        gw[i] += gy * xnf;
        local_dot = fmaf(gxnf, xnf, local_dot);
      } else {
        xn[i] = gxn[i] = 0.0f;
      }
    }

    reduce[tid] = local_dot;
    __syncthreads();
    for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
      if (tid < s) reduce[tid] += reduce[tid + s];
      __syncthreads();
    }
    const float mean_term = reduce[0] * (1.0f / (float)HIDDEN);

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
      const int col = tid + i * THREADS_PER_WG;
      if (col < HIDDEN) grad_x[xbase + col] = (__hip_bfloat16)(r * (gxn[i] - xn[i] * mean_term));
    }
    __syncthreads();
  }

  const long long gwbase = (long long)wg * HIDDEN;
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    const int col = tid + i * THREADS_PER_WG;
    if (col < HIDDEN) grad_weight_partial[gwbase + col] = gw[i];
  }
}
