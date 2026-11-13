#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#ifndef N_ELEMS
#define N_ELEMS 1
#endif
#ifndef N_PARTIALS
#define N_PARTIALS 1
#endif
#ifndef THREADS
#define THREADS 256
#endif

static_assert(THREADS % 64 == 0, "THREADS must contain whole AMD waves");

extern "C" __global__ __launch_bounds__(THREADS) void grad_norm_bf16_partials(
    float *__restrict__ partial, const __hip_bfloat16 *__restrict__ x) {
  // Eight adjacent BF16 values per lane turn each wave access into contiguous 1 KiB vector traffic. Four
  // independent accumulators hide the conversion/FMA dependency chain that throttles a scalar grid-stride loop.
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  constexpr long long N_VEC = N_ELEMS / 8;
  for (long long i = (long long)blockIdx.x * THREADS + threadIdx.x; i < N_VEC;
       i += (long long)N_PARTIALS * THREADS) {
    const uint4 packed = reinterpret_cast<const uint4 *>(x)[i];
    const __hip_bfloat162 *p = reinterpret_cast<const __hip_bfloat162 *>(&packed);
    const float2 a = __bfloat1622float2(p[0]), b = __bfloat1622float2(p[1]);
    const float2 c = __bfloat1622float2(p[2]), d = __bfloat1622float2(p[3]);
    sum0 = fmaf(a.x, a.x, sum0); sum0 = fmaf(a.y, a.y, sum0);
    sum1 = fmaf(b.x, b.x, sum1); sum1 = fmaf(b.y, b.y, sum1);
    sum2 = fmaf(c.x, c.x, sum2); sum2 = fmaf(c.y, c.y, sum2);
    sum3 = fmaf(d.x, d.x, sum3); sum3 = fmaf(d.y, d.y, sum3);
  }
  for (long long i = N_VEC * 8 + (long long)blockIdx.x * THREADS + threadIdx.x; i < N_ELEMS;
       i += (long long)N_PARTIALS * THREADS) { const float v = (float)x[i]; sum0 = fmaf(v, v, sum0); }
  float sum = (sum0 + sum1) + (sum2 + sum3);

  #pragma unroll
  for (int offset = 32; offset; offset >>= 1) sum += __shfl_down(sum, offset, 64);
  __shared__ float wave_sums[THREADS / 64];
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  if (lane == 0) wave_sums[wave] = sum;
  __syncthreads();

  if (wave == 0) {
    sum = lane < THREADS / 64 ? wave_sums[lane] : 0.0f;
    #pragma unroll
    for (int offset = 32; offset; offset >>= 1) sum += __shfl_down(sum, offset, 64);
    if (lane == 0) partial[blockIdx.x] = sum;
  }
}
