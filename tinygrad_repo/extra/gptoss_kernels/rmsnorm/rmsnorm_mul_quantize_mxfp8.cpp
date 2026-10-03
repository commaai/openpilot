#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

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
#ifndef EPS_LITERAL
#define EPS_LITERAL 1e-5f
#endif
#ifndef COOP_EPILOGUE
#define COOP_EPILOGUE 0
#endif

constexpr int ROWS = N_ELEMS / HIDDEN;
constexpr int BLOCK = 32;
constexpr int SCALE_BLOCKS = PADDED / BLOCK;
constexpr float FP8_MAX = 448.0f;

static_assert(N_ELEMS % HIDDEN == 0, "N_ELEMS must be divisible by HIDDEN");
static_assert(HIDDEN % BLOCK == 0 && PADDED % BLOCK == 0 && PADDED >= HIDDEN,
              "HIDDEN and PADDED must be block aligned");
static_assert(SCALE_BLOCKS <= THREADS_PER_WG, "one thread handles each MXFP8 block");

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void rmsnorm_mul_quantize_mxfp8(
    __hip_fp8_storage_t *__restrict__ q_out,
    uint8_t *__restrict__ e8_out,
    float *__restrict__ rrms_out,
    const __hip_bfloat16 *__restrict__ x,
    const __hip_bfloat16 *__restrict__ weight) {
  __shared__ float reduce[THREADS_PER_WG];
  __shared__ __hip_bfloat16 x_row[HIDDEN];

  const int tid = threadIdx.x;
  for (int row = blockIdx.x; row < ROWS; row += NUM_WG) {
    const long long xbase = (long long)row * HIDDEN;
    float sum_sq = 0.0f;
    for (int col = tid; col < HIDDEN; col += THREADS_PER_WG) {
      __hip_bfloat16 xb = x[xbase + col];
      x_row[col] = xb;
      float xf = (float)xb;
      sum_sq = fmaf(xf, xf, sum_sq);
    }

    reduce[tid] = sum_sq;
    __syncthreads();
    for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
      if (tid < s) reduce[tid] += reduce[tid + s];
      __syncthreads();
    }
    const float rrms = rsqrtf(reduce[0] * (1.0f / (float)HIDDEN) + EPS_LITERAL);
    if (tid == 0) rrms_out[row] = rrms;

#if COOP_EPILOGUE
    // Exact GPT-OSS shape: eight lanes share each 32-value block.  This keeps
    // all four waves active and emits aligned dwords instead of serializing
    // the full block in one lane (GPU1/7 ABBA64: 242.40 -> 138.89/137.99 us).
    constexpr int BLOCKS_PER_PASS = THREADS_PER_WG / 8;
    const int lane = tid & 7;
    const int group = tid >> 3;
    #pragma unroll
    for (int pass = 0; pass < (SCALE_BLOCKS + BLOCKS_PER_PASS - 1) / BLOCKS_PER_PASS; pass++) {
      const int block = pass * BLOCKS_PER_PASS + group;
      if (block < SCALE_BLOCKS) {
        const int col_base = block * BLOCK + lane * 4;
        float vals[4];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          const int col = col_base + i;
          float v = 0.0f;
          if (col < HIDDEN) {
            float xn = (float)x_row[col] * rrms;
            __hip_bfloat16 yb = (__hip_bfloat16)(xn * (float)weight[col]);
            v = (float)yb;
          }
          vals[i] = v;
          amax = fmaxf(amax, fabsf(v));
        }
        #pragma unroll
        for (int s = 4; s > 0; s >>= 1) amax = fmaxf(amax, __shfl_down(amax, s, 8));
        float qscale = 0.0f;
        if (lane == 0) {
          int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
          e8 = max(0, min(254, e8));
          qscale = exp2f((float)(127 - e8));
          e8_out[(long long)row * SCALE_BLOCKS + block] = (uint8_t)e8;
        }
        qscale = __shfl(qscale, 0, 8);
        __hip_fp8_storage_t packed[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          float v = fmaxf(-FP8_MAX, fminf(FP8_MAX, vals[i] * qscale));
          packed[i] = __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
        }
        const long long qbase = (long long)row * PADDED + block * BLOCK + lane * 4;
        *reinterpret_cast<uint32_t *>(&q_out[qbase]) = *reinterpret_cast<uint32_t *>(&packed[0]);
      }
    }
#else
    if (tid < SCALE_BLOCKS) {
      const int col_base = tid * BLOCK;
      float vals[BLOCK];
      float amax = 0.0f;
      #pragma unroll
      for (int i = 0; i < BLOCK; i++) {
        const int col = col_base + i;
        float v = 0.0f;
        if (col < HIDDEN) {
          float xn = (float)x_row[col] * rrms;
          __hip_bfloat16 yb = (__hip_bfloat16)(xn * (float)weight[col]);
          v = (float)yb;
        }
        vals[i] = v;
        amax = fmaxf(amax, fabsf(v));
      }
      int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
      e8 = max(0, min(254, e8));
      const float qscale = exp2f((float)(127 - e8));
      __hip_fp8_storage_t packed[BLOCK];
      #pragma unroll
      for (int i = 0; i < BLOCK; i++) {
        float v = fmaxf(-FP8_MAX, fminf(FP8_MAX, vals[i] * qscale));
        packed[i] = __hip_cvt_float_to_fp8(v, __HIP_SATFINITE, __HIP_E4M3);
      }
      const long long qbase = (long long)row * PADDED + col_base;
      *reinterpret_cast<uint4 *>(&q_out[qbase]) = *reinterpret_cast<uint4 *>(&packed[0]);
      *reinterpret_cast<uint4 *>(&q_out[qbase + 16]) = *reinterpret_cast<uint4 *>(&packed[16]);
      e8_out[(long long)row * SCALE_BLOCKS + tid] = (uint8_t)e8;
    }
#endif
    __syncthreads();
  }
}
