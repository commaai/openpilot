#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// Fuses the full pre-matmul preparation for a layer into a single HBM pass:
//   y = rmsnorm(x) * weight            (reduce-mean-square + rsqrt + per-elem mul)
//   fp8 = fp8_sat(y * (FP8_MAX / amax_state))
// Also writes:
//   rrms[row]        — saved for the rmsnorm backward
//   amax_buf[wg]     — per-WG |y| partials, reduced later to update amax_state
//
// Layout: one WG per row, ROWS_PER_WG rows per WG via grid-stride (ROWS = N_ELEMS / HIDDEN).
// Each thread handles HIDDEN / THREADS_PER_WG elements per row.

#ifndef N_ELEMS
#define N_ELEMS 67108864
#endif
#ifndef HIDDEN
#define HIDDEN 4096
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
#ifndef HAS_RESIDUAL
#define HAS_RESIDUAL 0
#endif

constexpr int VEC = 8;
constexpr float FP8_MAX = 448.0f;

static_assert(N_ELEMS % HIDDEN == 0, "N_ELEMS must be a multiple of HIDDEN");
static_assert(HIDDEN % (THREADS_PER_WG * VEC) == 0, "HIDDEN must be divisible by THREADS_PER_WG*VEC");

constexpr int ROWS = N_ELEMS / HIDDEN;
constexpr int ELEMS_PER_THREAD = HIDDEN / THREADS_PER_WG;  // each thread sees this many elems per row
constexpr int VECS_PER_THREAD = ELEMS_PER_THREAD / VEC;    // number of 8-wide vec loads

#if HAS_RESIDUAL
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_add_rmsnorm_mul_quantize_fp8(
    __hip_fp8_storage_t*  __restrict__ fp8_out,         // fp8, ROWS*HIDDEN
    __hip_bfloat16*       __restrict__ h_out,           // bf16, ROWS*HIDDEN — x + residual (saved for downstream)
    __hip_bfloat16*       __restrict__ x_normed_out,    // bf16, ROWS*HIDDEN
    float*                __restrict__ rrms_out,        // fp32, ROWS
    float*                __restrict__ amax_buf,        // fp32, NUM_WG
    const __hip_bfloat16* __restrict__ x,               // bf16, ROWS*HIDDEN
    const __hip_bfloat16* __restrict__ residual,        // bf16, ROWS*HIDDEN — added into x before rmsnorm
    const __hip_bfloat16* __restrict__ weight,          // bf16, HIDDEN
    const float*          __restrict__ amax_state)      // fp32 scalar
{
#else
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_rmsnorm_mul_quantize_fp8(
    __hip_fp8_storage_t*  __restrict__ fp8_out,         // fp8, ROWS*HIDDEN
    __hip_bfloat16*       __restrict__ x_normed_out,    // bf16, ROWS*HIDDEN (saved for rmsnorm bwd)
    float*                __restrict__ rrms_out,        // fp32, ROWS (fp32 to match rmsnorm_bwd.cpp expectation)
    float*                __restrict__ amax_buf,        // fp32, NUM_WG per-WG partials
    const __hip_bfloat16* __restrict__ x,               // bf16, ROWS*HIDDEN
    const __hip_bfloat16* __restrict__ weight,          // bf16, HIDDEN (per-hidden scale)
    const float*          __restrict__ amax_state)      // fp32 scalar
{
#endif
  __shared__ float sdata[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;

  const float scale = FP8_MAX / (static_cast<float>(*amax_state) + 1e-8f);
  const float inv_hidden = 1.0f / static_cast<float>(HIDDEN);
  float local_max = 0.0f;

  // Grid-stride over rows. Each WG processes rows (wg, wg+NUM_WG, wg+2*NUM_WG, ...).
  for (int row = wg; row < ROWS; row += NUM_WG) {
    const int row_off = row * HIDDEN;

    // Load row (+ residual if present) into registers.
    float regs[ELEMS_PER_THREAD];
    float sum_sq = 0.0f;
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h_base = tid * VEC + v * THREADS_PER_WG * VEC;
      float4 raw = *reinterpret_cast<const float4*>(&x[row_off + h_base]);
      const __hip_bfloat16 *xi = reinterpret_cast<const __hip_bfloat16*>(&raw);
#if HAS_RESIDUAL
      float4 res_raw = *reinterpret_cast<const float4*>(&residual[row_off + h_base]);
      const __hip_bfloat16 *ri = reinterpret_cast<const __hip_bfloat16*>(&res_raw);
      __hip_bfloat16 h_buf[VEC];
#endif
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
#if HAS_RESIDUAL
        const float f = static_cast<float>(xi[i]) + static_cast<float>(ri[i]);
        h_buf[i] = static_cast<__hip_bfloat16>(f);
#else
        const float f = static_cast<float>(xi[i]);
#endif
        regs[v * VEC + i] = f;
        sum_sq += f * f;
      }
#if HAS_RESIDUAL
      *reinterpret_cast<float4*>(&h_out[row_off + h_base]) = *reinterpret_cast<float4*>(h_buf);
#endif
    }

    // LDS tree-reduce sum_sq across the WG.
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
      if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
      __syncthreads();
    }
    const float mean_sq = sdata[0] * inv_hidden;
    const float rrms = 1.0f / sqrtf(mean_sq + EPS_LITERAL);

    if (tid == 0) rrms_out[row] = rrms;

    // Normalize, multiply by weight, quantize. Also write x_normed (for rmsnorm bwd).
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h_base = tid * VEC + v * THREADS_PER_WG * VEC;
      float4 w_raw = *reinterpret_cast<const float4*>(&weight[h_base]);
      const __hip_bfloat16 *wi = reinterpret_cast<const __hip_bfloat16*>(&w_raw);

      __hip_fp8_storage_t out[VEC];
      __hip_bfloat16 xn[VEC];
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        const float x_normed = regs[v * VEC + i] * rrms;
        xn[i] = static_cast<__hip_bfloat16>(x_normed);
        const float y = x_normed * static_cast<float>(wi[i]);
        local_max = fmaxf(local_max, fabsf(y));
        const float scaled = fmaxf(-FP8_MAX, fminf(FP8_MAX, y * scale));
        out[i] = __hip_cvt_float_to_fp8(scaled, __HIP_SATFINITE, __HIP_E4M3);
      }
      *reinterpret_cast<uint64_t*>(&fp8_out[row_off + h_base]) = *reinterpret_cast<uint64_t*>(out);
      *reinterpret_cast<float4*>(&x_normed_out[row_off + h_base]) = *reinterpret_cast<float4*>(xn);
    }
    __syncthreads();  // before next row's sum_sq reduce reuses sdata
  }

  // Final per-WG amax reduce.
  sdata[tid] = local_max;
  __syncthreads();
  for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) amax_buf[wg] = sdata[0];
}
