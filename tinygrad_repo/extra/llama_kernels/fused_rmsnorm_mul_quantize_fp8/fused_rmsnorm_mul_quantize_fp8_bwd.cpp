#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Full backward for fused_rmsnorm_mul_quantize_fp8.cpp. One HBM pass per row produces:
//   grad_x (bf16)              — gradient w.r.t. pre-rmsnorm x
//   grad_weight_partial (fp32) — per-WG partial of the weight gradient, reduced later
//
// Input (all read):
//   grad_fp8 (bf16)     — upstream grad w.r.t. fp8_out (bf16-typed gradient value)
//   x_normed (bf16)     — saved from the fwd kernel, shape (ROWS, HIDDEN)
//   rrms (fp32)         — saved rrms per row
//   weight (bf16)       — per-HIDDEN rmsnorm weight
//   amax_state (bf16)   — delayed amax used to compute the fp8 scale in fwd
//
// Chain: y = x_normed * weight; fp8 = sat(y * scale). Through STE: grad_y = grad_fp8 * scale.
//        grad_x_normed = grad_y * weight.
//        grad_weight   = sum_rows(grad_y * x_normed).
//        grad_x        = rrms * (grad_x_normed - x_normed * mean(grad_x_normed * x_normed, last_dim)).

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

constexpr int VEC = 8;
constexpr float FP8_MAX = 448.0f;

static_assert(N_ELEMS % HIDDEN == 0, "N_ELEMS must be a multiple of HIDDEN");
static_assert(HIDDEN % (THREADS_PER_WG * VEC) == 0, "HIDDEN must be divisible by THREADS_PER_WG*VEC");

constexpr int ROWS = N_ELEMS / HIDDEN;
constexpr int ELEMS_PER_THREAD = HIDDEN / THREADS_PER_WG;
constexpr int VECS_PER_THREAD = ELEMS_PER_THREAD / VEC;

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_rmsnorm_mul_quantize_fp8_bwd(
    __hip_bfloat16*       __restrict__ grad_x,              // out: bf16, ROWS*HIDDEN
    float*                __restrict__ grad_weight_partial, // out: fp32, NUM_WG*HIDDEN
    const __hip_bfloat16* __restrict__ grad_fp8,            // in:  bf16, ROWS*HIDDEN (grad of fp8_out)
    const __hip_bfloat16* __restrict__ x_normed,            // in:  bf16, ROWS*HIDDEN
    const float*          __restrict__ rrms,                // in:  fp32, ROWS
    const __hip_bfloat16* __restrict__ weight,              // in:  bf16, HIDDEN
    const float*          __restrict__ amax_state)          // in:  fp32 scalar
{
  __shared__ float sdata[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;

  const float scale = FP8_MAX / (static_cast<float>(*amax_state) + 1e-8f);
  const float inv_hidden = 1.0f / static_cast<float>(HIDDEN);

  // Per-thread accumulator for grad_weight (across all rows this WG touches).
  float gw_accum[ELEMS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++) gw_accum[i] = 0.0f;

  // Preload weight into registers (same across rows). Use ELEMS_PER_THREAD entries.
  float w_regs[ELEMS_PER_THREAD];
  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    const int h_base = tid * VEC + v * THREADS_PER_WG * VEC;
    float4 w_raw = *reinterpret_cast<const float4*>(&weight[h_base]);
    const __hip_bfloat16 *wi = reinterpret_cast<const __hip_bfloat16*>(&w_raw);
    #pragma unroll
    for (int i = 0; i < VEC; i++) w_regs[v * VEC + i] = static_cast<float>(wi[i]);
  }

  for (int row = wg; row < ROWS; row += NUM_WG) {
    const int row_off = row * HIDDEN;
    const float rrms_v = rrms[row];

    // Load grad_fp8 and x_normed rows into registers, compute grad_y and grad_x_normed.
    float g_y_regs[ELEMS_PER_THREAD];
    float xn_regs[ELEMS_PER_THREAD];
    float g_xn_regs[ELEMS_PER_THREAD];  // grad_x_normed
    float local_dot = 0.0f;  // sum(grad_x_normed * x_normed) for mean

    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h_base = tid * VEC + v * THREADS_PER_WG * VEC;
      float4 g_raw  = *reinterpret_cast<const float4*>(&grad_fp8[row_off + h_base]);
      float4 xn_raw = *reinterpret_cast<const float4*>(&x_normed[row_off + h_base]);
      const __hip_bfloat16 *gi  = reinterpret_cast<const __hip_bfloat16*>(&g_raw);
      const __hip_bfloat16 *xni = reinterpret_cast<const __hip_bfloat16*>(&xn_raw);

      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        const int idx = v * VEC + i;
        const float g_y = static_cast<float>(gi[i]) * scale;
        const float xn  = static_cast<float>(xni[i]);
        g_y_regs[idx] = g_y;
        xn_regs[idx]  = xn;
        g_xn_regs[idx] = g_y * w_regs[idx];           // grad_x_normed = grad_y * weight
        gw_accum[idx] += g_y * xn;                    // grad_weight contrib
        local_dot += g_xn_regs[idx] * xn;             // for mean
      }
    }

    // LDS reduce local_dot to sdata[0].
    sdata[tid] = local_dot;
    __syncthreads();
    for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
      if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
      __syncthreads();
    }
    const float mean_term = sdata[0] * inv_hidden;

    // Compute grad_x = rrms * (grad_x_normed - x_normed * mean_term) and write.
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const int h_base = tid * VEC + v * THREADS_PER_WG * VEC;
      __hip_bfloat16 out[VEC];
      #pragma unroll
      for (int i = 0; i < VEC; i++) {
        const int idx = v * VEC + i;
        const float dx = rrms_v * (g_xn_regs[idx] - xn_regs[idx] * mean_term);
        out[i] = static_cast<__hip_bfloat16>(dx);
      }
      *reinterpret_cast<float4*>(&grad_x[row_off + h_base]) = *reinterpret_cast<float4*>(out);
    }
    __syncthreads();
  }

  // Write this WG's grad_weight partial to HBM (fp32, NUM_WG x HIDDEN layout).
  const int gw_row_off = wg * HIDDEN;
  #pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    const int h_base = tid * VEC + v * THREADS_PER_WG * VEC;
    // Write 8 fp32 values with two float4 stores.
    float4 out_lo, out_hi;
    out_lo.x = gw_accum[v * VEC + 0]; out_lo.y = gw_accum[v * VEC + 1];
    out_lo.z = gw_accum[v * VEC + 2]; out_lo.w = gw_accum[v * VEC + 3];
    out_hi.x = gw_accum[v * VEC + 4]; out_hi.y = gw_accum[v * VEC + 5];
    out_hi.z = gw_accum[v * VEC + 6]; out_hi.w = gw_accum[v * VEC + 7];
    *reinterpret_cast<float4*>(&grad_weight_partial[gw_row_off + h_base + 0]) = out_lo;
    *reinterpret_cast<float4*>(&grad_weight_partial[gw_row_off + h_base + 4]) = out_hi;
  }
}
