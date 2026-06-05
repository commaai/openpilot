#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#ifndef N_ELEMS
#define N_ELEMS 234881024
#endif
#ifndef HIDDEN
#define HIDDEN 14336
#endif
#ifndef NUM_WG
#define NUM_WG 1024
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif

constexpr int VEC = 8;
constexpr float FP8_MAX = 448.0f;

static_assert(N_ELEMS % VEC == 0, "N_ELEMS must be divisible by VEC");
static_assert(HIDDEN % VEC == 0, "HIDDEN must be divisible by VEC (so VEC loads don't straddle block boundary)");

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_silu_mul_cast_amax_w13(
    __hip_fp8_storage_t*  __restrict__ fp8_out,         // fp8, N_ELEMS
    float*                __restrict__ amax_buf,        // fp32, NUM_WG (per-WG amaxes)
    const __hip_bfloat16* __restrict__ xw13,            // bf16, 2*N_ELEMS
    const float*          __restrict__ amax_state)      // fp32 scalar
{
  __shared__ float sdata[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;
  const int gid = wg * THREADS_PER_WG + tid;
  const int stride_elems = NUM_WG * THREADS_PER_WG * VEC;

  const float scale = FP8_MAX / (static_cast<float>(*amax_state) + 1e-8f);
  float local_max = 0.0f;

  // grid-stride over 8-element groups
  for (int base = gid * VEC; base < N_ELEMS; base += stride_elems) {
    // interleaved xw13 layout: xw1 and xw3 are not contiguous halves
    const int outer = base / HIDDEN;
    const int inner = base % HIDDEN;
    const int xw1_off = outer * 2 * HIDDEN + inner;
    const int xw3_off = xw1_off + HIDDEN;

    float4 x1_raw = *reinterpret_cast<const float4*>(&xw13[xw1_off]);
    float4 x3_raw = *reinterpret_cast<const float4*>(&xw13[xw3_off]);

    const __hip_bfloat16 *x1 = reinterpret_cast<const __hip_bfloat16*>(&x1_raw);
    const __hip_bfloat16 *x3 = reinterpret_cast<const __hip_bfloat16*>(&x3_raw);

    __hip_fp8_storage_t out[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      const float f1 = static_cast<float>(x1[i]);
      const float f3 = static_cast<float>(x3[i]);
      const float silu = f1 / (1.0f + __expf(-f1));
      const float x2 = silu * f3;
      local_max = fmaxf(local_max, fabsf(x2));
      const float x_scaled = fmaxf(-FP8_MAX, fminf(FP8_MAX, x2 * scale));
      out[i] = __hip_cvt_float_to_fp8(x_scaled, __HIP_SATFINITE, __HIP_E4M3);
    }

    *reinterpret_cast<uint64_t*>(&fp8_out[base]) = *reinterpret_cast<uint64_t*>(out);
  }

  // LDS tree reduction: per-workgroup amax
  sdata[tid] = local_max;
  __syncthreads();
  for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }

  if (tid == 0) amax_buf[wg] = sdata[0];
}
