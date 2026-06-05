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
static_assert(HIDDEN % VEC == 0, "HIDDEN must be divisible by VEC");

// fused silu*mul backward, three outputs in a single HBM pass:
//   1) bf16 grad_xw13      — consumed by downstream bf16 autograd chain
//   2) fp8  grad_xw13_fp8  — delayed-scale quantize using grad_amax_state (mailbox to matmul bwd)
//   3) fp32 grad_amax_buf  — per-WG partial |grad_xw13|, reduced into next step's grad_amax_state
// grad_amax_state is read for the fp8 scale. The store of new_grad_amax into grad_amax_state's
// buffer is built in Python as a separate effect and threaded into grad_a via .after(store).
extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fused_silu_mul_bwd_w13(
    __hip_bfloat16*       __restrict__ grad_xw13_out,        // bf16, 2*N_ELEMS
    __hip_fp8_storage_t*  __restrict__ grad_xw13_fp8_out,    // fp8,  2*N_ELEMS
    float*                __restrict__ grad_amax_buf,        // fp32, NUM_WG per-WG partials
    const __hip_bfloat16* __restrict__ xw13,                 // bf16, 2*N_ELEMS
    const __hip_bfloat16* __restrict__ grad_x2,              // bf16, N_ELEMS
    const float*          __restrict__ amax_state,           // fp32 scalar (fwd x2 amax)
    const float*          __restrict__ grad_amax_state)      // fp32 scalar (delayed grad amax)
{
  __shared__ float sdata[THREADS_PER_WG];

  const int tid = threadIdx.x;
  const int wg  = blockIdx.x;
  const int gid = wg * THREADS_PER_WG + tid;
  const int stride_elems = NUM_WG * THREADS_PER_WG * VEC;

  const float scale = FP8_MAX / (static_cast<float>(*amax_state) + 1e-8f);
  const float g_scale = FP8_MAX / (static_cast<float>(*grad_amax_state) + 1e-8f);
  float local_max = 0.0f;

  for (int base = gid * VEC; base < N_ELEMS; base += stride_elems) {
    const int outer = base / HIDDEN;
    const int inner = base % HIDDEN;
    const int xw1_off = outer * 2 * HIDDEN + inner;
    const int xw3_off = xw1_off + HIDDEN;

    float4 x1_raw = *reinterpret_cast<const float4*>(&xw13[xw1_off]);
    float4 x3_raw = *reinterpret_cast<const float4*>(&xw13[xw3_off]);
    float4 g_raw  = *reinterpret_cast<const float4*>(&grad_x2[base]);

    const __hip_bfloat16 *x1 = reinterpret_cast<const __hip_bfloat16*>(&x1_raw);
    const __hip_bfloat16 *x3 = reinterpret_cast<const __hip_bfloat16*>(&x3_raw);
    const __hip_bfloat16 *gv = reinterpret_cast<const __hip_bfloat16*>(&g_raw);

    __hip_bfloat16 out1[VEC], out3[VEC];
    __hip_fp8_storage_t fp8_1[VEC], fp8_3[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; i++) {
      const float f1 = static_cast<float>(x1[i]);
      const float f3 = static_cast<float>(x3[i]);
      const float fg = static_cast<float>(gv[i]);
      const float sig = 1.0f / (1.0f + __expf(-f1));
      const float silu = f1 * sig;
      const float silu_prime = sig + silu * (1.0f - sig);
      const float gs = fg * scale;
      const float g1 = gs * silu_prime * f3;
      const float g3 = gs * silu;
      out1[i] = static_cast<__hip_bfloat16>(g1);
      out3[i] = static_cast<__hip_bfloat16>(g3);
      local_max = fmaxf(local_max, fmaxf(fabsf(g1), fabsf(g3)));
      fp8_1[i] = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, g1 * g_scale)), __HIP_SATFINITE, __HIP_E4M3);
      fp8_3[i] = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, g3 * g_scale)), __HIP_SATFINITE, __HIP_E4M3);
    }

    *reinterpret_cast<float4*>(&grad_xw13_out[xw1_off]) = *reinterpret_cast<float4*>(out1);
    *reinterpret_cast<float4*>(&grad_xw13_out[xw3_off]) = *reinterpret_cast<float4*>(out3);
    *reinterpret_cast<uint64_t*>(&grad_xw13_fp8_out[xw1_off]) = *reinterpret_cast<uint64_t*>(fp8_1);
    *reinterpret_cast<uint64_t*>(&grad_xw13_fp8_out[xw3_off]) = *reinterpret_cast<uint64_t*>(fp8_3);
  }

  sdata[tid] = local_max;
  __syncthreads();
  for (int s = THREADS_PER_WG / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) grad_amax_buf[wg] = sdata[0];
}
