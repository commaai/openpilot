#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#if !defined(ROWS) || !defined(HIDDEN) || !defined(THREADS) || !defined(EPS_LITERAL)
#error "ROWS, HIDDEN, THREADS and EPS_LITERAL are required"
#endif

static __device__ __forceinline__ float exact_saved_rrms(float x) {
  volatile float q=__builtin_amdgcn_rcpf(x);
  const unsigned int bits=__builtin_bit_cast(unsigned int,x);
  if ((bits&0x7f800000u)==0x7f800000u) return q;
  volatile float err=__builtin_fmaf(-x,q,1.0f);
  return __builtin_fmaf(err,q,q);
}

extern "C" __global__ __launch_bounds__(THREADS) void gptoss_final_rmsnorm_denom_rrms(
    float *__restrict__ denom, float *__restrict__ rrms, const __hip_bfloat16 *__restrict__ x) {
  const int row = blockIdx.x * THREADS + threadIdx.x;
  if (row >= ROWS) return;
  const __hip_bfloat16 *p = x + (long long)row * HIDDEN;
  float acc = 0.0f;
#pragma unroll 4
  for (int col = 0; col < HIDDEN; col += 4) {
    const float a = (float)p[col + 0];
    const float b = (float)p[col + 1];
    const float c = (float)p[col + 2];
    const float d = (float)p[col + 3];
    acc = acc + a*a + b*b + c*c + d*d;
  }
  const float value=__ocml_sqrt_f32(acc*(1.0f/(float)HIDDEN)+EPS_LITERAL);
  denom[row]=value;
  rrms[row]=exact_saved_rrms(value);
}
