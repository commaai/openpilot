#include <hip/hip_runtime.h>
#include <stdint.h>

using bf16 = __bf16;
using uint4v = uint32_t __attribute__((ext_vector_type(4)));

#if !defined(ROWS) || !defined(REAL_D) || !defined(PAD_D)
#error "ROWS, REAL_D and PAD_D are required"
#endif

// Exact GPT-OSS final residual: bf16(bf16(x + bf16(proj + bias)) + moe).
// proj has the physical GEMM pitch PAD_D while every other matrix is packed REAL_D.
extern "C" __global__ __launch_bounds__(256) void gptoss_residual_join_vec(
    bf16 *__restrict__ out, const bf16 *__restrict__ x, const bf16 *__restrict__ proj,
    const bf16 *__restrict__ bias, const bf16 *__restrict__ moe) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  #pragma unroll
  for (int pass = 0; pass < 2; pass++) {
    const int col = (tid + pass * 256) * 8;
    if (col >= REAL_D) continue;
    const long long live_off = (long long)row * REAL_D + col;
    const long long pad_off = (long long)row * PAD_D + col;
    const uint4v xv = *reinterpret_cast<const uint4v *>(x + live_off);
    const uint4v pv = *reinterpret_cast<const uint4v *>(proj + pad_off);
    const uint4v bv = *reinterpret_cast<const uint4v *>(bias + col);
    const uint4v mv = *reinterpret_cast<const uint4v *>(moe + live_off);
    uint4v ov;
    const uint16_t *xh = reinterpret_cast<const uint16_t *>(&xv);
    const uint16_t *ph = reinterpret_cast<const uint16_t *>(&pv);
    const uint16_t *bh = reinterpret_cast<const uint16_t *>(&bv);
    const uint16_t *mh = reinterpret_cast<const uint16_t *>(&mv);
    uint16_t *oh = reinterpret_cast<uint16_t *>(&ov);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const bf16 xb = __builtin_bit_cast(bf16, xh[i]);
      const bf16 pb = __builtin_bit_cast(bf16, ph[i]);
      const bf16 bb = __builtin_bit_cast(bf16, bh[i]);
      const bf16 mb = __builtin_bit_cast(bf16, mh[i]);
      const bf16 attn = (bf16)((float)pb + (float)bb);
      const bf16 h = (bf16)((float)xb + (float)attn);
      oh[i] = __builtin_bit_cast(uint16_t, (bf16)((float)h + (float)mb));
    }
    *reinterpret_cast<uint4v *>(out + live_off) = ov;
  }
}
