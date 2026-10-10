typedef long unsigned int size_t;
typedef __bf16 bf16;
typedef float float4_t __attribute__((ext_vector_type(4)));

extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);

// GPT-OSS router dgrad join, specialized to the production local shard:
//   bf16(bf16(sum_e bf16(weight[e,d]) * fp32(grad_logits[t,e]))
//        + bf16(bf16(residual[t,d]) * 2**(127-e8[t,d/32])))
extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(64, 64)))
moe_router_dgrad_fused(bf16 *__restrict__ out, const bf16 *__restrict__ weight,
                       const float *__restrict__ grad_logits, const bf16 *__restrict__ residual,
                       const unsigned char *__restrict__ e8) {
  constexpr int K = 2880, KP = 3072, E = 32;
  constexpr int DIM_TILE = 320, DIMS_PER_THREAD = 5;
  const int lane_x = __ockl_get_local_id(0);
  const int lane_y = __ockl_get_local_id(1);
  const int dim_tile = __ockl_get_group_id(0);
  const int token8 = __ockl_get_group_id(1);
  const int dim0 = dim_tile * DIM_TILE + lane_y * 16 + lane_x;
  const int token0 = token8 * 8;

  float acc[8][DIMS_PER_THREAD] = {};
  #pragma unroll 1
  for (int ec = 0; ec < E / 4; ec++) {
    const int expert = ec * 4;
    const float4_t g0 = *reinterpret_cast<const float4_t *>(grad_logits + (long long)(token0 + 0) * E + expert);
    const float4_t g1 = *reinterpret_cast<const float4_t *>(grad_logits + (long long)(token0 + 1) * E + expert);
    const float4_t g2 = *reinterpret_cast<const float4_t *>(grad_logits + (long long)(token0 + 2) * E + expert);
    const float4_t g3 = *reinterpret_cast<const float4_t *>(grad_logits + (long long)(token0 + 3) * E + expert);
    const float4_t g4 = *reinterpret_cast<const float4_t *>(grad_logits + (long long)(token0 + 4) * E + expert);
    const float4_t g5 = *reinterpret_cast<const float4_t *>(grad_logits + (long long)(token0 + 5) * E + expert);
    const float4_t g6 = *reinterpret_cast<const float4_t *>(grad_logits + (long long)(token0 + 6) * E + expert);
    const float4_t g7 = *reinterpret_cast<const float4_t *>(grad_logits + (long long)(token0 + 7) * E + expert);
    #pragma unroll
    for (int j = 0; j < DIMS_PER_THREAD; j++) {
      const int d = dim0 + j * 64;
      const float w0 = (float)weight[(long long)(expert + 0) * K + d];
      const float w1 = (float)weight[(long long)(expert + 1) * K + d];
      const float w2 = (float)weight[(long long)(expert + 2) * K + d];
      const float w3 = (float)weight[(long long)(expert + 3) * K + d];
      acc[0][j] = acc[0][j] + w0*g0.x + w1*g0.y + w2*g0.z + w3*g0.w;
      acc[1][j] = acc[1][j] + w0*g1.x + w1*g1.y + w2*g1.z + w3*g1.w;
      acc[2][j] = acc[2][j] + w0*g2.x + w1*g2.y + w2*g2.z + w3*g2.w;
      acc[3][j] = acc[3][j] + w0*g3.x + w1*g3.y + w2*g3.z + w3*g3.w;
      acc[4][j] = acc[4][j] + w0*g4.x + w1*g4.y + w2*g4.z + w3*g4.w;
      acc[5][j] = acc[5][j] + w0*g5.x + w1*g5.y + w2*g5.z + w3*g5.w;
      acc[6][j] = acc[6][j] + w0*g6.x + w1*g6.y + w2*g6.z + w3*g6.w;
      acc[7][j] = acc[7][j] + w0*g7.x + w1*g7.y + w2*g7.z + w3*g7.w;
    }
  }

  #pragma unroll
  for (int t = 0; t < 8; t++) {
    #pragma unroll
    for (int j = 0; j < DIMS_PER_THREAD; j++) {
      const int token = token0 + t, d = dim0 + j * 64;
      const unsigned char scale_e = e8[(long long)token * (KP / 32) + d / 32];
      // Integer E8M0 exponents produce an exact power of two. e8=254 is 2^-127, a float subnormal.
      const unsigned scale_bits = scale_e >= 254 ? (0x00400000u >> ((unsigned)scale_e - 254u)) : (254u - (unsigned)scale_e) << 23;
      const float qscale = __builtin_bit_cast(float, scale_bits);
      const bf16 router = (bf16)acc[t][j];
      const bf16 dequant = (bf16)((float)residual[(long long)token * KP + d] * qscale);
      out[(long long)token * K + d] = (bf16)((float)router + (float)dequant);
    }
  }
}
