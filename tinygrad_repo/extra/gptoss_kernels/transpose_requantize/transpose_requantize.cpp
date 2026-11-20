#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>

constexpr int BLK = 32;
constexpr int TILE_M = BLK;
constexpr int TILE_N = 128;
constexpr int N_TILES_N = N_DIM / TILE_N;
constexpr int REQUANT_M_GROUPS = M_DIM / 128;
static_assert(M_DIM % 128 == 0 && N_DIM % TILE_N == 0);
typedef short short2v __attribute__((ext_vector_type(2)));

// Source15's M32 schedule: dequantize with its BF16 rounding boundary, then transpose and requantize.
extern "C" __global__ __launch_bounds__(128) void
transpose_requantize_mxfp8(__hip_fp8_storage_t* __restrict__ q_out,
                          uint8_t* __restrict__ e8_out,
                          uint32_t* __restrict__ si_out,
                          const __hip_fp8_storage_t* __restrict__ q_in,
                          const uint8_t* __restrict__ e8_in) {
  const int tid = threadIdx.x;
  int tile = blockIdx.x;
  const int tile_n = tile % N_TILES_N;
  tile /= N_TILES_N;
  const int tile_m = tile % (M_DIM / BLK);
  const int expert = tile / (M_DIM / BLK);
  const int m_group = tile_m / 4, sub = tile_m % 4;
  const long long n = tile_n * TILE_N + tid;
  float vals[TILE_M];
  float amax = 0.0f;
  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm++) {
    const long long m = tile_m * TILE_M + mm;
    const long long src = ((long long)expert * M_DIM + m) * N_DIM + n;
    const __hip_fp8_e4m3 fq = __builtin_bit_cast(__hip_fp8_e4m3, q_in[src]);
    const long long src_e8 = ((long long)expert * M_DIM + m) * (N_DIM / BLK) + n / BLK;
    const unsigned src_scale = e8_in[src_e8];
    const float dqscale = __builtin_bit_cast(float, src_scale ? src_scale << 23 : 0x00400000u);
    const float v = (float)(__hip_bfloat16)((float)fq * dqscale);
    vals[mm] = v;
    amax = fmaxf(amax, fabsf(v));
  }
  int e8 = (int)((__builtin_bit_cast(unsigned, amax) >> 23) & 0xFFu);
  e8 = max(0, min(254, e8));
  const float scale_f = __builtin_bit_cast(float, (unsigned)e8 << 23);
  __hip_fp8_storage_t out[TILE_M];
  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm += 2) {
    short2v acc = {0, 0};
    acc = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(acc, vals[mm], vals[mm + 1], scale_f, false);
    const unsigned char *rb = reinterpret_cast<const unsigned char *>(&acc);
    out[mm] = rb[0]; out[mm + 1] = rb[1];
  }
  const long long obase = ((long long)expert * N_DIM + n) * M_DIM + (long long)tile_m * TILE_M;
  *reinterpret_cast<uint4*>(&q_out[obase]) = *reinterpret_cast<uint4*>(&out[0]);
  *reinterpret_cast<uint4*>(&q_out[obase + 16]) = *reinterpret_cast<uint4*>(&out[16]);
  e8_out[((long long)expert * N_DIM + n) * (M_DIM / BLK) + tile_m] = (uint8_t)e8;
  reinterpret_cast<uint8_t *>(si_out)[(((long long)expert * REQUANT_M_GROUPS + m_group) * N_DIM + n) * 4 + sub] = (uint8_t)e8;
}
