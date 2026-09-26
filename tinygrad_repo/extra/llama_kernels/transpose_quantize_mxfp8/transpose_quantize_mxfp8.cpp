#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>

#ifndef M_DIM
#define M_DIM 8192
#endif
#ifndef N_DIM
#define N_DIM 14336
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif

constexpr int BLK = 32;
constexpr int TILE_M = BLK;                 // one mxfp8 block along M per tile
constexpr int TILE_N = THREADS_PER_WG;      // 256, one output column per thread
constexpr int LDS_STRIDE = TILE_N + 1;      // +1 pad: stride 257 ≡ 1 (mod 32) -> conflict-free column reads
constexpr int N_TILES_N = N_DIM / TILE_N;
constexpr float FP8_MAX = 448.0f;

static_assert(M_DIM % TILE_M == 0, "M_DIM must be a multiple of 32");
static_assert(N_DIM % TILE_N == 0, "N_DIM must be a multiple of 256");

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
transpose_quantize_mxfp8(__hip_fp8_storage_t* __restrict__ q,        // (N_DIM, M_DIM)
                         uint8_t* __restrict__ e8_out,               // (N_DIM, M_DIM/32)
                         const __hip_bfloat16* __restrict__ g)       // (M_DIM, N_DIM)
{
  __shared__ __hip_bfloat16 lds[TILE_M * LDS_STRIDE];
  const int tid    = threadIdx.x;
  const int tile_m = blockIdx.x / N_TILES_N;     // which 32-block along M
  const int tile_n = blockIdx.x % N_TILES_N;

  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm++)
    lds[mm * LDS_STRIDE + tid] = g[(long long)(tile_m * TILE_M + mm) * N_DIM + (tile_n * TILE_N + tid)];
  __syncthreads();

  float vals[TILE_M];
  float amax = 0.0f;
  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm++) {
    float v = (float)lds[mm * LDS_STRIDE + tid];
    vals[mm] = v;
    amax = fmaxf(amax, fabsf(v));
  }
  int e8 = (int)floorf(log2f(fmaxf(amax, 1e-38f))) + 127;
  e8 = max(0, min(254, e8));
  float qscale = exp2f((float)(127 - e8));

  const long long n = tile_n * TILE_N + tid;
  __hip_fp8_storage_t out[TILE_M];
  #pragma unroll
  for (int mm = 0; mm < TILE_M; mm++)
    out[mm] = __hip_cvt_float_to_fp8(fmaxf(-FP8_MAX, fminf(FP8_MAX, vals[mm] * qscale)), __HIP_SATFINITE, __HIP_E4M3);
  // 32 contiguous fp8 along M -> two 16-byte vector stores
  long long obase = n * M_DIM + (long long)(tile_m * TILE_M);
  *reinterpret_cast<uint4*>(&q[obase])      = *reinterpret_cast<uint4*>(&out[0]);
  *reinterpret_cast<uint4*>(&q[obase + 16]) = *reinterpret_cast<uint4*>(&out[16]);
  e8_out[n * (M_DIM / BLK) + tile_m] = (uint8_t)e8;
}
