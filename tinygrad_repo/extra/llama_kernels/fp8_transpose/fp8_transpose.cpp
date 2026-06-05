#include <hip/hip_runtime.h>

// LDS-staged 64x64 fp8 transpose.
//   in : (M_DIM, N_DIM) fp8 contiguous
//   out: (N_DIM, M_DIM) fp8 contiguous, out[c][r] = in[r][c]
//
// One WG processes one 64x64 output tile.  Each thread reads one uint4 (16 fp8) coalesced
// from input rows, stages into LDS, then writes one uint4 coalesced to the output (whose
// 16 fp8 come from 16 different input rows via in-LDS gather).
//
// LDS layout: lds[64][LDS_STRIDE] with LDS_STRIDE=65 (1 byte pad) to mitigate bank conflicts
// during the column-direction read of the write phase.

#ifndef M_DIM
#define M_DIM 16384
#endif
#ifndef N_DIM
#define N_DIM 28672
#endif
#ifndef THREADS_PER_WG
#define THREADS_PER_WG 256
#endif

constexpr int TILE = 64;
constexpr int VEC = 16;                    // fp8 per uint4 (128-bit) load/store
constexpr int LDS_PAD = 1;
constexpr int LDS_STRIDE = TILE + LDS_PAD; // 65 fp8 per row

static_assert(THREADS_PER_WG * VEC == TILE * TILE, "256 threads * 16 fp8 = 64*64");
static_assert(M_DIM % TILE == 0, "M_DIM must be a multiple of 64");
static_assert(N_DIM % TILE == 0, "N_DIM must be a multiple of 64");

constexpr int N_TILES_N = N_DIM / TILE;

struct alignas(16) fp8x16 { uint8_t v[16]; };

extern "C" __global__ __launch_bounds__(THREADS_PER_WG) void
fp8_transpose(uint8_t* __restrict__ out,         // (N_DIM, M_DIM)
              const uint8_t* __restrict__ in)    // (M_DIM, N_DIM)
{
  __shared__ uint8_t lds[TILE * LDS_STRIDE];

  const int tid     = threadIdx.x;
  const int wg_id   = blockIdx.x;
  const int tile_r  = wg_id / N_TILES_N;       // tile index along M dim of input
  const int tile_c  = wg_id % N_TILES_N;       // tile index along N dim of input

  const int a = tid / (TILE / VEC);            // 0..63 (row within tile during read; col within tile during write)
  const int b = tid % (TILE / VEC);            // 0..3
  const int b16 = b * VEC;                     // 0,16,32,48

  // ---- Read phase: input rows -> LDS rows
  {
    const long long src = (long long)(tile_r * TILE + a) * (long long)N_DIM
                        + (long long)(tile_c * TILE + b16);
    fp8x16 v = *reinterpret_cast<const fp8x16*>(&in[src]);
    *reinterpret_cast<fp8x16*>(&lds[a * LDS_STRIDE + b16]) = v;
  }
  __syncthreads();

  // ---- Write phase: LDS columns (gathered) -> output rows
  // out[(tile_c*TILE + a)][(tile_r*TILE + b16 + i)] = in[(tile_r*TILE + b16 + i)][(tile_c*TILE + a)]
  //                                                  = lds[b16 + i][a]
  {
    fp8x16 v;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
      v.v[i] = lds[(b16 + i) * LDS_STRIDE + a];
    }
    const long long dst = (long long)(tile_c * TILE + a) * (long long)M_DIM
                        + (long long)(tile_r * TILE + b16);
    *reinterpret_cast<fp8x16*>(&out[dst]) = v;
  }
}
