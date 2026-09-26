#include "kittens.cuh"

using namespace kittens;

#ifndef ROUTER_M
#define ROUTER_M 16384
#endif
#ifndef ROUTER_K
#define ROUTER_K 2880
#endif
#ifndef ROUTER_E
#define ROUTER_E 32
#endif

constexpr int BLOCK_M = 64;
constexpr int BLOCK_K = 64;
constexpr int NUM_WARPS = 4;
constexpr int THREADS = NUM_WARPS * WARP_THREADS;

using G = kittens::group<NUM_WARPS>;
using XST = st_bf<BLOCK_M, BLOCK_K, st_16x32_s>;
using WST = st_bf<ROUTER_E, BLOCK_K, st_16x32_s>;
using XRT = rt_bf<16, BLOCK_K, row_l, rt_16x32_s>;
using WRT = rt_bf<ROUTER_E, BLOCK_K, row_l, rt_16x32_s>;
using CRT = rt_fl<16, ROUTER_E, col_l, rt_16x16_s>;

static_assert(ROUTER_M % BLOCK_M == 0, "ROUTER_M must be divisible by 64");
static_assert(ROUTER_K % BLOCK_K == 0, "ROUTER_K must be divisible by 64");
static_assert(ROUTER_E == 32, "the small-N tile is specialized for 32 experts");

extern "C" __global__ __launch_bounds__(THREADS, 4) void moe_router_mfma(
    float *__restrict__ out, bf16 *__restrict__ x_ptr, bf16 *__restrict__ weight_ptr,
    bf16 *__restrict__ bias) {
  gl<bf16, 1, 1, ROUTER_M, ROUTER_K> X{x_ptr, nullptr, nullptr, nullptr, nullptr};
  gl<bf16, 1, 1, ROUTER_E, ROUTER_K> W{weight_ptr, nullptr, nullptr, nullptr, nullptr};

  __shared__ XST Xs;
  __shared__ WST Ws;

  XRT xr;
  WRT wr;
  CRT accum;
  zero(accum);

  const int block_m = __builtin_amdgcn_workgroup_id_x();
  const int warp_m = warpid();

  #pragma unroll
  for (int kk = 0; kk < ROUTER_K / BLOCK_K; kk++) {
    G::load(Xs, X, {0, 0, block_m, kk});
    G::load(Ws, W, {0, 0, 0, kk});
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    load(xr, subtile_inplace<16, BLOCK_K>(Xs, {warp_m, 0}));
    load(wr, subtile_inplace<ROUTER_E, BLOCK_K>(Ws, {0, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(accum, xr, wr, accum);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
  }

  // A 16x16 MFMA accumulator is column-layout: each lane owns four consecutive rows
  // at one column. Store all 64x32 FP32 results directly; no padded or undersized output ABI.
  const int lane = laneid();
  const int row0 = block_m * BLOCK_M + warp_m * 16 + 4 * (lane / 16);
  const int lane_col = lane % 16;
  #pragma unroll
  for (int j = 0; j < ROUTER_E / 16; j++) {
    const int col = j * 16 + lane_col;
    const float b = (float)bias[col];
    const float vals[4] = {accum.tiles[0][j].data[0].x, accum.tiles[0][j].data[0].y,
                           accum.tiles[0][j].data[1].x, accum.tiles[0][j].data[1].y};
    #pragma unroll
    for (int r = 0; r < 4; r++) out[(long long)(row0 + r) * ROUTER_E + col] = vals[r] + b;
  }
}
