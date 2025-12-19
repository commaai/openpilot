// https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/matmul/educational/level_04.cu
#include "kittens.cuh"
using namespace kittens;

constexpr int g_N = 8192;
constexpr int BLOCK_SIZE = 32;
#define NUM_WORKERS (1)

using sub_tile = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
using tile_gl =  gl<bf16, 1, 1, g_N, g_N>;

__launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void kernel(bf16 *c_ptr, bf16 *a_ptr, bf16 *b_ptr) {
  tile_gl g_C{c_ptr, nullptr, nullptr, nullptr, nullptr};
  tile_gl g_A{a_ptr, nullptr, nullptr, nullptr, nullptr};
  tile_gl g_B{b_ptr, nullptr, nullptr, nullptr, nullptr};

  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);
  st_bf<BLOCK_SIZE,BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>>();
  st_bf<BLOCK_SIZE,BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>>();

  rt_bf<BLOCK_SIZE,BLOCK_SIZE> A_reg;
  rt_bf<BLOCK_SIZE,BLOCK_SIZE> B_reg;
  rt_bf<BLOCK_SIZE,BLOCK_SIZE, ducks::rt_layout::col> B_reg_col;
  rt_fl<BLOCK_SIZE,BLOCK_SIZE> C_accum;

  int col = blockIdx.x;
  int row = blockIdx.y;

  warp::zero(C_accum);
  int num_tiles = (g_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int tile = 0; tile < num_tiles; ++tile) {
    warp::load(As, g_A, {0, 0, row, tile});
    warp::load(Bs, g_B, {0, 0, tile, col});
    __syncthreads();
    warp::load(A_reg, As);
    warp::load(B_reg, Bs);
    warp::swap_layout(B_reg_col, B_reg);
    __syncthreads();
    warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);
    __syncthreads();
  }
  warp::store(g_C, C_accum, {0, 0, row, col});
}
