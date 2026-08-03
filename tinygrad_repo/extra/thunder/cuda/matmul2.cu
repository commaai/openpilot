#include "kittens.cuh"
using namespace kittens;

constexpr int g_N = 8192;

constexpr int SUPER_N = 2;
constexpr int SUPER_M = 2;
constexpr int NUM_WORKERS = SUPER_N * SUPER_M;
constexpr int LOAD_TASKS = SUPER_N + SUPER_M;

constexpr int WORKER_M = 32;
constexpr int WORKER_N = 32;

constexpr int BLOCK_K = 32;
constexpr int BLOCK_M = WORKER_M * SUPER_M;
constexpr int BLOCK_N = WORKER_N * SUPER_N;

constexpr int PIPE_STAGES = 2;

using reg_tile_A = rt_bf<WORKER_M, BLOCK_K>;
using reg_tile_B_col = rt_bf<BLOCK_K, WORKER_N, ducks::rt_layout::col>;
using reg_tile_C = rt_fl<WORKER_M, WORKER_N>;

using shared_tile_A = st_bf<WORKER_M, BLOCK_K>;
using shared_tile_B = st_bf<BLOCK_K, WORKER_N>;
using shared_tile_C = st_bf<WORKER_M, WORKER_N>;

using gl_tile_A = gl<bf16, 1, 1, g_N, g_N, shared_tile_A>;
using gl_tile_B = gl<bf16, 1, 1, g_N, g_N, shared_tile_B>;
using gl_tile_C = gl<bf16, 1, 1, g_N, g_N, shared_tile_C>;

__launch_bounds__(NUM_WORKERS *WARP_THREADS, 1) __global__
    void kernel(bf16 *c_ptr, bf16 *a_ptr, bf16 *b_ptr) {
  gl_tile_C g_C{c_ptr, nullptr, nullptr, nullptr, nullptr};
  gl_tile_A g_A{a_ptr, nullptr, nullptr, nullptr, nullptr};
  gl_tile_B g_B{b_ptr, nullptr, nullptr, nullptr, nullptr};

  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int *)&__shm[0]);

  shared_tile_A(&As)[SUPER_M][PIPE_STAGES] =
      al.allocate<shared_tile_A, SUPER_M, PIPE_STAGES>();
  shared_tile_B(&Bs)[SUPER_N][PIPE_STAGES] =
      al.allocate<shared_tile_B, SUPER_N, PIPE_STAGES>();

  reg_tile_A A_reg;
  reg_tile_B_col B_reg_col;
  reg_tile_C C_accum;

  int warpid = kittens::warpid();
  int warp_m = warpid % SUPER_M;
  int warp_n = warpid / SUPER_M;

  int load_group_id = warpgroup::groupid();

  int block_row = blockIdx.y * SUPER_M;
  int block_col = blockIdx.x * SUPER_N;

  warp::zero(C_accum);
  int num_tiles = (g_N + BLOCK_K - 1) / BLOCK_K;

  for (int load_tile = 0; load_tile < (PIPE_STAGES - 1); load_tile++) {
    if (load_tile < num_tiles) {
      int load_smem_idx = load_tile % PIPE_STAGES;
      for (int task_id = warpid; task_id < LOAD_TASKS; task_id += NUM_WORKERS) {
        if (task_id < SUPER_M) {
          warp::load_async(As[task_id][load_smem_idx], g_A, {0, 0, block_row + task_id, load_tile});
        } else {
          int n_index = task_id - SUPER_M;
          warp::load_async(Bs[n_index][load_smem_idx], g_B, {0, 0, load_tile, block_col + n_index});
        }
      }
    }
  }

  for (int tile = 0; tile < num_tiles; tile++) {
    int compute_smem_idx = tile % PIPE_STAGES;

    int load_tile = tile + PIPE_STAGES - 1;
    int load_smem_idx = load_tile % PIPE_STAGES;

    if (load_tile < num_tiles) {
      for (int task_id = warpid; task_id < LOAD_TASKS; task_id += NUM_WORKERS) {
        if (task_id < SUPER_M) {
          warp::load_async(As[task_id][load_smem_idx], g_A,
                           {0, 0, block_row + task_id, load_tile});
        } else {
          int n_index = task_id - SUPER_M;
          warp::load_async(Bs[n_index][load_smem_idx], g_B,
                           {0, 0, load_tile, block_col + n_index});
        }
      }
      load_async_wait<1>();
    } else
      load_async_wait();
    __syncthreads();

    warp::load(A_reg, As[warp_m][compute_smem_idx]);
    warp::load(B_reg_col, Bs[warp_n][compute_smem_idx]);

    warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);
    __syncthreads();
  }
  warp::store(g_C, C_accum, {0, 0, block_row + warp_m, block_col + warp_n});
}
