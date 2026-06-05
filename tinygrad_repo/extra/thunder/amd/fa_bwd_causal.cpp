#include "kittens.cuh"
#include "utils.cpp"

#ifndef ATTN_B
constexpr int ATTN_B = 16; // batch size
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 64; // number of query heads
#endif

#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8; // number of key/value heads (for GQA)
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV; // queries per KV head group

#ifndef ATTN_N
constexpr int ATTN_N = 1024; // sequence length
#endif

constexpr int ATTN_D = 128; // dimension
constexpr int STEP_QO = 64; // block size for QO
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int SLICE_QO = 32;
constexpr int DOT_SLICE_QO = 16;
constexpr int WARP_SIZE_KV = 64; // warp size for KV
constexpr bool causal = true;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

using namespace kittens;

using _gl_QdO  = gl<bf16, ATTN_B, ATTN_N, ATTN_H, ATTN_D>;
using _gl_KV   = gl<bf16, ATTN_B, ATTN_N, ATTN_H_KV, ATTN_D>;
using _gl_dQ   = gl<bf16, ATTN_B, ATTN_H, ATTN_N, ATTN_D>;
using _gl_dKV  = gl<bf16, ATTN_B * GROUP_SIZE, ATTN_N, ATTN_H_KV, ATTN_D>;
using _gl_Lvec = gl<float, ATTN_B, ATTN_H, 1, ATTN_N>;

template<int D> struct attn_bwd_combined_globals {
  _gl_QdO Q;
  _gl_KV K, V;
  _gl_QdO dOg;
  _gl_dQ dQg;
  _gl_dKV dKg, dVg;
  _gl_Lvec L_vec, delta_vec;
  dim3 grid() { return dim3(ATTN_H, (ATTN_N / BLOCK_SIZE_KV), ATTN_B); }
  dim3 block() { return dim3(NUM_THREADS); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(bf16 *dQ_ptr, bf16 *dK_ptr, bf16 *dV_ptr, bf16 *dO_ptr, bf16 *Q_ptr, bf16 *K_ptr, bf16 *V_ptr, float *L_vec_ptr, float *delta_vec_ptr) {

  const int q_head_idx_fixed = blockIdx.x;  // This is the query head index [0, ATTN_H)
  const int kv_head_idx = q_head_idx_fixed / GROUP_SIZE;
  const int q_head_in_group = q_head_idx_fixed % GROUP_SIZE;
  const int seq_idx = blockIdx.y;
  const int batch_idx = blockIdx.z;
  const int first_q_head = q_head_idx_fixed;

  const int warpid = kittens::warpid();
  const int j = seq_idx * NUM_WARPS + warpid;

  // optimization on loops bounds
  const int total_steps_per_head = ATTN_N / STEP_QO;
  const int j_min = seq_idx * NUM_WARPS;
  const int k_start_min = j_min * WARP_SIZE_KV;
  // first Q step that can overlap this K_span:
  const int first_step = max(0, k_start_min / STEP_QO);
  const int num_steps_per_head = total_steps_per_head - first_step;
  const int num_steps = num_steps_per_head;
  const int k_pos = j * WARP_SIZE_KV;

  constexpr float L_SCALE_FACTOR = 1.44269504089f;
  constexpr float P_SCALE_FACTOR = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
  constexpr float dP_SCALE_FACTOR = (D == 128) ? 0.08838834764f : 0.125f;

  // Shared tiles
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);

  st_bf<BLOCK_SIZE_KV, D, st_16x16_s> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D, st_16x16_s>>();
  st_bf<SLICE_QO, D, st_16x32_s> (&Q_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D, st_16x32_s>, 2, 2>();
  st_bf<SLICE_QO, D, st_16x32_s> (&dO_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D, st_16x32_s>, 2, 2>();
  st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s>>();
  sv_fl<STEP_QO> (&L_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();
  sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

  // Register tiles
  using Q_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<368, 383>>, 4>; // 16 registers - a[112:127]
  using dO_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<78, 93>>, 4>; // 16 registers - v[72:87]
  using dO_col_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<94, 109>>, 4>; // 16 registers - v[88:103]
  using K_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<256, 303>, ducks::art::range<62, 77>>, 4>; // 64 registers - a[0:47] & v[56:71]
  using V_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<304, 367>>, 4>; // 64 registers - a[48:111]
  using P_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<46, 61>>, 4>; // 16 registers - v[40:55]
  using dP_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<62, 77>>, 4>; // 16 registers - v[56:71]
  using P_bf16_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<118, 125>>, 2>; // 8 registers - v[116:123]
  using dP_bf16_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<62, 69>>, 2>; // 8 registers - v[56:63]
  using P_bf16_col_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<118, 125>>, 4>; // 8 registers
  using dP_bf16_col_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<62, 69>>, 4>; // 8 registers
  using dS_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<30, 61>>, 4>; // 32 registers - v[24:55]
  using dQ_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<110, 117>>, 4>; // 8 registers - v[108:115]
  ducks::art::clobber<Q_ranges>();
  ducks::art::clobber<dO_ranges>();
  ducks::art::clobber<dO_col_ranges>();
  ducks::art::clobber<K_ranges>();
  ducks::art::clobber<V_ranges>();
  ducks::art::clobber<P_ranges>();
  ducks::art::clobber<dP_ranges>();
  ducks::art::clobber<P_bf16_ranges>();
  ducks::art::clobber<dP_bf16_ranges>();
  ducks::art::clobber<dS_ranges>();
  ducks::art::clobber<dQ_ranges>();


  using dV_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<128, 255>>, 16>; // 128 registers v[128:255]
  using dK_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<384, 511>>, 16>; // 128 registers a[128:255]
  ducks::art::clobber<dV_ranges>();
  ducks::art::clobber<dK_ranges>();

  art<bf16, DOT_SLICE_QO, D, row_l, rt_16x32_s, Q_ranges> Q_i; // 16 registers
  art<bf16, DOT_SLICE_QO, D, row_l, rt_16x32_s, dO_ranges> dO_i; // 16 registers
  art<bf16, DOT_SLICE_QO, D, col_l, rt_16x32_s, Q_ranges> Q_i_col; // 16 registers
  art<bf16, DOT_SLICE_QO, D, col_l, rt_16x32_s, dO_col_ranges> dO_i_col; // 16 registers
  art<bf16, WARP_SIZE_KV, D, row_l, rt_16x32_s, K_ranges> K_j; // 64 registers
  art<bf16, WARP_SIZE_KV, D, row_l, rt_16x32_s, V_ranges> V_j; // 64 registers
  constexpr int L_i = 126;
  constexpr int delta_i = 127;
  constexpr int neg_inf_v = 29;
  // Move -inf to VGPR neg_inf_v
  kittens::macros::clobber_gpr<neg_inf_v>();
  kittens::macros::v_mov_b32<neg_inf_v>(0xff800000);

  art<float, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, P_ranges> P_ij; // 16 registers
  art<float, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, dP_ranges> dP_ij; // 16 registers
  art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, P_bf16_ranges> P_ij_bf16; // 8 registers
  art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, dP_bf16_ranges> dP_ij_bf16; // 8 registers
  art<bf16, WARP_SIZE_KV, DOT_SLICE_QO, row_l, rt_16x16_s, ducks::art::transpose_2d<dP_bf16_ranges, 1, 4>> dP_ij_bf16_accum_row; // 8 registers

  art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s, P_bf16_col_ranges> P_ij_bf16_col; // 8 registers
  art<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s, dP_bf16_col_ranges> dP_ij_bf16_col; // 8 registers

  art<bf16, 256, 32, col_l, rt_32x16_4_s, K_ranges> K_j_col; // 64 registers // for dq
  art<bf16, 256, 16, col_l, rt_32x16_4_s, dS_ranges> dP_ij_bf16_col_T; // 32 registers // for dq

  art<float, D, WARP_SIZE_KV, col_l, rt_32x32_s, dK_ranges> dK_j_T; // 128 registers
  art<float, D, WARP_SIZE_KV, col_l, rt_32x32_s, dV_ranges> dV_j_T; // 128 registers
  art<float, 32, 16, col_l, rt_16x16_s, dQ_ranges> dQ_i_T; // 8 registers // for dq
  art<float, 16, 32, row_l, rt_16x16_s, ducks::art::transpose_2d<dQ_ranges, 2, 1>> dQ_i; // 8 registers // for dq

  // This is used for both dK_j_T and dV_j_T
  art<float, WARP_SIZE_KV, D, row_l, rt_32x32_s, ducks::art::transpose_2d<dV_ranges, 4, 2>> dV_j;

  // Construct gl objects with compile-time dims AFTER clobbers so compiler knows which VGPRs are taken
  _gl_dQ  dQg{dQ_ptr, nullptr, nullptr, nullptr, nullptr};
  _gl_dKV dKg{dK_ptr, nullptr, nullptr, nullptr, nullptr};
  _gl_dKV dVg{dV_ptr, nullptr, nullptr, nullptr, nullptr};
  _gl_QdO dOg{dO_ptr, nullptr, nullptr, nullptr, nullptr};
  _gl_QdO Q{Q_ptr, nullptr, nullptr, nullptr, nullptr};
  _gl_KV  K{K_ptr, nullptr, nullptr, nullptr, nullptr};
  _gl_KV  V{V_ptr, nullptr, nullptr, nullptr, nullptr};
  _gl_Lvec L_vec_gl{L_vec_ptr, nullptr, nullptr, nullptr, nullptr};
  _gl_Lvec delta_vec_gl{delta_vec_ptr, nullptr, nullptr, nullptr, nullptr};
  attn_bwd_combined_globals<D> g{Q, K, V, dOg, dQg, dKg, dVg, L_vec_gl, delta_vec_gl};

  // Swizzled offsets for Q and dO
  constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<bf16>();
  constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
  constexpr int memcpy_per_tile = BLOCK_SIZE_KV * DOT_SLICE_QO * sizeof(bf16) / (bytes_per_thread * NUM_THREADS);
  static_assert(BLOCK_SIZE_KV * DOT_SLICE_QO * sizeof(bf16) >= bytes_per_warp, "shared tile must be at least 1024 bytes");
  uint32_t swizzled_offsets_Q_dO[memcpy_per_tile];
  G::prefill_swizzled_offsets<1, false>(Q_i_smem[0][0], g.Q, swizzled_offsets_Q_dO);

  int tic = 0, toc = 1;

  // Load K_j from HBM to shared memory
  G::load<1, false>(K_j_smem, g.K, {batch_idx, seq_idx, kv_head_idx, 0});

  // Load V_j from HBM to registers
  load<1>(V_j, g.V, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});

  // Load Q, dO, L, delta for this specific query head
  load(L_smem[tic], g.L_vec, {batch_idx, first_q_head, 0, first_step});
  load(delta_smem[tic], g.delta_vec, {batch_idx, first_q_head, 0, first_step});
  G::load<1, false>(Q_i_smem[tic][0],  g.Q,   {batch_idx, first_step * 2 + 0, first_q_head, 0}, swizzled_offsets_Q_dO);
  G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, first_step * 2 + 0, first_q_head, 0}, swizzled_offsets_Q_dO);
  G::load<1, false>(Q_i_smem[tic][1],  g.Q,   {batch_idx, first_step * 2 + 1, first_q_head, 0}, swizzled_offsets_Q_dO);
  G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, first_step * 2 + 1, first_q_head, 0}, swizzled_offsets_Q_dO);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  // Addresses
  const uint32_t K_j_addr = get_address(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
  // Compute K_j_col_addr
  // uint32_t K_j_col_addr = get_address(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
  const uint32_t K_j_col_addr = [&] {
    const int laneid = kittens::laneid();
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<256, 32>(K_j_smem, {0, warpid}).data[0]);
    const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
    const int col_offset = ((laneid % 4) * 4);
    const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(bf16);
    const uint32_t addr = src_ptr + lane_byte_offset;
    return addr;
  }();

  auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
  const uint32_t dP_ij_bf16_accum_row_addr = get_address(attn_i_smem_subtile, dP_ij_bf16_accum_row);

  uint32_t Q_i_addr;
  uint32_t dO_i_addr;
  uint32_t dO_i_col_addr;
  uint32_t Q_i_col_addr;

  // Compute dP_ij_bf16_col_T_addr
  // const uint32_t dP_ij_bf16_col_T_addr = [&] {
  //   const int laneid = kittens::laneid();
  //   const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&attn_i_smem.data[0]);
  //   const int row_offset = (laneid % 16) / 4 + (laneid / 16) * 4;
  //   const int col_offset = ((laneid % 4) * 4);
  //   const int lane_byte_offset = (row_offset * 16 + col_offset) * sizeof(bf16);
  //   const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 7) << 3);
  //   const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
  //   return addr;
  // }();
  uint32_t dP_ij_bf16_col_T_addr = get_address(dP_ij_bf16_col_T, attn_i_smem);

  if (num_steps > 1) {
    // Prologue
    {
      const int q_head_idx = (0) / num_steps_per_head + first_q_head;
      const int q_seq_idx = ((0) % num_steps_per_head) + first_step;
      const int q_pos = q_seq_idx * STEP_QO;

      const int next_q_head_idx = (0 + 1) / num_steps_per_head + first_q_head;
      const int next_q_seq_idx = ((0 + 1) % num_steps_per_head) + first_step;

      // dot slice 0
      {
        load(L_smem[toc], g.L_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
        G::load<1, false>(Q_i_smem[toc][0], g.Q, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});

        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // mma_ABt(P_ij, Q_i, K_j);
        mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
        load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
        load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
        mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
        load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
        mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
        load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
        // Load dO_i from shared memory to registers
        // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 0, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
        // Load dO_i_col from shared memory to registers
        // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        // Compute dO_i_col_addr
        // dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_col_addr = [&] {
          const int laneid = kittens::laneid();
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 1, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
        // Dot slice 0
        kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
          // If the query position is less than the key position, set P_ij to -inf
          if (q_pos < k_pos) {
            mov<neg_inf_v>(P_ij);
          // If the query position is equal to the key position, we need to apply a causal mask
          } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 0] and set [0, 1:4] to -inf
            make_causal<0, 0, neg_inf_v>(P_ij, P_ij);
            mov<0, 1, neg_inf_v>(P_ij);
            mov<0, 2, neg_inf_v>(P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
          }
        }
        mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_ABt(dP_ij, dO_i, V_j);
        mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
        sub_row<0, 2, L_i>(P_ij, P_ij);
        mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 0>(P_ij, P_ij);
        mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
        // Load Q_i_col from shared memory to registers
        // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        // Compute Q_i_col_addr
        // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_col_addr = [&] {
          const int laneid = kittens::laneid();  
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const int addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 1>(P_ij, P_ij);
        mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
        load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
        mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
        sub_row<0, 3, L_i>(P_ij, P_ij);
        mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 0>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
        exp2<0, 2>(P_ij, P_ij);
        mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 1>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
        load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 3>(P_ij, P_ij);
        mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
        load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 2>(P_ij_bf16, P_ij);
        copy<0, 3>(P_ij_bf16, P_ij);
        mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
        swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
        mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
        mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col);
        // Load K_j_col from shared memory to registers
        // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col);
        sub_row<0, 0, delta_i>(dP_ij, dP_ij);
        sub_row<0, 1, delta_i>(dP_ij, dP_ij);
        mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col);
        load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col);
        mul<0, 0>(dP_ij, dP_ij, P_ij);
        mul<0, 1>(dP_ij, dP_ij, P_ij);
        copy<0, 0>(dP_ij_bf16, dP_ij);
        copy<0, 1>(dP_ij_bf16, dP_ij);
        sub_row<0, 2, delta_i>(dP_ij, dP_ij);
        mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col);
        load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)

        // Store dP_ij_bf16_accum_row to shared memory
        // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col);
        sub_row<0, 3, delta_i>(dP_ij, dP_ij);
        mul<0, 2>(dP_ij, dP_ij, P_ij);
        mul<0, 3>(dP_ij, dP_ij, P_ij);
        copy<0, 2>(dP_ij_bf16, dP_ij);
        copy<0, 3>(dP_ij_bf16, dP_ij);
        mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col);

        // dot slice 1
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
        
        store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col);
        swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(12)");
        // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col);
        load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col);
        // Load dP_ij_bf16_col_T from shared memory to registers
        // load(dP_ij_bf16_col_T, attn_i_smem);
        load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col);
        load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col);
        load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col);
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(6)");
        __builtin_amdgcn_s_barrier();
        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0}, swizzled_offsets_Q_dO);
        mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
        mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(10)");
        mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(2)");
      }

      // dot slice 1
      {
        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // mma_ABt(P_ij, Q_i, K_j);
        mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
        load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
        mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
        load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
        load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
        load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
        // Load dO_i from shared memory to registers
        // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
        load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 0, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
        // Load dO_i_col from shared memory to registers
        // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        // Compute dO_i_col_addr
        // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_col_addr = [&] {
          const int laneid = kittens::laneid();
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 1, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
        // Dot slice 1
        kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
          // If the query position is less than the key position, set P_ij to -inf
          if (q_pos < k_pos) {
            mov<neg_inf_v>(P_ij);
          // If the query position is equal to the key position, we need to apply a causal mask
          } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 1] and set [0, 2:4] to -inf
            make_causal<0, 1, neg_inf_v>(P_ij, P_ij);
            mov<0, 2, neg_inf_v>(P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
          }
        }
        mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_ABt(dP_ij, dO_i, V_j);
        mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
        sub_row<0, 2, L_i>(P_ij, P_ij);
        mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 0>(P_ij, P_ij);
        mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
        // Load Q_i_col from shared memory to registers
        // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        // Compute Q_i_col_addr
        // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_col_addr = [&] {
          const int laneid = kittens::laneid();  
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const int addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 1>(P_ij, P_ij);
        mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
        load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
        mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
        sub_row<0, 3, L_i>(P_ij, P_ij);
        mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 0>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
        exp2<0, 2>(P_ij, P_ij);
        mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 1>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
        load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 3>(P_ij, P_ij);
        mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
        load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 2>(P_ij_bf16, P_ij);
        copy<0, 3>(P_ij_bf16, P_ij);
        mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
        swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
        mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
        mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        // Load K_j_col from shared memory to registers
        // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 0, delta_i>(dP_ij, dP_ij);
        sub_row<0, 1, delta_i>(dP_ij, dP_ij);
        mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        mul<0, 0>(dP_ij, dP_ij, P_ij);
        mul<0, 1>(dP_ij, dP_ij, P_ij);
        copy<0, 0>(dP_ij_bf16, dP_ij);
        copy<0, 1>(dP_ij_bf16, dP_ij);
        sub_row<0, 2, delta_i>(dP_ij, dP_ij);
        mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        // Store dP_ij_bf16_accum_row to shared memory
        // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 3, delta_i>(dP_ij, dP_ij);
        mul<0, 2>(dP_ij, dP_ij, P_ij);
        mul<0, 3>(dP_ij, dP_ij, P_ij);
        copy<0, 2>(dP_ij_bf16, dP_ij);
        copy<0, 3>(dP_ij_bf16, dP_ij);
        mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

        // dot slice 2
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
        
        store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(12)");
        // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        // Load dP_ij_bf16_col_T from shared memory to registers
        // load(dP_ij_bf16_col_T, attn_i_smem);
        load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
        mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
        mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt vmcnt(4) lgkmcnt(6)");
        __builtin_amdgcn_s_barrier();
        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0}, swizzled_offsets_Q_dO);
        mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(10)");
        mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(2)");
      }

      // dot slice 2
      {
        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // mma_ABt(P_ij, Q_i, K_j);
        mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
        load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
        mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
        load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
        load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
        load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
        // Load dO_i from shared memory to registers
        // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
        load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 0, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
        // Load dO_i_col from shared memory to registers
        // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        // Compute dO_i_col_addr
        // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_col_addr = [&] {
          const int laneid = kittens::laneid();
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 1, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
        // Dot slice 2
        kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
          // If the query position is less than the key position, set P_ij to -inf
          if (q_pos < k_pos) {
            mov<neg_inf_v>(P_ij);
          // If the query position is equal to the key position, we need to apply a causal mask
          } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 2] and set [0, 3:4] to -inf
            make_causal<0, 2, neg_inf_v>(P_ij, P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
          }
        }
        mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_ABt(dP_ij, dO_i, V_j);
        mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
        sub_row<0, 2, L_i>(P_ij, P_ij);
        mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 0>(P_ij, P_ij);
        mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
        // Load Q_i_col from shared memory to registers
        // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        // Compute Q_i_col_addr
        // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_col_addr = [&] {
          const int laneid = kittens::laneid();  
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const int addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 1>(P_ij, P_ij);
        mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
        load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
        mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
        sub_row<0, 3, L_i>(P_ij, P_ij);
        mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 0>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
        exp2<0, 2>(P_ij, P_ij);
        mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 1>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
        load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 3>(P_ij, P_ij);
        mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
        load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 2>(P_ij_bf16, P_ij);
        copy<0, 3>(P_ij_bf16, P_ij);
        mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
        swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
        mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
        mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        // Load K_j_col from shared memory to registers
        // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 0, delta_i>(dP_ij, dP_ij);
        sub_row<0, 1, delta_i>(dP_ij, dP_ij);
        mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        mul<0, 0>(dP_ij, dP_ij, P_ij);
        mul<0, 1>(dP_ij, dP_ij, P_ij);
        copy<0, 0>(dP_ij_bf16, dP_ij);
        copy<0, 1>(dP_ij_bf16, dP_ij);
        sub_row<0, 2, delta_i>(dP_ij, dP_ij);
        mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        // Store dP_ij_bf16_accum_row to shared memory
        // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 3, delta_i>(dP_ij, dP_ij);
        mul<0, 2>(dP_ij, dP_ij, P_ij);
        mul<0, 3>(dP_ij, dP_ij, P_ij);
        copy<0, 2>(dP_ij_bf16, dP_ij);
        copy<0, 3>(dP_ij_bf16, dP_ij);
        mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

        // dot slice 3
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
        
        store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(12)");
        // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        // Load dP_ij_bf16_col_T from shared memory to registers
        // load(dP_ij_bf16_col_T, attn_i_smem);
        load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
        mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
        mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt vmcnt(4) lgkmcnt(6)");
        __builtin_amdgcn_s_barrier();
        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0}, swizzled_offsets_Q_dO);
        mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(10)");
        mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(2)");
      }

      // dot slice 3
      {
        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // mma_ABt(P_ij, Q_i, K_j);
        mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
        load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
        mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
        load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
        load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
        load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
        // Load dO_i from shared memory to registers
        // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
        load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 0, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
        // Load dO_i_col from shared memory to registers
        // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        // Compute dO_i_col_addr
        // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_col_addr = [&] {
          const int laneid = kittens::laneid();
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 1, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
        // Dot slice 3
        kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
          // If the query position is less than the key position, set P_ij to -inf
          if (q_pos < k_pos) {
            mov<neg_inf_v>(P_ij);
          // If the query position is equal to the key position, we need to apply a causal mask
          } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 3]
            make_causal<0, 3, neg_inf_v>(P_ij, P_ij);
          }
        }
        mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_ABt(dP_ij, dO_i, V_j);
        mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
        sub_row<0, 2, L_i>(P_ij, P_ij);
        mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 0>(P_ij, P_ij);
        mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
        // Load Q_i_col from shared memory to registers
        // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        // Compute Q_i_col_addr
        // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_col_addr = [&] {
          const int laneid = kittens::laneid();  
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const int addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 1>(P_ij, P_ij);
        mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
        load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
        mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
        sub_row<0, 3, L_i>(P_ij, P_ij);
        mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 0>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
        exp2<0, 2>(P_ij, P_ij);
        mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 1>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
        load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 3>(P_ij, P_ij);
        mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
        load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 2>(P_ij_bf16, P_ij);
        copy<0, 3>(P_ij_bf16, P_ij);
        mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
        swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
        mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
        mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        // Load K_j_col from shared memory to registers
        // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 0, delta_i>(dP_ij, dP_ij);
        sub_row<0, 1, delta_i>(dP_ij, dP_ij);
        mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        mul<0, 0>(dP_ij, dP_ij, P_ij);
        mul<0, 1>(dP_ij, dP_ij, P_ij);
        copy<0, 0>(dP_ij_bf16, dP_ij);
        copy<0, 1>(dP_ij_bf16, dP_ij);
        sub_row<0, 2, delta_i>(dP_ij, dP_ij);
        mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        // Store dP_ij_bf16_accum_row to shared memory
        // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 3, delta_i>(dP_ij, dP_ij);
        mul<0, 2>(dP_ij, dP_ij, P_ij);
        mul<0, 3>(dP_ij, dP_ij, P_ij);
        copy<0, 2>(dP_ij_bf16, dP_ij);
        copy<0, 3>(dP_ij_bf16, dP_ij);
        mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

        // dot slice 0 - next iteration
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[toc], 0));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[toc], 0));
        
        store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(12)");
        // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        // Load dP_ij_bf16_col_T from shared memory to registers
        // load(dP_ij_bf16_col_T, attn_i_smem);
        load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
        mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
        mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt vmcnt(4) lgkmcnt(6)");
        __builtin_amdgcn_s_barrier();
        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(10)");
        mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(2)");
      }
      tic ^= 1; toc ^= 1;
    }

    // 9. for 1 <= i <= T_r (1024 / 32 = 32)  
    for (int i = 1; i < num_steps - 1; ++i, tic ^= 1, toc ^= 1) {
      const int last_q_head_idx = (i - 1) / num_steps_per_head + first_q_head;
      const int last_q_seq_idx = ((i - 1) % num_steps_per_head) + first_step;

      const int q_head_idx = i / num_steps_per_head + first_q_head;
      const int q_seq_idx = (i % num_steps_per_head) + first_step;
      const int q_pos = q_seq_idx * STEP_QO;

      const int next_q_head_idx = (i + 1) / num_steps_per_head + first_q_head;
      const int next_q_seq_idx = ((i + 1) % num_steps_per_head) + first_step;

      // dot slice 0
      {
        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // mma_ABt(P_ij, Q_i, K_j);
        G::load<1, false>(Q_i_smem[toc][0], g.Q, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0}, swizzled_offsets_Q_dO);
        mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
        load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
        load(L_smem[toc], g.L_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
        mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
        load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
        load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
        load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
        // Load dO_i from shared memory to registers
        // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 0, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
        // Load dO_i_col from shared memory to registers
        // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        // Compute dO_i_col_addr
        // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_col_addr = [&] {
          const int laneid = kittens::laneid();
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 1, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
        // Dot slice 0
        kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
          // If the query position is less than the key position, set P_ij to -inf
          if (q_pos < k_pos) {
            mov<neg_inf_v>(P_ij);
          // If the query position is equal to the key position, we need to apply a causal mask
          } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 0] and set [0, 1:4] to -inf
            make_causal<0, 0, neg_inf_v>(P_ij, P_ij);
            mov<0, 1, neg_inf_v>(P_ij);
            mov<0, 2, neg_inf_v>(P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
          }
        }
        mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_ABt(dP_ij, dO_i, V_j);
        mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
        sub_row<0, 2, L_i>(P_ij, P_ij);
        mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 0>(P_ij, P_ij);
        mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
        // Load Q_i_col from shared memory to registers
        // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        // Compute Q_i_col_addr
        // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_col_addr = [&] {
          const int laneid = kittens::laneid();  
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const int addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 1>(P_ij, P_ij);
        mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
        load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
        mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
        sub_row<0, 3, L_i>(P_ij, P_ij);
        mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 0>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
        exp2<0, 2>(P_ij, P_ij);
        mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 1>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
        load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 3>(P_ij, P_ij);
        mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
        load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 2>(P_ij_bf16, P_ij);
        copy<0, 3>(P_ij_bf16, P_ij);
        mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
        swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
        mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
        mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        // Load K_j_col from shared memory to registers
        // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 0, delta_i>(dP_ij, dP_ij);
        sub_row<0, 1, delta_i>(dP_ij, dP_ij);
        mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        mul<0, 0>(dP_ij, dP_ij, P_ij);
        mul<0, 1>(dP_ij, dP_ij, P_ij);
        copy<0, 0>(dP_ij_bf16, dP_ij);
        copy<0, 1>(dP_ij_bf16, dP_ij);
        sub_row<0, 2, delta_i>(dP_ij, dP_ij);
        mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        // Store dP_ij_bf16_accum_row to shared memory
        // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 3, delta_i>(dP_ij, dP_ij);
        mul<0, 2>(dP_ij, dP_ij, P_ij);
        mul<0, 3>(dP_ij, dP_ij, P_ij);
        copy<0, 2>(dP_ij_bf16, dP_ij);
        copy<0, 3>(dP_ij_bf16, dP_ij);
        mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

        // dot slice 1
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
        
        store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(12)");
        // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        // Load dP_ij_bf16_col_T from shared memory to registers
        // load(dP_ij_bf16_col_T, attn_i_smem);
        load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);
        mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);
        mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt vmcnt(4) lgkmcnt(6)");
        __builtin_amdgcn_s_barrier();
        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0}, swizzled_offsets_Q_dO);
        mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
        mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(10)");
        mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);

        asm volatile("s_waitcnt lgkmcnt(2)");
      }

      // dot slice 1
      {
        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // mma_ABt(P_ij, Q_i, K_j);
        mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
        load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
        mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
        load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
        load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
        load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
        // Load dO_i from shared memory to registers
        // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
        load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 0, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
        // Load dO_i_col from shared memory to registers
        // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        // Compute dO_i_col_addr
        // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_col_addr = [&] {
          const int laneid = kittens::laneid();
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 1, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
        // Dot slice 1
        kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
          // If the query position is less than the key position, set P_ij to -inf
          if (q_pos < k_pos) {
            mov<neg_inf_v>(P_ij);
          // If the query position is equal to the key position, we need to apply a causal mask
          } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 1] and set [0, 2:4] to -inf
            make_causal<0, 1, neg_inf_v>(P_ij, P_ij);
            mov<0, 2, neg_inf_v>(P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
          }
        }
        mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_ABt(dP_ij, dO_i, V_j);
        mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
        sub_row<0, 2, L_i>(P_ij, P_ij);
        mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 0>(P_ij, P_ij);
        mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
        // Load Q_i_col from shared memory to registers
        // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        // Compute Q_i_col_addr
        // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_col_addr = [&] {
          const int laneid = kittens::laneid();  
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const int addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 1>(P_ij, P_ij);
        mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
        load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
        mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
        sub_row<0, 3, L_i>(P_ij, P_ij);
        mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 0>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
        exp2<0, 2>(P_ij, P_ij);
        mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 1>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
        load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 3>(P_ij, P_ij);
        mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
        load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 2>(P_ij_bf16, P_ij);
        copy<0, 3>(P_ij_bf16, P_ij);
        mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
        swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
        mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
        mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        // Load K_j_col from shared memory to registers
        // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 0, delta_i>(dP_ij, dP_ij);
        sub_row<0, 1, delta_i>(dP_ij, dP_ij);
        mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        mul<0, 0>(dP_ij, dP_ij, P_ij);
        mul<0, 1>(dP_ij, dP_ij, P_ij);
        copy<0, 0>(dP_ij_bf16, dP_ij);
        copy<0, 1>(dP_ij_bf16, dP_ij);
        sub_row<0, 2, delta_i>(dP_ij, dP_ij);
        mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        // Store dP_ij_bf16_accum_row to shared memory
        // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 3, delta_i>(dP_ij, dP_ij);
        mul<0, 2>(dP_ij, dP_ij, P_ij);
        mul<0, 3>(dP_ij, dP_ij, P_ij);
        copy<0, 2>(dP_ij_bf16, dP_ij);
        copy<0, 3>(dP_ij_bf16, dP_ij);
        mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

        // dot slice 2
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
        
        store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(12)");
        // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        // Load dP_ij_bf16_col_T from shared memory to registers
        // load(dP_ij_bf16_col_T, attn_i_smem);
        load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 0, 0}, warpid);
        mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 0, 0}, warpid);
        mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt vmcnt(4) lgkmcnt(6)");
        __builtin_amdgcn_s_barrier();
        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0}, swizzled_offsets_Q_dO);
        mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(10)");
        mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(2)");
      }

      // dot slice 2
      {
        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // mma_ABt(P_ij, Q_i, K_j);
        mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
        load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
        mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
        load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
        load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
        load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
        // Load dO_i from shared memory to registers
        // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
        load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 0, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
        // Load dO_i_col from shared memory to registers
        // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        // Compute dO_i_col_addr
        // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_col_addr = [&] {
          const int laneid = kittens::laneid();
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 1, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
        // Dot slice 2
        kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
          // If the query position is less than the key position, set P_ij to -inf
          if (q_pos < k_pos) {
            mov<neg_inf_v>(P_ij);
          // If the query position is equal to the key position, we need to apply a causal mask
          } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 2] and set [0, 3:4] to -inf
            make_causal<0, 2, neg_inf_v>(P_ij, P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
          }
        }
        mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_ABt(dP_ij, dO_i, V_j);
        mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
        sub_row<0, 2, L_i>(P_ij, P_ij);
        mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 0>(P_ij, P_ij);
        mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
        // Load Q_i_col from shared memory to registers
        // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        // Compute Q_i_col_addr
        // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_col_addr = [&] {
          const int laneid = kittens::laneid();  
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const int addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 1>(P_ij, P_ij);
        mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
        load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
        mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
        sub_row<0, 3, L_i>(P_ij, P_ij);
        mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 0>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
        exp2<0, 2>(P_ij, P_ij);
        mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 1>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
        load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 3>(P_ij, P_ij);
        mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
        load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 2>(P_ij_bf16, P_ij);
        copy<0, 3>(P_ij_bf16, P_ij);
        mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
        swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
        mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
        mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        // Load K_j_col from shared memory to registers
        // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 0, delta_i>(dP_ij, dP_ij);
        sub_row<0, 1, delta_i>(dP_ij, dP_ij);
        mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        mul<0, 0>(dP_ij, dP_ij, P_ij);
        mul<0, 1>(dP_ij, dP_ij, P_ij);
        copy<0, 0>(dP_ij_bf16, dP_ij);
        copy<0, 1>(dP_ij_bf16, dP_ij);
        sub_row<0, 2, delta_i>(dP_ij, dP_ij);
        mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        // Store dP_ij_bf16_accum_row to shared memory
        // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 3, delta_i>(dP_ij, dP_ij);
        mul<0, 2>(dP_ij, dP_ij, P_ij);
        mul<0, 3>(dP_ij, dP_ij, P_ij);
        copy<0, 2>(dP_ij_bf16, dP_ij);
        copy<0, 3>(dP_ij_bf16, dP_ij);
        mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

        // dot slice 3
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
        
        store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(12)");
        // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        // Load dP_ij_bf16_col_T from shared memory to registers
        // load(dP_ij_bf16_col_T, attn_i_smem);
        load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
        mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
        mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt vmcnt(4) lgkmcnt(6)");
        __builtin_amdgcn_s_barrier();
        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0}, swizzled_offsets_Q_dO);
        mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(10)");
        mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(2)");
      }

      // dot slice 3
      {
        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        // mma_ABt(P_ij, Q_i, K_j);
        mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
        load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
        mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
        load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
        load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
        mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
        load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
        // Load dO_i from shared memory to registers
        // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
        load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 0, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
        mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
        mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(6)");
        mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
        // Load dO_i_col from shared memory to registers
        // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        // Compute dO_i_col_addr
        // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        dO_i_col_addr = [&] {
          const int laneid = kittens::laneid();
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
        sub_row<0, 1, L_i>(P_ij, P_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
        load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
        mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
        // Dot slice 3
        kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
          // If the query position is less than the key position, set P_ij to -inf
          if (q_pos < k_pos) {
            mov<neg_inf_v>(P_ij);
          // If the query position is equal to the key position, we need to apply a causal mask
          } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 3]
            make_causal<0, 3, neg_inf_v>(P_ij, P_ij);
          }
        }
        mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_ABt(dP_ij, dO_i, V_j);
        mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
        sub_row<0, 2, L_i>(P_ij, P_ij);
        mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 0>(P_ij, P_ij);
        mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
        // Load Q_i_col from shared memory to registers
        // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        // Compute Q_i_col_addr
        // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        Q_i_col_addr = [&] {
          const int laneid = kittens::laneid();  
          const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}).data[0]);
          const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
          const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
          const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
          const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
          const int addr = src_ptr + swizzled_lane_byte_offset;
          return addr;
        }();
        load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 1>(P_ij, P_ij);
        mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
        load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
        mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
        mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
        sub_row<0, 3, L_i>(P_ij, P_ij);
        mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 0>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
        exp2<0, 2>(P_ij, P_ij);
        mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 1>(P_ij_bf16, P_ij);
        mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
        load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
        exp2<0, 3>(P_ij, P_ij);
        mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
        load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
        mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
        copy<0, 2>(P_ij_bf16, P_ij);
        copy<0, 3>(P_ij_bf16, P_ij);
        mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
        swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
        mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
        asm volatile("s_waitcnt lgkmcnt(8)");
        // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
        mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        // Load K_j_col from shared memory to registers
        // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 0, delta_i>(dP_ij, dP_ij);
        sub_row<0, 1, delta_i>(dP_ij, dP_ij);
        mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        mul<0, 0>(dP_ij, dP_ij, P_ij);
        mul<0, 1>(dP_ij, dP_ij, P_ij);
        copy<0, 0>(dP_ij_bf16, dP_ij);
        copy<0, 1>(dP_ij_bf16, dP_ij);
        sub_row<0, 2, delta_i>(dP_ij, dP_ij);
        mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        // Store dP_ij_bf16_accum_row to shared memory
        // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        sub_row<0, 3, delta_i>(dP_ij, dP_ij);
        mul<0, 2>(dP_ij, dP_ij, P_ij);
        mul<0, 3>(dP_ij, dP_ij, P_ij);
        copy<0, 2>(dP_ij_bf16, dP_ij);
        copy<0, 3>(dP_ij_bf16, dP_ij);
        mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

        // dot slice 0 - next iteration
        load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[toc], 0));
        load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[toc], 0));
        
        store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
        mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(12)");
        // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
        mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        // Load dP_ij_bf16_col_T from shared memory to registers
        // load(dP_ij_bf16_col_T, attn_i_smem);
        load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        mul<L_i, L_i>(L_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
        mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
        mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
        load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt vmcnt(4) lgkmcnt(6)");
        __builtin_amdgcn_s_barrier();
        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
        mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        // Load Q_i from shared memory to registers
        // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}));
        Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}));
        load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}), Q_i_addr);
        load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(4)");
        __builtin_amdgcn_s_barrier();
        mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}), Q_i_addr);
        load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[toc][0], {0, 0}), Q_i_addr);
        mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
        // Load K_j from shared memory to registers
        // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
        asm volatile("s_waitcnt lgkmcnt(10)");
        mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
        mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
        asm volatile("s_waitcnt lgkmcnt(2)");
      }
    }
  }

  const int last_q_head_idx = (num_steps - 2) / num_steps_per_head + first_q_head;
  const int last_q_seq_idx = ((num_steps - 2) % num_steps_per_head) + first_step;

  const int q_head_idx = (num_steps - 1) / num_steps_per_head + first_q_head;
  const int q_seq_idx = ((num_steps - 1) % num_steps_per_head) + first_step;
  const int q_pos = q_seq_idx * STEP_QO;
  // Epilogue
  {
    // dot slice 0
    {

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp2(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      // mma_ABt(P_ij, Q_i, K_j);
      mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
      load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
      mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
      load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
      mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
      load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
      mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
      load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
      mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
      // Load dO_i from shared memory to registers
      // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
      sub_row<0, 0, L_i>(P_ij, P_ij);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
      load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
      mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
      // Load dO_i_col from shared memory to registers
      // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      // Compute dO_i_col_addr
      // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      dO_i_col_addr = [&] {
        const int laneid = kittens::laneid();
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
        const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
        const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
        return addr;
      }();
      load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
      sub_row<0, 1, L_i>(P_ij, P_ij);
      asm volatile("s_waitcnt lgkmcnt(8)");
      mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
      load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
      // Dot slice 0
      kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
        // If the query position is less than the key position, set P_ij to -inf
        if (q_pos < k_pos) {
          mov<neg_inf_v>(P_ij);
        // If the query position is equal to the key position, we need to apply a causal mask
        } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 0] and set [0, 1:4] to -inf
            make_causal<0, 0, neg_inf_v>(P_ij, P_ij);
            mov<0, 1, neg_inf_v>(P_ij);
            mov<0, 2, neg_inf_v>(P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
        }
      }
      mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(8)");
      // mma_ABt(dP_ij, dO_i, V_j);
      mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
      sub_row<0, 2, L_i>(P_ij, P_ij);
      mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 0>(P_ij, P_ij);
      mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
      // Load Q_i_col from shared memory to registers
      // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      // Compute Q_i_col_addr
      // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      Q_i_col_addr = [&] {
        const int laneid = kittens::laneid();  
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
        const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
        const int addr = src_ptr + swizzled_lane_byte_offset;
        return addr;
      }();
      load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 1>(P_ij, P_ij);
      mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
      load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
      mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
      mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
      sub_row<0, 3, L_i>(P_ij, P_ij);
      mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 0>(P_ij_bf16, P_ij);
      mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
      exp2<0, 2>(P_ij, P_ij);
      mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 1>(P_ij_bf16, P_ij);
      mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
      load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 3>(P_ij, P_ij);
      mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
      load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 2>(P_ij_bf16, P_ij);
      copy<0, 3>(P_ij_bf16, P_ij);
      mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(8)");
      // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
      mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      // Load K_j_col from shared memory to registers
      // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      sub_row<0, 0, delta_i>(dP_ij, dP_ij);
      sub_row<0, 1, delta_i>(dP_ij, dP_ij);
      mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      mul<0, 0>(dP_ij, dP_ij, P_ij);
      mul<0, 1>(dP_ij, dP_ij, P_ij);
      copy<0, 0>(dP_ij_bf16, dP_ij);
      copy<0, 1>(dP_ij_bf16, dP_ij);
      sub_row<0, 2, delta_i>(dP_ij, dP_ij);
      mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      // Store dP_ij_bf16_accum_row to shared memory
      // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      sub_row<0, 3, delta_i>(dP_ij, dP_ij);
      mul<0, 2>(dP_ij, dP_ij, P_ij);
      mul<0, 3>(dP_ij, dP_ij, P_ij);
      copy<0, 2>(dP_ij_bf16, dP_ij);
      copy<0, 3>(dP_ij_bf16, dP_ij);
      mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

      // dot slice 1
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
      
      store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      asm volatile("s_waitcnt lgkmcnt(12)");
      // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
      mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(8)");
      __builtin_amdgcn_s_barrier();
      mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      // Load dP_ij_bf16_col_T from shared memory to registers
      // load(dP_ij_bf16_col_T, attn_i_smem);
      load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      mul<L_i, L_i>(L_SCALE_FACTOR);
      if (num_steps > 1) {
        atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);
      }
      mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      if (num_steps > 1) {
        atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);
      }
      mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(6)");
      __builtin_amdgcn_s_barrier();
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      // Load Q_i from shared memory to registers
      // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(4)");
      __builtin_amdgcn_s_barrier();
      mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      // Load K_j from shared memory to registers
      // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(10)");
      mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(2)");
    }

    // dot slice 1
    {
      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp2(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      // mma_ABt(P_ij, Q_i, K_j);
      mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
      load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
      mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
      load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
      mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
      load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
      mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
      load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
      mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
      // Load dO_i from shared memory to registers
      // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
      sub_row<0, 0, L_i>(P_ij, P_ij);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
      load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
      mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
      // Load dO_i_col from shared memory to registers
      // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      // Compute dO_i_col_addr
      // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      dO_i_col_addr = [&] {
        const int laneid = kittens::laneid();
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
        const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
        const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
        return addr;
      }();
      load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
      sub_row<0, 1, L_i>(P_ij, P_ij);
      asm volatile("s_waitcnt lgkmcnt(8)");
      mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
      load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
      // Dot slice 1
      kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
        // If the query position is less than the key position, set P_ij to -inf
        if (q_pos < k_pos) {
          mov<neg_inf_v>(P_ij);
        // If the query position is equal to the key position, we need to apply a causal mask
        } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 1] and set [0, 2:4] to -inf
            make_causal<0, 1, neg_inf_v>(P_ij, P_ij);
            mov<0, 2, neg_inf_v>(P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
        }
      }
      mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(8)");
      // mma_ABt(dP_ij, dO_i, V_j);
      mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
      sub_row<0, 2, L_i>(P_ij, P_ij);
      mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 0>(P_ij, P_ij);
      mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
      // Load Q_i_col from shared memory to registers
      // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      // Compute Q_i_col_addr
      // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      Q_i_col_addr = [&] {
        const int laneid = kittens::laneid();  
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
        const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
        const int addr = src_ptr + swizzled_lane_byte_offset;
        return addr;
      }();
      load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 1>(P_ij, P_ij);
      mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
      load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
      mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
      mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
      sub_row<0, 3, L_i>(P_ij, P_ij);
      mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 0>(P_ij_bf16, P_ij);
      mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
      exp2<0, 2>(P_ij, P_ij);
      mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 1>(P_ij_bf16, P_ij);
      mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
      load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 3>(P_ij, P_ij);
      mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
      load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 2>(P_ij_bf16, P_ij);
      copy<0, 3>(P_ij_bf16, P_ij);
      mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(8)");
      // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
      mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      // Load K_j_col from shared memory to registers
      // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      sub_row<0, 0, delta_i>(dP_ij, dP_ij);
      sub_row<0, 1, delta_i>(dP_ij, dP_ij);
      mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      mul<0, 0>(dP_ij, dP_ij, P_ij);
      mul<0, 1>(dP_ij, dP_ij, P_ij);
      copy<0, 0>(dP_ij_bf16, dP_ij);
      copy<0, 1>(dP_ij_bf16, dP_ij);
      sub_row<0, 2, delta_i>(dP_ij, dP_ij);
      mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      // Store dP_ij_bf16_accum_row to shared memory
      // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      sub_row<0, 3, delta_i>(dP_ij, dP_ij);
      mul<0, 2>(dP_ij, dP_ij, P_ij);
      mul<0, 3>(dP_ij, dP_ij, P_ij);
      copy<0, 2>(dP_ij_bf16, dP_ij);
      copy<0, 3>(dP_ij_bf16, dP_ij);
      mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

      // dot slice 2
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
      
      store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      asm volatile("s_waitcnt lgkmcnt(12)");
      // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
      mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(8)");
      __builtin_amdgcn_s_barrier();
      mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      // Load dP_ij_bf16_col_T from shared memory to registers
      // load(dP_ij_bf16_col_T, attn_i_smem);
      load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      mul<L_i, L_i>(L_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
      mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
      mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(6)");
      __builtin_amdgcn_s_barrier();
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      // Load Q_i from shared memory to registers
      // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(4)");
      __builtin_amdgcn_s_barrier();
      mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      // Load K_j from shared memory to registers
      // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(10)");
      mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(2)");
    }

    // dot slice 2
    {
      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp2(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      // mma_ABt(P_ij, Q_i, K_j);
      mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
      load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
      mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
      load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
      mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
      load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
      mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
      load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
      mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
      // Load dO_i from shared memory to registers
      // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
      sub_row<0, 0, L_i>(P_ij, P_ij);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
      load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
      mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
      // Load dO_i_col from shared memory to registers
      // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      // Compute dO_i_col_addr
      // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      dO_i_col_addr = [&] {
        const int laneid = kittens::laneid();
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
        const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
        const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
        return addr;
      }();
      load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
      sub_row<0, 1, L_i>(P_ij, P_ij);
      asm volatile("s_waitcnt lgkmcnt(8)");
      mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
      load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
      // Dot slice 2
      kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
        // If the query position is less than the key position, set P_ij to -inf
        if (q_pos < k_pos) {
          mov<neg_inf_v>(P_ij);
        // If the query position is equal to the key position, we need to apply a causal mask
        } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 2] and set [0, 3:4] to -inf
            make_causal<0, 2, neg_inf_v>(P_ij, P_ij);
            mov<0, 3, neg_inf_v>(P_ij);
        }
      }
      mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(8)");
      // mma_ABt(dP_ij, dO_i, V_j);
      mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
      sub_row<0, 2, L_i>(P_ij, P_ij);
      mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 0>(P_ij, P_ij);
      mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
      // Load Q_i_col from shared memory to registers
      // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      // Compute Q_i_col_addr
      // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      Q_i_col_addr = [&] {
        const int laneid = kittens::laneid();  
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
        const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
        const int addr = src_ptr + swizzled_lane_byte_offset;
        return addr;
      }();
      load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 1>(P_ij, P_ij);
      mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
      load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
      mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
      mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
      sub_row<0, 3, L_i>(P_ij, P_ij);
      mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 0>(P_ij_bf16, P_ij);
      mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
      exp2<0, 2>(P_ij, P_ij);
      mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 1>(P_ij_bf16, P_ij);
      mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
      load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 3>(P_ij, P_ij);
      mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
      load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 2>(P_ij_bf16, P_ij);
      copy<0, 3>(P_ij_bf16, P_ij);
      mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(8)");
      // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
      mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      // Load K_j_col from shared memory to registers
      // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      sub_row<0, 0, delta_i>(dP_ij, dP_ij);
      sub_row<0, 1, delta_i>(dP_ij, dP_ij);
      mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      mul<0, 0>(dP_ij, dP_ij, P_ij);
      mul<0, 1>(dP_ij, dP_ij, P_ij);
      copy<0, 0>(dP_ij_bf16, dP_ij);
      copy<0, 1>(dP_ij_bf16, dP_ij);
      sub_row<0, 2, delta_i>(dP_ij, dP_ij);
      mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      // Store dP_ij_bf16_accum_row to shared memory
      // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      sub_row<0, 3, delta_i>(dP_ij, dP_ij);
      mul<0, 2>(dP_ij, dP_ij, P_ij);
      mul<0, 3>(dP_ij, dP_ij, P_ij);
      copy<0, 2>(dP_ij_bf16, dP_ij);
      copy<0, 3>(dP_ij_bf16, dP_ij);
      mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

      // dot slice 3
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
      
      store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      asm volatile("s_waitcnt lgkmcnt(12)");
      // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
      mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(8)");
      __builtin_amdgcn_s_barrier();
      mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      // Load dP_ij_bf16_col_T from shared memory to registers
      // load(dP_ij_bf16_col_T, attn_i_smem);
      load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      mul<L_i, L_i>(L_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
      mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
      mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(6)");
      __builtin_amdgcn_s_barrier();
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);

      mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      // Load Q_i from shared memory to registers
      // load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      Q_i_addr = get_address(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load<0, 0>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      load<0, 1>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(4)");
      __builtin_amdgcn_s_barrier();
      mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<0, 2>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      load<0, 3>(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_addr);
      mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      // Load K_j from shared memory to registers
      // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load<0, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<0, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<0, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<0, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mul<0, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<0, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<1, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<1, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mul<0, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<0, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(10)");
      mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<1, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<1, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(2)");
    }

    // dot slice 3
    {
      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp2(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      // mma_ABt(P_ij, Q_i, K_j);
      mma_ABt<0, 0, 0>(P_ij, Q_i, K_j);
      load<2, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<2, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 0, 1>(P_ij, Q_i, K_j, P_ij);
      mma_ABt<0, 0, 2>(P_ij, Q_i, K_j, P_ij);
      load<2, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<2, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 0, 3>(P_ij, Q_i, K_j, P_ij);
      mul<1, 0, 0>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<1, 0, 1>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_ABt<0, 1, 0>(P_ij, Q_i, K_j);
      load<3, 0>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<3, 1>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 1, 1>(P_ij, Q_i, K_j, P_ij);
      mul<1, 0, 2>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mul<1, 0, 3>(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
      mma_ABt<0, 1, 2>(P_ij, Q_i, K_j, P_ij);
      load<3, 2>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      load<3, 3>(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}), K_j_addr);
      mma_ABt<0, 1, 3>(P_ij, Q_i, K_j, P_ij);
      mul<0, 0>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 2, 0>(P_ij, Q_i, K_j);
      // Load dO_i from shared memory to registers
      // load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      dO_i_addr = get_address(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      load<0, 0>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      load<0, 1>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      mma_ABt<0, 2, 1>(P_ij, Q_i, K_j, P_ij);
      sub_row<0, 0, L_i>(P_ij, P_ij);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 2, 2>(P_ij, Q_i, K_j, P_ij);
      load<0, 2>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      load<0, 3>(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_addr);
      mma_ABt<0, 2, 3>(P_ij, Q_i, K_j, P_ij);
      mul<0, 1>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(6)");
      mma_ABt<0, 3, 0>(P_ij, Q_i, K_j);
      // Load dO_i_col from shared memory to registers
      // load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      // Compute dO_i_col_addr
      // uint32_t dO_i_col_addr = get_address(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      dO_i_col_addr = [&] {
        const int laneid = kittens::laneid();
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
        const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
        const uint32_t addr = src_ptr + swizzled_lane_byte_offset;
        return addr;
      }();
      load<0, 0>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      load<0, 1>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      mma_ABt<0, 3, 1>(P_ij, Q_i, K_j, P_ij);
      sub_row<0, 1, L_i>(P_ij, P_ij);
      asm volatile("s_waitcnt lgkmcnt(8)");
      mma_ABt<0, 3, 2>(P_ij, Q_i, K_j, P_ij);
      load<0, 2>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      load<0, 3>(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}), dO_i_col_addr);
      mma_ABt<0, 3, 3>(P_ij, Q_i, K_j, P_ij);
      // Dot slice 3
      kittens::macros::v_mov_b32<neg_inf_v>(0xff800000); if constexpr (causal) {
        // If the query position is less than the key position, set P_ij to -inf
        if (q_pos < k_pos) {
          mov<neg_inf_v>(P_ij);
        // If the query position is equal to the key position, we need to apply a causal mask
        } else if (q_pos == k_pos) {
            // Apply the causal mask to [0, 3]
            make_causal<0, 3, neg_inf_v>(P_ij, P_ij);
        }
      }
      mul<0, 2>(P_ij, P_ij, P_SCALE_FACTOR);
      asm volatile("s_waitcnt lgkmcnt(8)");
      // mma_ABt(dP_ij, dO_i, V_j);
      mma_ABt<0, 0, 0>(dP_ij, dO_i, V_j);
      sub_row<0, 2, L_i>(P_ij, P_ij);
      mma_ABt<0, 0, 1>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 0>(P_ij, P_ij);
      mma_ABt<0, 0, 2>(dP_ij, dO_i, V_j, dP_ij);
      // Load Q_i_col from shared memory to registers
      // load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      // Compute Q_i_col_addr
      // uint32_t Q_i_col_addr = get_address(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      Q_i_col_addr = [&] {
        const int laneid = kittens::laneid();  
        const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}).data[0]);
        const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
        const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
        const int lane_byte_offset = (row_offset * 32 + col_offset) * sizeof(bf16);
        const int swizzled_lane_byte_offset = lane_byte_offset ^ ((lane_byte_offset >> 9) << 5);
        const int addr = src_ptr + swizzled_lane_byte_offset;
        return addr;
      }();
      load<0, 0>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 0, 3>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 1>(P_ij, P_ij);
      mma_ABt<0, 1, 0>(dP_ij, dO_i, V_j);
      load<0, 1>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 1, 1>(dP_ij, dO_i, V_j, dP_ij);
      mul<0, 3>(P_ij, P_ij, P_SCALE_FACTOR);
      mma_ABt<0, 1, 2>(dP_ij, dO_i, V_j, dP_ij);
      sub_row<0, 3, L_i>(P_ij, P_ij);
      mma_ABt<0, 1, 3>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 0>(P_ij_bf16, P_ij);
      mma_ABt<0, 2, 0>(dP_ij, dO_i, V_j);
      exp2<0, 2>(P_ij, P_ij);
      mma_ABt<0, 2, 1>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 1>(P_ij_bf16, P_ij);
      mma_ABt<0, 2, 2>(dP_ij, dO_i, V_j, dP_ij);
      load<0, 2>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 2, 3>(dP_ij, dO_i, V_j, dP_ij);
      exp2<0, 3>(P_ij, P_ij);
      mma_ABt<0, 3, 0>(dP_ij, dO_i, V_j);
      load<0, 3>(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}), Q_i_col_addr);
      mma_ABt<0, 3, 1>(dP_ij, dO_i, V_j, dP_ij);
      copy<0, 2>(P_ij_bf16, P_ij);
      copy<0, 3>(P_ij_bf16, P_ij);
      mma_ABt<0, 3, 2>(dP_ij, dO_i, V_j, dP_ij);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_ABt<0, 3, 3>(dP_ij, dO_i, V_j, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(8)");
      // mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
      mma_AtB<0, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      // Load K_j_col from shared memory to registers
      // load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load<0, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<0, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      sub_row<0, 0, delta_i>(dP_ij, dP_ij);
      sub_row<0, 1, delta_i>(dP_ij, dP_ij);
      mma_AtB<1, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      load<1, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<1, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<1, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      mul<0, 0>(dP_ij, dP_ij, P_ij);
      mul<0, 1>(dP_ij, dP_ij, P_ij);
      copy<0, 0>(dP_ij_bf16, dP_ij);
      copy<0, 1>(dP_ij_bf16, dP_ij);
      sub_row<0, 2, delta_i>(dP_ij, dP_ij);
      mma_AtB<2, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      load<2, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      // Store dP_ij_bf16_accum_row to shared memory
      // store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      store<0, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      store<1, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      mma_AtB<2, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      sub_row<0, 3, delta_i>(dP_ij, dP_ij);
      mul<0, 2>(dP_ij, dP_ij, P_ij);
      mul<0, 3>(dP_ij, dP_ij, P_ij);
      copy<0, 2>(dP_ij_bf16, dP_ij);
      copy<0, 3>(dP_ij_bf16, dP_ij);
      mma_AtB<3, 0, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);

      // dot slice 0 - next iteration
      
      store<2, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      store<3, 0>(attn_i_smem_subtile, dP_ij_bf16_accum_row, dP_ij_bf16_accum_row_addr);
      mma_AtB<3, 1, 0>(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      asm volatile("s_waitcnt lgkmcnt(12)");
      // mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
      mma_AtB<0, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<2, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<3, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<3, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<4, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(8)");
      __builtin_amdgcn_s_barrier();
      mma_AtB<1, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      // Load dP_ij_bf16_col_T from shared memory to registers
      // load(dP_ij_bf16_col_T, attn_i_smem);
      load<0, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<1, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<2, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<3, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      mma_AtB<1, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      atomic_pk_add_bf16_with_warpid<2, 0, 0>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
      mma_AtB<2, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<4, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<5, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<4, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<5, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<2, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      atomic_pk_add_bf16_with_warpid<2, 0, 1>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
      mma_AtB<3, 0, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      load<6, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<7, 0>(dP_ij_bf16_col_T, attn_i_smem, dP_ij_bf16_col_T_addr);
      load<5, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<3, 1, 0>(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(6)");
      __builtin_amdgcn_s_barrier();
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      // mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      mma_AtB<0, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      load<6, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<6, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<0, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      load<7, 0>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      load<7, 1>(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}), K_j_col_addr);
      mma_AtB<0, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<0, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      // ds_read_b128 a[112:115]
      // ds_read_b128 a[116:119]
      mma_AtB<0, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(4)");
      __builtin_amdgcn_s_barrier();
      mma_AtB<0, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      // ds_read_b128 a[120:123]
      // ds_read_b128 a[124:127]
      mma_AtB<0, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 0>(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      // ds_read_b128 a[0:3]
      // ds_read_b128 a[4:7]
      mma_AtB<1, 0, 1>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 2>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      // ds_read_b128 a[8:11]
      // ds_read_b128 a[12:15]
      mma_AtB<1, 0, 3>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      mma_AtB<1, 0, 4>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      // ds_read_b128 a[16:19]
      // ds_read_b128 a[20:23]
      mma_AtB<1, 0, 5>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(10)");
      mma_AtB<1, 0, 6>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      // ds_read_b128 a[24:27]
      // ds_read_b128 a[28:31]
      mma_AtB<1, 0, 7>(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
      asm volatile("s_waitcnt lgkmcnt(2)");
    }
  }

  store<1>(g.dVg, dV_j, {batch_idx * GROUP_SIZE + q_head_in_group, 0, kv_head_idx, 0}, {0, j, 0, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  // We first copy dV_j_T from accumulator GPRs to vector GPRs and then perform the store
  accvgpr_read(dV_j_T, dK_j_T);
  mul(dV_j_T, dV_j_T, dP_SCALE_FACTOR);
  store<1>(g.dKg, dV_j, {batch_idx * GROUP_SIZE + q_head_in_group, 0, kv_head_idx, 0}, {0, j, 0, 0});

  // Write out final dQ_i slice
  mul(dQ_i_T, dQ_i_T, dP_SCALE_FACTOR);
  atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 3, 0}, warpid);
}

template __global__ void attend_bwd_combined_ker<ATTN_D>(bf16*, bf16*, bf16*, bf16*, bf16*, bf16*, bf16*, float*, float*);
