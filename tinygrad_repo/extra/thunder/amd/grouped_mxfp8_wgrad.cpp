#include "kittens.cuh"

using namespace kittens;

#ifndef WGRAD_M
constexpr int WGRAD_M = 8192;
#endif
#ifndef WGRAD_N
constexpr int WGRAD_N = 8192;
#endif
#ifndef WGRAD_K
constexpr int WGRAD_K = 8192;
#endif
#ifndef WGRAD_E
constexpr int WGRAD_E = 8;
#endif

constexpr int NUM_WARPS  = 8;
constexpr int WARPS_ROW  = 2;
constexpr int WARPS_COL  = 4;
constexpr int BLOCK_ROW  = 256;
constexpr int BLOCK_COL  = 256;
constexpr int BLOCK_K    = 128;
constexpr int HALF_ROW   = BLOCK_ROW / 2;
constexpr int HALF_COL   = BLOCK_COL / 2;
constexpr int REG_M      = BLOCK_ROW / WARPS_ROW / 2;
constexpr int REG_N      = BLOCK_COL / WARPS_COL / 2;

using G = kittens::group<NUM_WARPS>;

__global__ __launch_bounds__(512, 2) void grouped_mxfp8_wgrad_kernel(bf16 *C_ptr, fp8e4m3 *A_ptr, fp8e4m3 *B_ptr,
    fp8e8m0 *scale_A_ptr, fp8e8m0 *scale_B_ptr,
    const int *__restrict__ expert_off) {
    constexpr int M = WGRAD_M, N = WGRAD_N, K = WGRAD_K, E = WGRAD_E;

    kittens::gl<fp8e4m3, 1, 1, N, M>     A{A_ptr, nullptr, nullptr, nullptr, nullptr};  // g^T
    kittens::gl<fp8e4m3, 1, 1, K, M>     B{B_ptr, nullptr, nullptr, nullptr, nullptr};  // x^T
    kittens::gl<bf16, 1, 1, E * N, K>    C{C_ptr, nullptr, nullptr, nullptr, nullptr};  // grad_w, experts stacked

    constexpr int m_blocks    = M / BLOCK_K;      // 128-wide blocks along the contraction (token) axis
    constexpr int tiles_N     = N / BLOCK_ROW;
    constexpr int tiles_K     = K / BLOCK_COL;
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    kittens::gl<fp8e8m0, m_blocks * tiles_N, 1, 16, 64> scale_A_gl{scale_A_ptr, nullptr, nullptr, nullptr, nullptr};
    kittens::gl<fp8e8m0, m_blocks * tiles_K, 1, 16, 64> scale_B_gl{scale_B_ptr, nullptr, nullptr, nullptr, nullptr};

    using ST_A     = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B     = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    using RT_A     = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B     = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C     = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2];
    __shared__ ST_B Bs[2];
    __shared__ ST_Scale scale_A_smem, scale_B_smem;

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    const int wg        = blockIdx.x;
    const int e         = wg / (tiles_N * tiles_K);
    const int rem       = wg % (tiles_N * tiles_K);
    const int block_row = rem / tiles_K;   // over grad_w rows (N)
    const int block_col = rem % tiles_K;   // over grad_w cols (K)

    const int o0  = __builtin_amdgcn_readfirstlane(expert_off[e]);
    const int kk0 = o0 / BLOCK_K;
    const int nk  = (__builtin_amdgcn_readfirstlane(expert_off[e + 1]) - o0) / BLOCK_K;

    const int warp_m = warpid() / WARPS_COL;
    const int warp_n = warpid() % WARPS_COL;

    using T = fp8e4m3;
    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G::prefill_swizzled_offsets(As[0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    i32x4 a_srd = make_srsrc(a_base, (uint32_t)((uint64_t)N * a_row_stride), a_row_stride);
    i32x4 b_srd = make_srsrc(b_base, (uint32_t)((uint64_t)K * b_row_stride), b_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t a_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&As[1].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[0].data[0]) + wid * elem_per_warp * sizeof(T)));
    uint32_t b_lds_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&Bs[1].data[0]) + wid * elem_per_warp * sizeof(T)));

    const int a_row_h0 = warp_m * REG_M;
    const int a_row_h1 = HALF_ROW + warp_m * REG_M;
    const int b_row_h0 = warp_n * REG_N;
    const int b_row_h1 = HALF_COL + warp_n * REG_N;

    #pragma unroll 1
    for (int t = 0; t < nk; t++) {
        const int kk = kk0 + t;
        G::load(As[0], A, {0, 0, block_row * 2,     kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_0));
        G::load(As[1], A, {0, 0, block_row * 2 + 1, kk}, sw_A, a_srd, a_base, __builtin_amdgcn_readfirstlane(a_lds_1));
        G::load(Bs[0], B, {0, 0, block_col * 2,     kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_0));
        G::load(Bs[1], B, {0, 0, block_col * 2 + 1, kk}, sw_B, b_srd, b_base, __builtin_amdgcn_readfirstlane(b_lds_1));
        G::load(scale_A_smem, scale_A_gl, {kk * tiles_N + block_row, 0, 0, 0});
        G::load(scale_B_smem, scale_B_gl, {kk * tiles_K + block_col, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem.data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem.data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem.data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem.data, b_row_h1);

        auto bs0 = subtile_inplace<REG_N, BLOCK_K>(Bs[0], {warp_n, 0}); load(b0, bs0);
        auto bs1 = subtile_inplace<REG_N, BLOCK_K>(Bs[1], {warp_n, 0}); load(b1, bs1);
        auto as0 = subtile_inplace<REG_M, BLOCK_K>(As[0], {warp_m, 0}); load(a, as0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);
        auto as1 = subtile_inplace<REG_M, BLOCK_K>(As[1], {warp_m, 0}); load(a, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);
        __builtin_amdgcn_s_barrier();
    }

    const int crow_base = e * (N / REG_M);   // grad_w rows are experts stacked; store coord is in REG_M units
    store(C, cA, {0, 0, crow_base + block_row * WARPS_ROW * 2 + warp_m,              block_col * WARPS_COL * 2 + warp_n});
    store(C, cB, {0, 0, crow_base + block_row * WARPS_ROW * 2 + warp_m,              block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(C, cC, {0, 0, crow_base + block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m,  block_col * WARPS_COL * 2 + warp_n});
    store(C, cD, {0, 0, crow_base + block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m,  block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
}
