/**
 * @file
 * @brief Warpgroup matrix-multiply accumulate operations. These ops are necessary to achieve full utilization on H100 GPUs.
 */



//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  ------------------------------------------------------ FENCES ------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------


/**
 * @brief Synchronize the warp group and ensure that all writes to shared memory are visible to all threads in the warp group.
 *
 * This function acts as a fence for shared memory operations, ensuring that all previous writes are visible before proceeding.
 * This function should be called before running wgmma::mma or wgmma::dot instructions.
 *
 * @tparam height The height of the matrix `dst`.
 * @tparam width The width of the matrix `dst`.
 * @param dst[in,out] The destination register-tile matrix to be synchronized.
 */
template<ducks::rt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
    #pragma unroll
    for(int i = 0; i < D::height; i++) {
        #pragma unroll
        for(int j = 0; j < D::width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                if constexpr(std::is_same_v<typename D::T, float>) {
                    asm volatile("" : "+f"(dst.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.tiles[i][j].data[k].y) :: "memory");
                } else {
                    asm volatile("" : "+r"(*(uint32_t*)&dst.tiles[i][j].data[k]) :: "memory");
                }
            }
        }
    }
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}
template<ducks::crt::row_layout D>
__device__ static inline void mma_fence(D &dst) {
    KITTENS_CHECK_WARPGROUP
    #pragma unroll
    for(int i = 0; i < D::height; i++) {
        #pragma unroll
        for(int j = 0; j < D::width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.real.packed_per_tile; k++) {
                if constexpr(std::is_same_v<typename D::T, float>) {
                    asm volatile("" : "+f"(dst.real.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.real.tiles[i][j].data[k].y) :: "memory");
                    asm volatile("" : "+f"(dst.imag.tiles[i][j].data[k].x) :: "memory");
                    asm volatile("" : "+f"(dst.imag.tiles[i][j].data[k].y) :: "memory");
                } else {
                    asm volatile("" : "+r"(*(uint32_t*)&dst.real.tiles[i][j].data[k]) :: "memory");
                    asm volatile("" : "+r"(*(uint32_t*)&dst.imag.tiles[i][j].data[k]) :: "memory");
                }
            }
        }
    }
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}
template<typename T=kittens::ducks::default_type> // prevents static assert being instantiated unless called.
__device__ static inline void mma_fence() {
    KITTENS_CHECK_WARPGROUP
    asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
}

/**
 * @brief Commit the current set of warp group matrix multiply accumulate calls.
 */
template<typename T=kittens::ducks::default_type> // prevents static assert being instantiated unless called.
__device__ static inline void mma_commit_group() {
    KITTENS_CHECK_WARPGROUP
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @brief Wait for the warp group to reach a synchronization point.
 *
 * This function stalls the current warpgroup until enough WGMMA committed groups have been completed.
 *
 * @tparam N The number of remaining active WGMMA committed groups allowed. This will stall until the number of active groups is less than or equal to N. Defaults to 0.
 */
template<int N=0>
__device__ static inline void mma_async_wait() {
    KITTENS_CHECK_WARPGROUP
    asm volatile ("wgmma.wait_group.sync.aligned %0;" : : "n"(N) : "memory");
}


//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  ------------------------------------------------------ NORMAL ------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------

/*
 ### OPTIONS:

 REG+SMEM -> REG
 - mma_AB   (accum) [DONE]
 - mm_AB    (reset) [DONE]
 - mma_ABt  (accum) [DONE]
 - mm_ABt   (reset) [DONE]
 
 SMEM+SMEM -> REG
 - mma_AB   (accum) [DONE]
 - mm_AB    (reset) [DONE]
 - mma_ABt  (accum) [DONE]
 - mm_ABt   (reset) [DONE]
 - mma_AtB  (accum) [DONE]
 - mm_AtB   (reset) [DONE]
 - mma_AtBt (accum) [DONE]
 - mm_AtBt  (reset) [DONE]
 
Note: mma is an alias for mma_AB and dot is an alias for mma_ABt
*/

// [(register, shared) -> register] edition
/**
 * @brief Perform matrix multiply-accumulate operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function multiplies a register tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::width;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    #endif
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b); // apologies for this hack -- it either calls ST constructor or copy constructor.

    if constexpr (fence) { mma_fence(d); }

    // Do it
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d, m);
        base::rt_st(
            d_ref,
            a.tiles[m][0],
            b_desc.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.tiles[m][k],
                b_desc.chunk_descriptor(k),
                1
            );
        }
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::height;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::width;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    #endif
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

// [(register, shared) -> register] edition
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function computes an outer product of a register tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::width;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d, m);
        base::rt_st(
            d_ref,
            a.tiles[m][0],
            b_desc.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.tiles[m][k],
                b_desc.chunk_descriptor(k),
                1
            );
        }
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::rt::row_layout D, ducks::rt::row_layout A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function computes an outer product of a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_A The layout of the matrix `a`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::height;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::width;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix multiply using warp group matrix multiply-accumulate (WGMMA) primitives, with A transposed.
 *
 * This function computes an outer product of a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_A The layout of the matrix `a`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::width;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::height;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    #endif
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 1, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc(b);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d,
                               const A &a,
                               const B &b) {
    mma_AtB<D, A, B, 1, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix multiply using warp group matrix multiply-accumulate (WGMMA) primitives, with A and B transposed.
 *
 * This function computes an outer product of a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam D The destination register tile type.
 * @tparam A The source shared tile type.
 * @tparam B The source shared tile type.
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 */
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::width;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::height;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    #endif
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 1, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc(b);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    base::st_st(
        d,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d,
            a_desc.chunk_descriptor(k),
            b_desc.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::rt::row_layout D, ducks::st_descriptor::input A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d,
                                const A &a,
                                const B &b) {
    mma_AtBt<D, A, B, 1, 0>(d, a, b);
}



//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  -------------------------------------------------- COMPLEX INPUTS --------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------
//  --------------------------------------------------------------------------------------------------------------------


/*
 ### OPTIONS:

 REG+SMEM -> REG
 - mma_AB   (accum) [TODO]
 - mm_AB    (reset) [TODO]
 - mma_ABt  (accum) [TODO]
 - mm_ABt   (reset) [TODO]
 
 SMEM+SMEM -> REG
 - mma_AB   (accum) [TODO]
 - mm_AB    (reset) [TODO]
 - mma_ABt  (accum) [TODO]
 - mm_ABt   (reset) [TODO]
 - mma_AtB  (accum) [TODO]
 - mm_AtB   (reset) [TODO]
 - mma_AtBt (accum) [TODO]
 - mm_AtBt  (reset) [TODO]
 
Note: mma is an alias for mma_AB and dot is an alias for mma_ABt
*/

// [(register, shared) -> register] edition
/**
 * @brief Perform matrix multiply-accumulate operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function multiplies a register tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The width of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::width;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    #endif
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    #pragma unroll // Do real part
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_COL_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.real, m);
        base::rt_st(
            d_ref,
            a.real.tiles[m][0],
            b_desc_real.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.real.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1
            );
        }
        #pragma unroll
        for(int k = 0; k < K; k++) {
            base::rt_st<-1>( // INVERT THE SIGN OF THE IMAGINARY PART
                d_ref,
                a.imag.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1
            );
        }
    }
    #pragma unroll // Do imaginary part
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_AB>, TILE_COL_DIM<T_AB>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.imag, m);
        base::rt_st(
            d_ref,
            a.real.tiles[m][0],
            b_desc_imag.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.real.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1
            );
        }
        #pragma unroll
        for(int k = 0; k < K; k++) {
            base::rt_st(
                d_ref,
                a.imag.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1
            );
        }
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::height;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::width;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    #endif
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_imag(a.imag);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    base::st_st(
        d.real,
        a_desc_real.chunk_descriptor(0),
        b_desc_real.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>( // INVERT THE SIGN OF THE IMAGINARY PART
            d.real,
            a_desc_imag.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    base::st_st(
        d.imag,
        a_desc_real.chunk_descriptor(0),
        b_desc_imag.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_imag.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AB(D &d,
                              const A &a,
                              const B &b) {
    mma_AB<D, A, B, 1, 0>(d, a, b);
}

// [(register, shared) -> register] edition
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function computes an outer product of a register tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam N_DIV_4 The height of the matrix `a` divided by 4.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source register tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M_DIV_4 = A::height;
    static_assert(D::height == M_DIV_4); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::width;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 0, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_D>, TILE_ROW_DIM<T_D>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.real, m);
        base::rt_st(
            d_ref,
            a.real.tiles[m][0],
            b_desc_real.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.real.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1
            );
        }
        #pragma unroll
        for(int k = 0; k < K; k++) {
            base::rt_st<-1>( // INVERT THE SIGN OF THE IMAGINARY PART
                d_ref,
                a.imag.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1
            );
        }
    }
    #pragma unroll
    for(int m = 0; m < M_DIV_4; m++) {
        rt<T_D, TILE_ROW_DIM<T_AB>, TILE_ROW_DIM<T_AB>*N, ducks::rt_layout::row> &d_ref = group<1>::subtile_inplace<TILE_ROW_DIM<T_AB>>(d.imag, m);
        base::rt_st(
            d_ref,
            a.real.tiles[m][0],
            b_desc_imag.chunk_descriptor(0),
            accumulate
        );
        #pragma unroll
        for(int k = 1; k < K; k++) {
            base::rt_st(
                d_ref,
                a.real.tiles[m][k],
                b_desc_imag.chunk_descriptor(k),
                1
            );
        }
        #pragma unroll
        for(int k = 0; k < K; k++) {
            base::rt_st(
                d_ref,
                a.imag.tiles[m][k],
                b_desc_real.chunk_descriptor(k),
                1
            );
        }
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::crt::row_layout D, ducks::crt::row_layout A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix outer product operation using warp group matrix multiply-accumulate (WGMMA) primitives.
 *
 * This function computes an outer product of a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_A The layout of the matrix `a`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::height;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::width;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 0, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 0> a_desc_imag(a.imag);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    base::st_st(
        d.real,
        a_desc_real.chunk_descriptor(0),
        b_desc_real.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>( // INVERT THE SIGN OF THE IMAGINARY PART
            d.real,
            a_desc_imag.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    base::st_st(
        d.imag,
        a_desc_real.chunk_descriptor(0),
        b_desc_imag.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_imag.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_ABt(D &d,
                               const A &a,
                               const B &b) {
    mma_ABt<D, A, B, 1, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix multiply using warp group matrix multiply-accumulate (WGMMA) primitives, with A transposed.
 *
 * This function computes an outer product of a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 * @tparam K The common dimension of matrices `a` and `b`.
 * @tparam M The height of the matrices `b` and `d`.
 * @tparam L_A The layout of the matrix `a`.
 * @tparam L_B The layout of the matrix `b`.
 * @param d[out] The destination register tile where the result is accumulated or written.
 * @param a[in] The source shared tile to be multiplied.
 * @param b[in] The source shared tile to be multiplied.
 */
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::width;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::width;
    constexpr int K = A::height;
    static_assert(B::height == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    #endif
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_COL_DIM<T_AB>*N, 1, 1>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_imag(a.imag);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 1> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    base::st_st(
        d.real,
        a_desc_real.chunk_descriptor(0),
        b_desc_real.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>( // INVERT THE SIGN OF THE IMAGINARY PART
            d.real,
            a_desc_imag.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    base::st_st(
        d.imag,
        a_desc_real.chunk_descriptor(0),
        b_desc_imag.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_imag.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AtB(D &d,
                               const A &a,
                               const B &b) {
    mma_AtB<D, A, B, 1, 0>(d, a, b);
}

// [(shared, shared) -> register] edition
/**
 * @brief Perform matrix multiply using warp group matrix multiply-accumulate (WGMMA) primitives, with A and B transposed.
 *
 * This function computes an outer product of a shared tile `a` with a shared tile `b` and writes the result into a register tile `d`.
 *
 * @tparam D The destination register tile type.
 * @tparam A The source shared tile type.
 * @tparam B The source shared tile type.
 * @tparam accumulate Whether to accumulate the result into `d` or overwrite `d`.
 */
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B, int fence=1, int accumulate=1>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b) {
    // Checks
    KITTENS_CHECK_WARPGROUP
    constexpr int M = A::width;
    static_assert(M == 4);
    static_assert(D::height == 1); // output register is correctly sized
    constexpr int N = B::height;
    constexpr int K = A::height;
    static_assert(B::width == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T;
    using T_D  = D::T;
    #ifdef KITTENS_HOPPER
    static_assert(!std::is_same_v<T_AB, fp8e4m3> && !std::is_same_v<T_AB, fp8e5m2>, "Currently unsupported type");
    static_assert(!std::is_same_v<T_D, fp8e4m3> && !std::is_same_v<T_D, fp8e5m2>, "Currently unsupported type");
    #endif
    using base = kittens::detail::wgmma::base<T_D, T_AB, TILE_ROW_DIM<T_AB>*N, 1, 0>;
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_real(a.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, 1> a_desc_imag(a.imag);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_real(b.real);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, 0> b_desc_imag(b.imag);

    if constexpr (fence) { mma_fence(d); }

    // Do it
    base::st_st(
        d.real,
        a_desc_real.chunk_descriptor(0),
        b_desc_real.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.real,
            a_desc_real.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st<-1>( // INVERT THE SIGN OF THE IMAGINARY PART
            d.real,
            a_desc_imag.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    base::st_st(
        d.imag,
        a_desc_real.chunk_descriptor(0),
        b_desc_imag.chunk_descriptor(0),
        accumulate
    );
    #pragma unroll
    for(int k = 1; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_real.chunk_descriptor(k),
            b_desc_imag.chunk_descriptor(k),
            1
        );
    }
    #pragma unroll
    for(int k = 0; k < K; k++) {
        base::st_st(
            d.imag,
            a_desc_imag.chunk_descriptor(k),
            b_desc_real.chunk_descriptor(k),
            1
        );
    }
    mma_commit_group(); // commit the group of these WGMMA calls.
}
template<ducks::crt::row_layout D, ducks::st_descriptor::complex_input A, ducks::st_descriptor::complex_input B>
__device__ static inline void mm_AtBt(D &d,
                                const A &a,
                                const B &b) {
    mma_AtBt<D, A, B, 1, 0>(d, a, b);
}

// Some extra wrappers for prettiness

template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mma(D &d,
                                  const A &a,
                                  const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mma_AtBt(d, a, b);
        } else {
            mma_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mma_ABt(d, a, b);
        } else {
            mma_AB(d, a, b);
        }
    }
}
template<int trans_A, int trans_B, typename D, typename A, typename B>
__device__ static inline void mm(D &d,
                                  const A &a,
                                  const B &b) {
    if constexpr(trans_A == transpose::T) {
        if constexpr(trans_B == transpose::T) {
            mm_AtBt(d, a, b);
        } else {
            mm_AtB(d, a, b);
        }
    } else {
        if constexpr(trans_B == transpose::T) {
            mm_ABt(d, a, b);
        } else {
            mm_AB(d, a, b);
        }
    }
}