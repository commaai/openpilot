/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

__device__ static inline void mfma161632(      float2 (&D)[2],
                                         const half_2 (&A)[4],
                                         const half_2 (&B)[4],
                                         const float2 (&C)[2]) {
    
    typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 fp16x8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
    *(floatx4_t*)D = __builtin_amdgcn_mfma_f32_16x16x32_f16(
        (*(fp16x8_t*)A),
        (*(fp16x8_t*)B),
        *(floatx4_t*)C,
        0, 0, 0
    );
}

/**
 * @brief gfx1250 WMMA bf16 16x16x32 helper.
 *
 * Wraps `__builtin_amdgcn_wmma_f32_16x16x32_bf16` -- the bf16 entry point
 * into the gfx1250 matrix unit. Instruction shape: A is 16x32 bf16, B is
 * 32x16 bf16, C/D is 16x16 fp32.
 *
 * Operand reuse cache (the `_reuse` template params). The matrix unit has a
 * small staging cache holding the most recently consumed A and B operand
 * bundles. If the *next* WMMA call uses the same A and/or B as the current
 * one, setting the matching `_reuse=true` on that next call tells the
 * hardware to read the operand from the cache instead of from the VGPR file
 * -- saving VGPR read bandwidth and easing the VGPR-bank pressure that the
 * WMMA operand-read path would otherwise create.
 *
 * Rules of use:
 *   - Only set `_reuse=true` on the **second-or-later** call of a contiguous
 *     burst that shares the operand. The first call has nothing to reuse and
 *     must be `_reuse=false`.
 *   - The reusing call must have the **identical opcode** and the **same
 *     operand VGPR block** as the previous call. Mixing dtypes or changing
 *     the source register block is undefined.
 *   - The cache is small -- one slot per operand. The most recent A (or B)
 *     evicts the previous one.
 *
 * Callers normally do not set these hints by hand -- the higher-level
 * `mma_ABt` derives them from the iteration position in its m-outer/n-inner
 * zigzag traversal, so passing the defaults here just means "first call of a
 * burst" and `mma_ABt` overrides them for subsequent calls.
 *
 * `D`, `A`, `B`, `C` are the `rt_base::data` arrays:
 *   - C accumulator: `float2[4]` (8 floats per lane).
 *   - A operand:     `bf16_2[8]` (16 bf16 per lane).
 *   - B operand:     `bf16_2[8]` (16 bf16 per lane, row-major for ABt).
 */
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161632(      float2 (&D)[4],
                                         const bf16_2 (&A)[8],
                                         const bf16_2 (&B)[8],
                                         const float2 (&C)[4]) {
    typedef __attribute__((__vector_size__(16 * sizeof(__bf16)))) __bf16 bf16x16_t;
    typedef __attribute__((__vector_size__(8 * sizeof(float)))) float floatx8_t;
    *(floatx8_t*)D = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
        /*a_neg=*/ false, *(bf16x16_t*)A,
        /*b_neg=*/ false, *(bf16x16_t*)B,
        /*c_mod=*/ 0,     *(floatx8_t*)C,
        A_reuse, B_reuse);
}

/// @brief gfx1250 WMMA fp16 16x16x32 helper (parallel signature to bf16).
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161632(      float2 (&D)[4],
                                         const half_2 (&A)[8],
                                         const half_2 (&B)[8],
                                         const float2 (&C)[4]) {
    typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 fp16x16_t;
    typedef __attribute__((__vector_size__(8 * sizeof(float)))) float floatx8_t;
    *(floatx8_t*)D = __builtin_amdgcn_wmma_f32_16x16x32_f16(
        /*a_neg=*/ false, *(fp16x16_t*)A,
        /*b_neg=*/ false, *(fp16x16_t*)B,
        /*c_mod=*/ 0,     *(floatx8_t*)C,
        A_reuse, B_reuse);
}

__device__ static inline void mfma161632(      float2 (&D)[2],
                                         const bf16_2 (&A)[4],
                                         const bf16_2 (&B)[4],
                                         const float2 (&C)[2]) {

    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
    *(floatx4_t*)D = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
        (*(bf16x8_t*)A),
        (*(bf16x8_t*)B),
        *(floatx4_t*)C,
        0, 0, 0
    );
}
__device__ static inline void mfma323216(      float2 (&D)[8],
                                         const bf16_2 (&A)[4],
                                         const bf16_2 (&B)[4],
                                         const float2 (&C)[8]) {
    // Cast to the correct vector types that the intrinsic expects
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(bf16x8_t*)(A),
        *(bf16x8_t*)(B),
        *(floatx16_t*)C,
        0, 0, 0
    );
}

__device__ static inline void mfma323216(      float2 (&D)[8],
                                         const half_2 (&A)[4],
                                         const half_2 (&B)[4],
                                         const float2 (&C)[8]) {
    // Cast to the correct vector types that the intrinsic expects
    typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 fp16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    
    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_f16(
        *(fp16x8_t*)(A),
        *(fp16x8_t*)(B),
        *(floatx16_t*)C,
        0, 0, 0
    );
}

__device__ static inline void mfma323232(      float2 (&D)[8],
                                         const bf16_2 (&A)[8],
                                         const bf16_2 (&B)[8],
                                         const float2 (&C)[8]) {
    // Cast to the correct vector types that the intrinsic expects
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    
    *(floatx16_t*)C = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(bf16x8_t*)A,
        *(bf16x8_t*)B,
        *(floatx16_t*)C,
        0, 0, 0
    );

    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(bf16x8_t*)(A + 4),
        *(bf16x8_t*)(B + 4),
        *(floatx16_t*)C,
        0, 0, 0
    );
}

__device__ static inline void mfma323264(      float2 (&D)[8],
                                         const fp8e4m3_4 (&A)[8],
                                         const fp8e4m3_4 (&B)[8],
                                         const float2 (&C)[8]) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    *(floatx16_t*)D = {__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        *(intx8_t*)A,
        *(intx8_t*)B,
        *(floatx16_t*)C,
        0, 0, 0, 0, 0, 0
    )};
}

__device__ static inline void mfma1616128(      float2 (&D)[2],
                                         const fp8e4m3_4 (&A)[8],
                                         const fp8e4m3_4 (&B)[8],
                                         const float2 (&C)[2]) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    *(floatx4_t*)D = {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        *(intx8_t*)A,
        *(intx8_t*)B,
        *(floatx4_t*)C,
        0, 0, 0, 0, 0, 0
    )};
}

template<int opsel_a, int opsel_b, int cbsz = 0, int blgp = 0>
__device__ static inline void mfma1616128_scaled(      float2 (&D)[2],
                                         const fp8e4m3_4 (&A)[8],
                                         const fp8e4m3_4 (&B)[8],
                                         const float2 (&C)[2],
                                         const fp8e8m0_4 *scale_a,
                                         const fp8e8m0_4 *scale_b) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    *(floatx4_t*)D = {__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        *(intx8_t*)A,
        *(intx8_t*)B,
        *(floatx4_t*)C,
        cbsz,         // cbsz: 0=fp8(e4m3) A, 1=bf8(e5m2) A
        blgp,         // blgp: 0=fp8(e4m3) B, 1=bf8(e5m2) B
        opsel_a,      // opsel_a
        *scale_a,     // scale_a
        opsel_b,      // opsel_b
        *scale_b      // scale_b
    )};
}

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                        const rt_base<bf16, ducks::rt_layout::row, A_shape> &a,
                                        const rt_base<bf16, ducks::rt_layout::col, B_shape> &b, // in col-major mode
                                        const rt_base<float, ducks::rt_layout::col, C_shape> &c) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");
    
    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> && 
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 32 && B_cols == 16 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        mfma161632(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> && 
                  A_rows == 32 && A_cols == 16 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
        mfma323216(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}

/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the mma function for matrices in row layout.
 *
 * The `A_reuse` / `B_reuse` template parameters propagate down to the gfx1250
 * WMMA builtin's operand-reuse hints (see `wmma161632`). They default to
 * `false` and are silently ignored on CDNA (MFMA has no operand reuse cache).
 *
 * @tparam A_reuse,B_reuse  Operand-reuse hints (gfx1250 only).
 * @tparam D_shape,A_shape,B_shape,C_shape  Register-tile shape tags.
 * @tparam MM_Operand_T     Operand element type for A and B. Defaults to bf16.
 *
 * @param[out] d The output rt_base<float, col_layout> accumulator.
 * @param[in]  a The first input rt_base<MM_Operand_T, row_layout> matrix.
 * @param[in]  b The second input rt_base<MM_Operand_T, row_layout> matrix in row-major mode.
 * @param[in]  c The input rt_base<float, col_layout> accumulator matrix.
 */
template<bool A_reuse = false, bool B_reuse = false,
         ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape,
         ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape,
         typename MM_Operand_T = bf16>
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
    const rt_base<MM_Operand_T, ducks::rt_layout::row, A_shape> &a,
    const rt_base<MM_Operand_T, ducks::rt_layout::row, B_shape> &b, // in row-major mode
    const rt_base<float, ducks::rt_layout::col, C_shape> &c) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");

    // gfx1250 WMMA path: `rt_16x16` D + `rt_16x32` A/B shapes. The wave-32 lane
    // storage gives `elements_per_thread = 256/32 = 8` for the accumulator and
    // `512/32 = 16` for the operands. The bf16/f16 16x16x32 case routes to the
    // WMMA builtin.
    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        wmma161632<A_reuse, B_reuse>(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination for gfx1250 mma_ABt_base");
    }
}

/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * for block-scaled matrices in row layout.
 *
 * @param[out] d The output rt_base<float, col_layout> accumulator.
 * @param[in] a The first input rt_base<Operand_T, row_layout> matrix.
 * @param[in] b The second input rt_base<Operand_T, row_layout> matrix.
 * @param[in] c The input rt_base<float, col_layout> accumulator matrix.
 */
template<int opsel_a, int opsel_b, int cbsz = 0, int blgp = 0, ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape, typename MM_Operand_T>
__device__ static inline void mma_ABt_base_scaled(rt_base<float, ducks::rt_layout::col, D_shape> &d,
    const rt_base<MM_Operand_T, ducks::rt_layout::row, A_shape> &a,
    const rt_base<MM_Operand_T, ducks::rt_layout::row, B_shape> &b,
    const rt_base<float, ducks::rt_layout::col, C_shape> &c,
    const fp8e8m0_4 *scale_a,
    const fp8e8m0_4 *scale_b) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");

    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                A_rows == 16 && A_cols == 128 &&
                B_rows == 16 && B_cols == 128 &&
                std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        mfma1616128_scaled<opsel_a, opsel_b, cbsz, blgp>(d.data, a.data, b.data, c.data, scale_a, scale_b);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}

/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                           const rt_base<bf16, ducks::rt_layout::col, A_shape> &a,
                                           const rt_base<bf16, ducks::rt_layout::col, B_shape> &b, // in col-major mode
                                           const rt_base<float, ducks::rt_layout::col, C_shape> &c) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");
    
    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> && 
                  A_rows == 32 && A_cols == 16 &&
                  B_rows == 32 && B_cols == 16 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        mfma161632(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> && 
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
        mfma323216(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> &&
                  A_rows == 32 && A_cols == 32 &&
                  B_rows == 32 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
        mfma323232(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
template<ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape, ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                            const rt_base<bf16, ducks::rt_layout::col, A_shape> &a,
                                            const rt_base<bf16, ducks::rt_layout::row, B_shape> &b, // in col-major mode
                                            const rt_base<float, ducks::rt_layout::col, C_shape> &c) {

    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");

    constexpr int A_rows = A_shape::rows;
    constexpr int A_cols = A_shape::cols;
    constexpr int B_rows = B_shape::rows;
    constexpr int B_cols = B_shape::cols;

    constexpr int A_stride = A_shape::stride;
    constexpr int B_stride = B_shape::stride;
    static_assert(A_stride == B_stride, "A and B must have the same stride");
    
    if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> && 
                  A_rows == 32 && A_cols == 16 &&
                  B_rows == 16 && B_cols == 32 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_16x16>) {
        mfma161632(d.data, a.data, b.data, c.data);
    } else if constexpr (std::is_same_v<D_shape, typename ducks::rt_shape::rt_32x32> && 
                  A_rows == 16 && A_cols == 32 &&
                  B_rows == 32 && B_cols == 16 &&
                  std::is_same_v<C_shape, typename ducks::rt_shape::rt_32x32>) {
        mfma323216(d.data, a.data, b.data, c.data);
    } else {
        static_assert(false, "Unsupported shape combination");
    }

}

/**
 * @brief Matrix multiply-accumulate operation.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `mma_ABt_base` function.
 *
 * @tparam D Accumulator register-tile type (col layout).
 * @tparam A First operand register-tile type (row layout).
 * @tparam B Second operand register-tile type (row layout), row-major for ABt.
 * @tparam C Input accumulator register-tile type (col layout).
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {

    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );

    // gfx1250: m-outer, n-inner zigzag with reuse hints derived from
    // (M, S) -- both compile-time template parameters of the nested lambdas.
    [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
        ([&]<std::size_t M>() {
            [&]<std::size_t... Ss>(std::index_sequence<Ss...>) {
                ([&]<std::size_t S>() {
                    constexpr std::size_t n        = (M & 1) ? (D::height - 1 - S) : S;
                    constexpr bool        B_reuse  = (S > 0);              // within-column
                    constexpr bool        A_reuse  = (S == 0) && (M > 0);  // column-switch (zigzag)
                    // k = 0: pull accumulator from `c`.
                    mma_ABt_base<A_reuse, B_reuse>(
                        d.tiles[n][M], a.tiles[n][0], b.tiles[M][0], c.tiles[n][M]);
                    // k >= 1: chain into `d`. A and B both advance, so no reuse.
                    if constexpr (A::width > 1) {
                        [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
                            ([&]<std::size_t K>() {
                                constexpr std::size_t k = K + 1;
                                mma_ABt_base<false, false>(
                                    d.tiles[n][M], a.tiles[n][k], b.tiles[M][k], d.tiles[n][M]);
                            }.template operator()<Ks>(), ...);
                        }(std::make_index_sequence<A::width - 1>{});
                    }
                }.template operator()<Ss>(), ...);
            }(std::make_index_sequence<D::height>{});
        }.template operator()<Ms>(), ...);
    }(std::make_index_sequence<D::width>{});
}

/**
 * @brief Block scaled dot product operation for row layout.
 *
 * This function performs the dot product operation
 * for block-scaled matrices in row layout.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 * @param[in] scale_a Pointer to the packed E8M0 scale for the A matrix.
 * @param[in] scale_b Pointer to the packed E8M0 scale for the B matrix.
 */
template<int cbsz = 0, int blgp = 0, ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_ABt_scaled(D &d,
                                const A &a,
                                const B &b,
                                const C &c,
                                const fp8e8m0_4 *scale_a,
                                const fp8e8m0_4 *scale_b) {

    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>)
    );

    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    mma_ABt_base_scaled<N, M, cbsz, blgp>(
                        d.tiles[N][M],
                        a.tiles[N][0],
                        b.tiles[M][0],
                        c.tiles[N][M],
                        scale_a,
                        scale_b
                    );
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<D::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<D::height>{});
}

/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
}