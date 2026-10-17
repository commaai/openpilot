/**
 * @file
 * @brief Reduction operations mapping tiles to vectors.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Perform a row-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout, typename T::shape>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::T2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    static_assert(!std::is_same_v<RT, fp8e4m3>, "Unsupported type for reduction");

    const int leader = laneid() % T::base_tile_rows;
    const int max_shift = T::base_tile_threads_per_reduction / 2;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        dtype accum_packed = src.tiles[i][0].data[0];
        for (int k = 1; k < src.packed_per_base_tile; k++) {
            accum_packed = op::template op<dtype>(accum_packed, src.tiles[i][0].data[k]);
        }

        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for (int k = 0; k < src.packed_per_base_tile; k++) {
                accum_packed = op::template op<dtype>(accum_packed, src.tiles[i][j].data[k]);
            }
        }
        RT accum_single = op::template op<RT>(accum_packed.x, accum_packed.y);

        if constexpr (std::is_same_v<RT, bf16> && T::base_tile_rows == 32) {
            uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(accum_single), __bfloat16_as_ushort(accum_single), false, true);
            accum_single = op::template op<RT>(__ushort_as_bfloat16(res.x), __ushort_as_bfloat16(res.y));
        }
        else if constexpr (std::is_same_v<RT, half> && T::base_tile_rows == 32) {
            uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(accum_single), __half_as_ushort(accum_single), false, true);
            accum_single = op::template op<RT>(__ushort_as_half(res.x), __ushort_as_half(res.y));
        } else if constexpr (std::is_same_v<RT, float> && T::base_tile_rows == 32) {
            uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(accum_single), __float_as_uint(accum_single), false, true);
            accum_single = op::template op<RT>(__uint_as_float(res.x), __uint_as_float(res.y));
        } else {
            for (int shift = max_shift; shift > 0; shift--) {
                accum_single = op::template op<RT>(accum_single, __shfl_down(accum_single, shift * T::base_tile_rows));
            }

            accum_single = __shfl(accum_single, leader);
        }

        if(reset) {
            row_accum[i][0] = accum_single;
        }
        else {
            row_accum[i][0] = op::template op<RT>(src_accum[i][0], accum_single);
        }
    }
}

/**
 * @brief Perform a row-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */

template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout, typename T::shape>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using RT2 = V::dtype;
    using RT = base_types::packing<RT2>::unpacked_type;

    static_assert(!std::is_same_v<RT, fp8e4m3>, "Unsupported type for reduction");

    const int leader = (laneid() / T::base_tile_cols) * T::base_tile_cols;
    const int packed_per_tile = src.packed_per_base_tile;
    const int max_shift = T::base_tile_cols / 2;

    RT2 accum[packed_per_tile];

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            accum[k] = src.tiles[i][0].data[k];
        }
        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                accum[k] = op::template op<RT2>(accum[k], src.tiles[i][j].data[k]);
            }
        }

        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            for (int shift = max_shift; shift > 0; shift /= 2) {
                accum[k] = op::template op<RT2>(accum[k], packed_shfl_down(MASK_ALL, accum[k], shift));
            }
        }

        if constexpr (reset) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                row_accum[i][k] = accum[k];
            }
        }
        else {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                row_accum[i][k] = op::template op<RT2>(src_accum[i][k], accum[k]);
            }
        }

        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            row_accum[i][k] = packed_shfl(MASK_ALL, row_accum[i][k], leader);
        }
    }
}

// Col reduction.
/**
 * @brief Perform a column-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the columns of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for row-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */

template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout, typename T::shape>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    using RT2 = V::dtype;
    using RT = base_types::packing<RT2>::unpacked_type;

    static_assert(!std::is_same_v<RT, fp8e4m3>, "Unsupported type for reduction");

    const int leader = (laneid() / T::base_tile_rows) * T::base_tile_rows;
    constexpr int packed_per_tile = T::packed_per_base_tile;
    constexpr int max_shift = T::base_tile_rows / 2;

    RT2 accum[packed_per_tile];

    #pragma unroll
    for(int j = 0; j < src.width; j++) {
        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            accum[k] = src.tiles[0][j].data[k];
        }
        #pragma unroll
        for(int i = 1; i < src.height; i++) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                accum[k] = op::template op<RT2>(accum[k], src.tiles[i][j].data[k]);
            }
        }

        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            for (int shift = max_shift; shift > 0; shift /= 2) {
                accum[k] = op::template op<RT2>(accum[k], packed_shfl_down(MASK_ALL, accum[k], shift));
            }
        }

        if constexpr (reset) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                col_accum[j][k] = accum[k];
            }
        }
        else {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                col_accum[j][k] = op::template op<RT2>(src_accum[j][k], accum[k]);
            }
        }

        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            col_accum[j][k] = packed_shfl(MASK_ALL, col_accum[j][k], leader);
        }
    }
}
/**
 * @brief Perform a column-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the columns of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout, typename T::shape>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<RT2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    static_assert(!std::is_same_v<RT, fp8e4m3>, "Unsupported type for reduction");

    const int leader = laneid() % T::base_tile_cols;
    const int max_shift = T::base_tile_threads_per_reduction / 2;

    #pragma unroll
    for(int j = 0; j < src.width; j++) { // note now width is the outer loop
        RT2 accum_packed = src.tiles[0][j].data[0];
        #pragma unroll
        for(int k = 1; k < src.packed_per_base_tile; k++) {
            accum_packed = op::template op<RT2>(accum_packed, src.tiles[0][j].data[k]);
        }
        #pragma unroll
        for(int i = 1; i < src.height; i++) { // and height is the inner loop
            #pragma unroll
            for(int k = 0; k < src.packed_per_base_tile; k++) {
                accum_packed = op::template op<RT2>(accum_packed, src.tiles[i][j].data[k]);
            }
        }

        RT accum_single = op::template op<RT>(accum_packed.x, accum_packed.y);
        
        // Graciously taken from https://github.com/triton-lang/triton/pull/7321/files
        // When numLaneToReduce == 2 && interleave == 32:
        //   step 1: use permlane32_swap() to swap the row 2 and 3 of acc and
        //           the row 0 and 1 of the copy of acc
        //   step 2: apply reduction to the result values to get final result
        if constexpr (std::is_same_v<RT, bf16> && T::base_tile_cols == 32) {
            uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(accum_single), __bfloat16_as_ushort(accum_single), false, true);
            accum_single = op::template op<RT>(__ushort_as_bfloat16(res.x), __ushort_as_bfloat16(res.y));
        }
        else if constexpr (std::is_same_v<RT, half> && T::base_tile_cols == 32) {
            uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(accum_single), __half_as_ushort(accum_single), false, true);
            accum_single = op::template op<RT>(__ushort_as_half(res.x), __ushort_as_half(res.y));
        } else if constexpr (std::is_same_v<RT, float> && T::base_tile_cols == 32) {
            uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(accum_single), __float_as_uint(accum_single), false, true);
            accum_single = op::template op<RT>(__uint_as_float(res.x), __uint_as_float(res.y));
        } else {
            for (int shift = max_shift; shift > 0; shift--) {
                accum_single = op::template op<RT>(accum_single, __shfl_down(accum_single, shift * T::base_tile_cols));
            }

            accum_single = __shfl(accum_single, leader);
        }

        if(reset) {
            col_accum[j][0] = accum_single;
        }
        else {
            col_accum[j][0] = op::template op<RT>(src_accum[j][0], accum_single);
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// two-operand row reductions. (Accumulate and REPLACE.)
/**
 * @brief Store the maximum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the sum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the product of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}
// three-operand row reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the sum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the product of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

// two-operand col reductions. (Accumulate and REPLACE.)

/**
 * @brief Store the maximum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the sum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the product of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}
// three-operand col reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the sum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the product of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

}