/**
 * @file
 * @brief Warp-scope reductions on shared tiles.
 */

#pragma once

#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {

/**
 * Performs row-wise reduction on a matrix using a specified operation.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type with row layout.
 * @param row_accum The accumulator where the result of the reduction is stored.
 * @param src The source matrix on which to perform the reduction.
 * @param src_accum The initial value of the accumulator, used when reset is false.
 * @param reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 */
template<typename op, typename SV, typename ST, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_reduce(threadgroup SV &row_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid) {
    using dtype = typename SV::dtype;
    #pragma clang loop unroll(full)
    for (int row = laneid; row < ST::rows; row += mittens::SIMD_THREADS) {
        dtype accum = src[{row, 0}];
        #pragma clang loop unroll(full)
        for (int col = 1; col < src.cols; col++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            row_accum[row] = accum;
        } else {
            row_accum[row] = op::template op<dtype>(src_accum[row], accum);
        }
    }
}

/**
 * Performs column-wise reduction on a matrix using a specified operation.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The shared vector type for the column accumulator.
 * @tparam T The shared matrix type with column layout.
 * @param col_accum The accumulator where the result of the reduction is stored.
 * @param src The source matrix on which to perform the reduction.
 * @param src_accum The initial value of the accumulator, used when reset is false.
 * @param reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 */
template<typename op, typename SV, typename ST, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_reduce(threadgroup SV &col_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid) {
    using dtype = typename SV::dtype;
    #pragma clang loop unroll(full)
    for (int col = laneid; col < src.cols; col += mittens::SIMD_THREADS) {
        dtype accum = src[int2(0, col)];
        #pragma clang loop unroll(full)
        for (int row = 1; row < src.rows; row++) {
            accum = op::template op<dtype>(accum, src[int2(row, col)]);
        }
        if (reset) {
            col_accum[col] = accum;
        } else {
            col_accum[col] = op::template op<dtype>(src_accum[col], accum);
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Store the maximum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_max(threadgroup SV &row_accum, threadgroup const ST &src, const ushort laneid)  {
    row_reduce<base_ops::max, SV, ST, true>(row_accum, src, row_accum, laneid);
}
/**
 * @brief Store the minimum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_min(threadgroup SV &row_accum, threadgroup const ST &src, const ushort laneid)  {
    row_reduce<base_ops::min, SV, ST, true>(row_accum, src, row_accum, laneid);
}
/**
 * @brief Store the sum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_sum(threadgroup SV &row_accum, threadgroup const ST &src, const ushort laneid)  {
    row_reduce<base_ops::sum, SV, ST, true>(row_accum, src, row_accum, laneid);
}
/**
 * @brief Store the product of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_prod(threadgroup SV &row_accum, threadgroup const ST &src, const ushort laneid) {
    row_reduce<base_ops::mul, SV, ST, true>(row_accum, src, row_accum, laneid);
}

/**
 * @brief Store the maximum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_max(threadgroup SV &row_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid)  {
    row_reduce<base_ops::max, SV, ST, false>(row_accum, src, src_accum, laneid);
}
/**
 * @brief Store the minimum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_min(threadgroup SV &row_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid)  {
    row_reduce<base_ops::min, SV, ST, false>(row_accum, src, src_accum, laneid);
}
/**
 * @brief Store the sum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_sum(threadgroup SV &row_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid)  {
    row_reduce<base_ops::sum, SV, ST, false>(row_accum, src, src_accum, laneid);
}
/**
 * @brief Store the product of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_prod(threadgroup SV &row_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid) {
    row_reduce<base_ops::mul, SV, ST, false>(row_accum, src, src_accum, laneid);
}

/**
 * @brief Store the maximum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_max(threadgroup SV &col_accum, threadgroup const ST &src, const ushort laneid)  {
    col_reduce<base_ops::max, SV, ST, true>(col_accum, src, col_accum, laneid);
}
/**
 * @brief Store the minimum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_min(threadgroup SV &col_accum, threadgroup const ST &src, const ushort laneid)  {
    col_reduce<base_ops::min, SV, ST, true>(col_accum, src, col_accum, laneid);
}
/**
 * @brief Store the sum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_sum(threadgroup SV &col_accum, threadgroup const ST &src, const ushort laneid)  {
    col_reduce<base_ops::sum, SV, ST, true>(col_accum, src, col_accum, laneid);
}
/**
 * @brief Store the product of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_prod(threadgroup SV &col_accum, threadgroup const ST &src, const ushort laneid) {
    col_reduce<base_ops::mul, SV, ST, true>(col_accum, src, col_accum, laneid);
}

/**
 * @brief Store the maximum of each column of the src shared matrix, as well as the src_accum shared vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_max(threadgroup SV &col_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid)  {
    col_reduce<base_ops::max, SV, ST, false>(col_accum, src, src_accum, laneid);
}
/**
 * @brief Store the minimum of each column of the src shared matrix, as well as the src_accum shared vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_min(threadgroup SV &col_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid)  {
    col_reduce<base_ops::min, SV, ST, false>(col_accum, src, src_accum, laneid);
}
/**
 * @brief Store the sum of each column of the src shared tile, as well as the src_accum row vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_sum(threadgroup SV &col_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid)  {
    col_reduce<base_ops::sum, SV, ST, false>(col_accum, src, src_accum, laneid);
}
/**
 * @brief Store the product of each column of the src shared tile, as well as the src_accum row vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename SV, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
col_prod(threadgroup SV &col_accum, threadgroup const ST &src, threadgroup const SV &src_accum, const ushort laneid) {
    col_reduce<base_ops::mul, SV, ST, false>(col_accum, src, src_accum, laneid);
}

}
