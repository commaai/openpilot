/**
 * @file
 * @brief Reduction operations mapping tiles to vectors.
 */

#pragma once //doneington (but register col layotus)

#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {
    
namespace meta {
    
//template<typename op, typename RT>
//static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>(), void>::type
//row_reduce_unroll_inner(int i, thread const RT *src, thread typename RT::T& accum_thread) {
//    accum_thread = op::template op<typename RT::T>(accum_thread, src->tiles[i][0].data.thread_elements()[0]);
//    accum_thread = op::template op<typename RT::T>(accum_thread, src->tiles[i][0].data.thread_elements()[1]);
//}
//
//template<typename op, typename RV, typename RT, bool reset>
//static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
//row_reduce_unroll(int i, thread RV *row_accum, thread const RT *src, thread const RV *src_accum, const short leader) {
//    using T = typename RV::T;
//    T accum_thread = op::template op<T>(src->tiles[i][0].data.thread_elements()[0], src->tiles[i][0].data.thread_elements()[1]);
//    
//    meta::unroll_i_in_range<1, RT::width, 1>::run(meta::row_reduce_unroll_inner<op, RT>, src, accum_thread);
//    accum_thread = op::template op<T>(accum_thread, shfl_down_sync<T>(accum_thread, 1));
//    accum_thread = op::template op<T>(accum_thread, shfl_down_sync<T>(accum_thread, 8));
//
//    accum_thread = shfl_sync<T>(accum_thread, leader);
//    
//    if(reset) { (*row_accum)[i][0] = accum_thread; }
//    else { (*row_accum)[i][0] = op::template op<T>((*src_accum)[i][0], accum_thread); }
//}
    
//template<typename op, typename RT>
//static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>(), void>::type
//row_reduce_unroll_inner(int i, thread const RT *src, thread typename RT::T2& accum_thread) {
//    accum_thread = op::template op<typename RT::T2>(accum_thread, {src->tiles[i][0].data.thread_elements()[0], src->tiles[i][0].data.thread_elements()[1]});
//}
    
/*
 pragma clang loop unroll(full)
 for(int i = 0; i < src.height; i++) {
     T accum_thread = op::template op<T>(src.tiles[i][0].data.thread_elements()[0], src.tiles[i][0].data.thread_elements()[1]);
     #pragma clang loop unroll(full)
     for(int j = 1; j < src.width; j++) {
         accum_thread = op::template op<T>(accum_thread, src.tiles[i][j].data.thread_elements()[0]);
         accum_thread = op::template op<T>(accum_thread, src.tiles[i][j].data.thread_elements()[1]);
     }
     accum_thread = op::template op<T>(accum_thread, shfl_down_sync<T>(accum_thread, 1));
     accum_thread = op::template op<T>(accum_thread, shfl_down_sync<T>(accum_thread, 8));

     accum_thread = shfl_sync<T>(accum_thread, leader);

     if(reset) { row_accum[i][0] = accum_thread; }
     else { row_accum[i][0] = op::template op<T>(src_accum[i][0], accum_thread); }
 }
 */

template<typename op, typename RV, typename RT, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_reduce_unroll(int i, thread RV *row_accum, thread const RT *src, thread const RV *src_accum, const short leader) {
    using T = typename RV::T;
    using T2 = typename RV::T2;
    T accum_thread = op::template op<T>(src->tiles[i][0].data.thread_elements()[0], src->tiles[i][0].data.thread_elements()[1]);
    for(int j = 1; j < src->width; j++) {
        accum_thread = op::template op<T>(accum_thread, src->tiles[i][j].data.thread_elements()[0]);
        accum_thread = op::template op<T>(accum_thread, src->tiles[i][j].data.thread_elements()[1]);
    }
    
    T shfl_val = shfl_down_sync<T>(accum_thread, 1);
    accum_thread = op::template op<T>(accum_thread, shfl_val);
    shfl_val = shfl_down_sync<T>(accum_thread, 8);
    accum_thread = op::template op<T>(accum_thread, shfl_val);

    accum_thread = shfl_sync<T>(accum_thread, leader);
    
    if(reset) {
        (*row_accum)[i][0] = accum_thread;
    }
    else {
        (*row_accum)[i][0] = op::template op<T>((*src_accum)[i][0], accum_thread);;
    }
}
    
    
}
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
template<typename op, typename RV, typename RT, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_reduce(thread RV &row_accum, thread const RT &src, thread const RV &src_accum, const short laneid) {
    static_assert(ducks::is_ortho_layout<typename RV::layout>(), "rv must be ortho for row RT");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rv and rt must be the same type"); // compatible type
    static_assert(RV::outer_dim == RT::height, "rv and rt dims don't match"); // compatible size
    using T = typename RV::T;
    using T2 = typename RV::T2;
    const short leader = (laneid / 16) * 16 + ((laneid / 2) % 4) * 2;
    
//    constexpr const uint32_t COL_0 = 0x00550055;
//    constexpr const uint32_t COL_1 = 0x00AA00AA;
//    constexpr const uint32_t COL_2 = 0x55005500;
//    constexpr const uint32_t COL_3 = 0xAA00AA00;
//    
//    constexpr const uint32_t COL_0_2 = COL_0 | COL_2;
//    constexpr const uint32_t COL_0_1 = COL_0 | COL_1;
//    constexpr const uint32_t COL_2_3 = COL_2 | COL_3;
//    const ushort src_lane1 = laneid + ((COL_0_2 >> laneid) & 1) * 1 + ((COL_1   >> laneid) & 1) * 7 - ((COL_3 >> laneid) & 1) * 9;
//    const ushort src_lane2 = laneid + ((COL_0_1 >> laneid) & 1) * 8 - ((COL_2_3 >> laneid) & 1) * 8;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        T accum_thread = op::template op<T>(src.tiles[i][0].data.thread_elements()[0], src.tiles[i][0].data.thread_elements()[1]);
//        #pragma clang loop unroll(full)
//        for(int j = 1; j < src.width; j++) {
//            accum_thread = op::template op<T>(accum_thread, src.tiles[i][j].data.thread_elements()[0]);
//            accum_thread = op::template op<T>(accum_thread, src.tiles[i][j].data.thread_elements()[1]);
//        }
//        accum_thread = op::template op<T>(accum_thread, shfl_sync<T>(accum_thread, src_lane1));
//        accum_thread = op::template op<T>(accum_thread, shfl_sync<T>(accum_thread, src_lane2));
//
//        
//        if(reset) { row_accum[i][0] = accum_thread; }
//        else { row_accum[i][0] = op::template op<T>(src_accum[i][0], accum_thread); }
//    }
    
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        T accum_thread = op::template op<T>(src.tiles[i][0].data.thread_elements()[0], src.tiles[i][0].data.thread_elements()[1]);
//        #pragma clang loop unroll(full)
//        for(int j = 1; j < src.width; j++) {
//            accum_thread = op::template op<T>(accum_thread, src.tiles[i][j].data.thread_elements()[0]);
//            accum_thread = op::template op<T>(accum_thread, src.tiles[i][j].data.thread_elements()[1]);
//        }
//        accum_thread = op::template op<T>(accum_thread, shfl_down_sync<T>(accum_thread, 1));
//        accum_thread = op::template op<T>(accum_thread, shfl_down_sync<T>(accum_thread, 8));
//
//        accum_thread = shfl_sync<T>(accum_thread, leader);
//
//        if(reset) { row_accum[i][0] = accum_thread; }
//        else { row_accum[i][0] = op::template op<T>(src_accum[i][0], accum_thread); }
//    }
        
    meta::unroll_i_in_range<0, RT::height, 1>::run(meta::row_reduce_unroll<op, RV, RT, reset>, &row_accum, &src, &src_accum, leader);
}
    
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
template<typename op, typename RV, typename RT, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_reduce(thread RV &row_accum, thread const RT &src, thread const RV &src_accum, const short laneid) {
    static_assert(ducks::is_align_layout<typename RV::layout>(), "rv must be align for row RT");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rv and rt must be the same type"); // compatible type
    static_assert(RV::outer_dim == RT::height, "rv and rt dims don't match"); // compatible size
    
    using T  = typename RV::T;
    using T2 = typename RV::T2;

    const int leader = (laneid % 2) + ((laneid / 8) % 2) * 8;
    #pragma clang loop unroll(full)
    for(int i = 0; i < src.height; i++) {
        T2 accum_thread = {src.tiles[i][0].data.thread_elements()[0], src.tiles[i][0].data.thread_elements()[1]};
        #pragma clang loop unroll(full)
        for(int j = 1; j < src.width; j++) {
            accum_thread = op::template op<T2>(accum_thread, {src.tiles[i][j].data.thread_elements()[0], src.tiles[i][j].data.thread_elements()[1]});
        }
        // Now we need to do a lil shuffle to make everyone happy.

        accum_thread = op::template op<T2>(accum_thread, shfl_down_sync<T2>(accum_thread, 2));
        accum_thread = op::template op<T2>(accum_thread, shfl_down_sync<T2>(accum_thread, 4));
        accum_thread = op::template op<T2>(accum_thread, shfl_down_sync<T2>(accum_thread, 16));

        accum_thread = shfl_sync<T2>(accum_thread, leader);

        if(reset) {
            row_accum[i][0] = accum_thread[0];
            row_accum[i][1] = accum_thread[1];
        }
        else {
            row_accum[i][0] = op::template op<T>(row_accum[i][0], accum_thread[0]);
            row_accum[i][1] = op::template op<T>(row_accum[i][1], accum_thread[1]);
        }
    }
}
    
    
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
template<typename op, typename RV, typename RT, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_reduce(thread RV &col_accum, thread const RT &src, thread const RV &src_accum, const ushort laneid) {
    static_assert(ducks::is_align_layout<typename RV::layout>(), "rv must be align layout");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rt and rv must be same type"); // compatible type
    static_assert(RV::outer_dim == RT::width, "rv and rt dims don't match"); // compatible size
    
    using dtype = typename RV::dtype;
    using T2 = typename base_types::packing<dtype>::packed_type;

    const int leader = (laneid % 2) + ((laneid / 8) % 2) * 8;
    #pragma clang loop unroll(full)
    for(int j = 0; j < src.width; j++) {
//        dtype accum_left_cols  = src.tiles[0][j].data.thread_elements()[0];
//        dtype accum_right_cols = src.tiles[0][j].data.thread_elements()[1];
        T2 accum_cols  = {src.tiles[0][j].data.thread_elements()[0], src.tiles[0][j].data.thread_elements()[1]};
//        dtype accum_right_cols = src.tiles[0][j].data.thread_elements()[1];
        #pragma clang loop unroll(full)
        for(int i = 1; i < src.height; i++) {
//            accum_left_cols  = op::template op<dtype>(accum_left_cols , src.tiles[i][j].data.thread_elements()[0]);
//            accum_right_cols = op::template op<dtype>(accum_right_cols, src.tiles[i][j].data.thread_elements()[1]);
            accum_cols = op::template op<T2>(accum_cols, {src.tiles[i][j].data.thread_elements()[0], src.tiles[i][j].data.thread_elements()[1]});
        }

//        accum_left_cols = op::template op<dtype>(accum_left_cols, shfl_down_sync<dtype>(accum_left_cols, 2));
//        accum_left_cols = op::template op<dtype>(accum_left_cols, shfl_down_sync<dtype>(accum_left_cols, 4));
//        accum_left_cols = op::template op<dtype>(accum_left_cols, shfl_down_sync<dtype>(accum_left_cols, 16));
        
//        accum_right_cols = op::template op<dtype>(accum_right_cols, shfl_down_sync<dtype>(accum_right_cols, 2));
//        accum_right_cols = op::template op<dtype>(accum_right_cols, shfl_down_sync<dtype>(accum_right_cols, 4));
//        accum_right_cols = op::template op<dtype>(accum_right_cols, shfl_down_sync<dtype>(accum_right_cols, 16));
        accum_cols = op::template op<T2>(accum_cols, shfl_down_sync<T2>(accum_cols, 2));
        accum_cols = op::template op<T2>(accum_cols, shfl_down_sync<T2>(accum_cols, 4));
        accum_cols = op::template op<T2>(accum_cols, shfl_down_sync<T2>(accum_cols, 16));

//        accum_left_cols  = shfl_sync<dtype>(accum_left_cols, leader);
//        accum_right_cols = shfl_sync<dtype>(accum_right_cols, leader);
        accum_cols = shfl_sync<T2>(accum_cols, leader);
        

        if(reset) {
//            col_accum[j][0] = accum_left_cols;
//            col_accum[j][1] = accum_right_cols;
            col_accum[j][0] = accum_cols[0];
            col_accum[j][1] = accum_cols[1];
        }
        else {
//            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_left_cols);
//            col_accum[j][1] = op::template op<dtype>(src_accum[j][1], accum_right_cols);
            col_accum[j][0] = op::template op<dtype>(src_accum[j][0], accum_cols[0]);
            col_accum[j][1] = op::template op<dtype>(src_accum[j][1], accum_cols[1]);
        }
    }
}

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
template<typename op, typename RV, typename RT, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_reduce(thread RV &col_accum, thread const RT &src, thread const RV &src_accum, const ushort laneid) {
    static_assert(ducks::is_ortho_layout<typename RV::layout>(), "rv must be ortho layout");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rt and rv must be same type"); // compatible type
    static_assert(RV::outer_dim == RT::width, "rv and rt dims don't match"); // compatible size
    
    using T = typename RV::T;
    using T2 = typename base_types::packing<T>::packed_type;

    const int leader = (laneid / 16) * 16 + ((laneid / 2) % 4) * 2; // lololol
    #pragma clang loop unroll(full)
    for(int i = 0; i < src.width; i++) {
        T accum_thread = op::template op<T>(src.tiles[0][i].data.thread_elements()[0], src.tiles[0][i].data.thread_elements()[1]);
        #pragma clang loop unroll(full)
        for(int j = 1; j < src.height; j++) {
            accum_thread = op::template op<T>(accum_thread, src.tiles[j][i].data.thread_elements()[0]);
            accum_thread = op::template op<T>(accum_thread, src.tiles[j][i].data.thread_elements()[1]);
        }
        // Now we need to do a lil shuffle to make everyone happy.

        accum_thread = op::template op<T>(accum_thread, shfl_down_sync<T>(accum_thread, 1));
        accum_thread = op::template op<T>(accum_thread, shfl_down_sync<T>(accum_thread, 8));

        accum_thread = shfl_sync<T>(accum_thread, leader);

        if(reset) {
            col_accum[i][0] = accum_thread;
        }
        else {
            col_accum[i][0] = op::template op<T>(col_accum[i][0], accum_thread);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_max(thread RV &row_accum, thread const RT &src, const int laneid)  {
    row_reduce<base_ops::max, RV, RT, true>(row_accum, src, row_accum, laneid);
}
/**
 * @brief Store the minimum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_min(thread RV &row_accum, thread const RT &src, const int laneid)  {
    row_reduce<base_ops::min, RV, RT, true>(row_accum, src, row_accum, laneid);
}
/**
 * @brief Store the sum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_sum(thread RV &row_accum, thread const RT &src, const int laneid)  {
    row_reduce<base_ops::sum, RV, RT, true>(row_accum, src, row_accum, laneid);
}
/**
 * @brief Store the product of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_prod(thread RV &row_accum, thread const RT &src, const int laneid) {
    row_reduce<base_ops::mul, RV, RT, true>(row_accum, src, row_accum, laneid);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_max(thread RV &row_accum, thread const RT &src, thread const RV &src_accum, const int laneid)  {
//    using T = typename RV::T;
//    using T2 = typename RV::T2;
//    const short leader = (laneid / 16) * 16 + ((laneid / 2) % 4) * 2;
//    
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        T accum_thread = metal::max(src.tiles[i][0].data.thread_elements()[0], src.tiles[i][0].data.thread_elements()[1]);
//        #pragma clang loop unroll(full)
//        for(int j = 1; j < src.width; j++) {
//            accum_thread = metal::max(accum_thread, src.tiles[i][j].data.thread_elements()[0]);
//            accum_thread = metal::max(accum_thread, src.tiles[i][j].data.thread_elements()[1]);
//        }
//        accum_thread = metal::max(accum_thread, shfl_down_sync<T>(accum_thread, 1));
//        accum_thread = metal::max(accum_thread, shfl_down_sync<T>(accum_thread, 8));
//        accum_thread = shfl_sync<T>(accum_thread, leader);
//        if(false) { row_accum[i][0] = accum_thread; }
//        else { row_accum[i][0] = metal::max(src_accum[i][0], accum_thread); }
//    }
    
    row_reduce<base_ops::max, RV, RT, false>(row_accum, src, src_accum, laneid);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_min(thread RV &row_accum, thread const RT &src, thread const RV &src_accum, const int laneid)  {
    row_reduce<base_ops::min, RV, RT, false>(row_accum, src, src_accum, laneid);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_sum(thread RV &row_accum, thread const RT &src, thread const RV &src_accum, const int laneid)  {
//    using T = typename RV::T;
//    using T2 = typename RV::T2;
//    const short leader = (laneid / 16) * 16 + ((laneid / 2) % 4) * 2;
//    
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        T accum_thread = (src.tiles[i][0].data.thread_elements()[0] + src.tiles[i][0].data.thread_elements()[1]);
//        #pragma clang loop unroll(full)
//        for(int j = 1; j < src.width; j++) {
//            accum_thread = (accum_thread + src.tiles[i][j].data.thread_elements()[0]);
//            accum_thread = (accum_thread + src.tiles[i][j].data.thread_elements()[1]);
//        }
//        T shfl_val = shfl_down_sync<T>(accum_thread, 1);
//        accum_thread = (accum_thread + shfl_val);
//        shfl_val = shfl_down_sync<T>(accum_thread, 8);
//        accum_thread = (accum_thread + shfl_val);
//        accum_thread = shfl_sync<T>(accum_thread, leader);
////        accum_thread = metal::simd_sum(accum_thread);
//        if(false) {
//            row_accum[i][0] = accum_thread;
//        }
//        else {
//            T src_val = src_accum[i][0];
//            row_accum[i][0] = (src_val + accum_thread);
//        }
//    }
    row_reduce<base_ops::sum, RV, RT, false>(row_accum, src, src_accum, laneid);
}
//template<typename RV, typename RT>
//static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
//row_sum(thread RV &row_accum, thread const RT &src, thread const RV &src_accum, const int laneid, const int warpId, threadgroup typename RT::T* smem)  {
//    using T = typename RV::T;
//    using T2 = typename RV::T2;
//    using T4 = typename base_types::packing<T>::packed_four;
//    const short leader = (laneid / 16) * 16 + ((laneid / 2) % 4) * 2;
//    const short qid = laneid / 4;
//    const int offsetX = (qid & 4) + (laneid / 2) % 4;
//    const int offsetY = (qid & 2) + laneid % 2;
//    const int smem_idx_row = 32 * warpId + offsetY * 4;
//    const int smem_idx = smem_idx_row + offsetX;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < src.height; i++) {
//        T accum_thread = src.tiles[i][0].data.thread_elements()[0] + src.tiles[i][0].data.thread_elements()[1];
//        #pragma clang loop unroll(full)
//        for(int j = 1; j < src.width; j++) {
//            accum_thread = accum_thread + src.tiles[i][0].data.thread_elements()[0];
//            accum_thread = accum_thread + src.tiles[i][0].data.thread_elements()[1];
//        }
//        {
//            metal::simdgroup_barrier(metal::mem_flags::mem_none);
//            smem[smem_idx] = accum_thread;
//            metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
//            T4 vals = *(threadgroup T4*)(&smem[smem_idx_row]);
//            accum_thread = vals[0] + vals[1] + vals[2] + vals[3];
//        }
//        row_accum[i][0] = src_accum[i][0] + accum_thread;
//        
//    }
//}
    
    
/**
 * @brief Store the product of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_prod(thread RV &row_accum, thread const RT &src, thread const RV &src_accum, const int laneid) {
    row_reduce<base_ops::mul, RV, RT, false>(row_accum, src, src_accum, laneid);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_max(thread RV &col_accum, thread const RT &src, const int laneid)  {
    col_reduce<base_ops::max, RV, RT, true>(col_accum, src, col_accum, laneid);
}
/**
 * @brief Store the minimum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_min(thread RV &col_accum, thread const RT &src, const int laneid)  {
    col_reduce<base_ops::min, RV, RT, true>(col_accum, src, col_accum, laneid);
}
/**
 * @brief Store the sum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_sum(thread RV &col_accum, thread const RT &src, const int laneid)  {
    col_reduce<base_ops::sum, RV, RT, true>(col_accum, src, col_accum, laneid);
}

/**
 * @brief Store the product of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_prod(thread RV &col_accum, thread const RT &src, const int laneid) {
    col_reduce<base_ops::mul, RV, RT, true>(col_accum, src, col_accum, laneid);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_max(thread RV &col_accum, thread const RT &src, thread const RV &src_accum, const int laneid)  {
    col_reduce<base_ops::max, RV, RT, false>(col_accum, src, src_accum, laneid);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_min(thread RV &col_accum, thread const RT &src, thread const RV &src_accum, const int laneid)  {
    col_reduce<base_ops::min, RV, RT, false>(col_accum, src, src_accum, laneid);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_sum(thread RV &col_accum, thread const RT &src, thread const RV &src_accum, const int laneid)  {
    col_reduce<base_ops::sum, RV, RT, false>(col_accum, src, src_accum, laneid);
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
template<typename RV, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_prod(thread RV &col_accum, thread const RT &src, thread const RV &src_accum, const int laneid) {
    col_reduce<base_ops::mul, RV, RT, false>(col_accum, src, src_accum, laneid);
}
    

}
