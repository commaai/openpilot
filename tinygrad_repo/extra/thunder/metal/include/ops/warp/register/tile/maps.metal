#pragma once // doneington but add register tile col 
#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {
/* ----------  Uniform tile maps (independent of layout)  ---------- */

namespace meta {
template<typename op, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
unary_map_unroll(int i, int j, thread RT *dst, thread const RT *src) {
    using T2 = typename RT::T2;
    T2 vals = op::template op<T2>(T2{src->tiles[i][j].data.thread_elements()[0], src->tiles[i][j].data.thread_elements()[1]});
    dst->tiles[i][j].data.thread_elements()[0] = vals[0];
    dst->tiles[i][j].data.thread_elements()[1] = vals[1];
}
}
/**
 * @brief Applies a unary operation to each element of a tile.
 *
 * @tparam op Unary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 */
template<typename op, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
unary_map(thread RT &dst, thread const RT &src) {
    using T = typename RT::T;
    ducks::assert_register_tile<RT>();
    using T2 = typename RT::T2;
    using T4 = typename base_types::packing<typename RT::dtype>::packed_four;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < dst.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < dst.width; j++) {
//            T2 op2 = op::template op<T2>(T2{src.tiles[i][j].data.thread_elements()[0], src.tiles[i][j].data.thread_elements()[1]});
////            dst.tiles[i][j].data.thread_elements()[0] = op::template op<typename RT::dtype>(src.tiles[i][j].data.thread_elements()[0]);
////            dst.tiles[i][j].data.thread_elements()[1] = op::template op<typename RT::dtype>(src.tiles[i][j].data.thread_elements()[1]);
//            
//            dst.tiles[i][j].data.thread_elements()[0] = op2[0];
//            dst.tiles[i][j].data.thread_elements()[1] = op2[1];
//            
////            dst.tiles[i][j].data.thread_elements()[0] = base_ops::abs::template op<T>(src.tiles[i][j].data.thread_elements()[0]);
////            dst.tiles[i][j].data.thread_elements()[1] = base_ops::abs::template op<T>(src.tiles[i][j].data.thread_elements()[1]);
////            dst.tiles[i][j].data.thread_elements()[0] = (T)(metal::abs(-1.f));
////            dst.tiles[i][j].data.thread_elements()[1] = (T)(metal::abs(-1.f));
//            
////            ((T)(((float)src.tiles[i][j].data.thread_elements()[0])));
////            dst.tiles[i][j].data.thread_elements()[1] = metal::abs((T)((float)src.tiles[i][j].data.thread_elements()[1]));
//            
////            dst.tiles[i][j].data.thread_elements()[0] = base_types::constants<typename RT::dtype>::one();
////            dst.tiles[i][j].data.thread_elements()[1] = base_types::constants<typename RT::dtype>::one();
////            metal::simdgroup_barrier(metal::mem_flags::mem_none);
//            
//////            T2 val = op::template op<T2>(T2{src.tiles[i][j].data.thread_elements()[0],
//////                                            src.tiles[i][j].data.thread_elements()[1]});
//////            dst.tiles[i][j].data.thread_elements()[0] = val[0];
//////            dst.tiles[i][j].data.thread_elements()[1] = val[1];
////////            
//////            T4 val = op::template op<T4>(T4{src.tiles[i][j].data.thread_elements()[0],
//////                                            src.tiles[i][j].data.thread_elements()[1],
//////                                            src.tiles[i][j+1].data.thread_elements()[0],
//////                                            src.tiles[i][j+1].data.thread_elements()[1],});
//////            dst.tiles[i][j].data.thread_elements()[0] = val[0];
//////            dst.tiles[i][j].data.thread_elements()[1] = val[1];
//////            dst.tiles[i][j+1].data.thread_elements()[0] = val[2];
//////            dst.tiles[i][j+1].data.thread_elements()[1] = val[3];
//        }
//    }
    
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::unary_map_unroll<op, RT>, &dst, &src);
}

    
namespace meta {
template<typename op, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
bin_map_unroll(int i, int j, thread RT *dst, thread const RT *src, thread const typename RT::dtype *param) {
    using T  = typename RT::T;
    using T2 = typename RT::T2;
//    T2 vals = op::template op<T2>({src->tiles[i][j].data.thread_elements()[0], src->tiles[i][j].data.thread_elements()[1]}, {*param, *param});
//    dst->tiles[i][j].data.thread_elements()[0] = vals[0];
//    dst->tiles[i][j].data.thread_elements()[1] = vals[1];
    dst->tiles[i][j].data.thread_elements()[0] = op::template op<T>(src->tiles[i][j].data.thread_elements()[0], *param);
    dst->tiles[i][j].data.thread_elements()[1] = op::template op<T>(src->tiles[i][j].data.thread_elements()[1], *param);
}
}
/**
 * @brief Applies a binary operation to each element of a tile with a scalar parameter.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param param[in] Scalar parameter for the binary operation.
 */
template<typename op, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
bin_map(thread RT &dst, thread const RT &src, thread const typename RT::dtype &param) {
//    using T  = typename RT::T;
//    using T2 = typename RT::T2;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < dst.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < dst.width; j++) {
//            T2 vals = op::template op<T2>({src.tiles[i][j].data.thread_elements()[0], src.tiles[i][j].data.thread_elements()[1]}, {param, param});
//            dst.tiles[i][j].data.thread_elements()[0] = vals[0];
//            dst.tiles[i][j].data.thread_elements()[1] = vals[1];
////            dst.tiles[i][j].data.thread_elements()[0] = op::template op<typename RT::dtype>(src.tiles[i][j].data.thread_elements()[0], param);
////            dst.tiles[i][j].data.thread_elements()[1] = op::template op<typename RT::dtype>(src.tiles[i][j].data.thread_elements()[1], param);
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::bin_map_unroll<op, RT>, &dst, &src, &param);
}

namespace meta {
template<typename op, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
binary_map_unroll(int i, int j, thread RT *dst, thread const RT *lhs, thread const RT *rhs) {
    using T2 = typename RT::T2;
    using T4 = typename base_types::packing<typename RT::dtype>::packed_four;
    dst->tiles[i][j].data.thread_elements()[0] = op::template op<typename RT::dtype>(lhs->tiles[i][j].data.thread_elements()[0],
                                                                                    rhs->tiles[i][j].data.thread_elements()[0]);
    dst->tiles[i][j].data.thread_elements()[1] = op::template op<typename RT::dtype>(lhs->tiles[i][j].data.thread_elements()[1],
                                                                                    rhs->tiles[i][j].data.thread_elements()[1]);
//    T2 vals = op::template op<T2>({lhs->tiles[i][j].data.thread_elements()[0], lhs->tiles[i][j].data.thread_elements()[1]},
//                                                  {rhs->tiles[i][j].data.thread_elements()[0], rhs->tiles[i][j].data.thread_elements()[1]});
////
//    dst->tiles[i][j].data.thread_elements()[0] = vals[0];
//    dst->tiles[i][j].data.thread_elements()[1] = vals[1];
    
//    dst->tiles[i][j].data.thread_elements()[0] = op::template op<typename RT::dtype>(lhs->tiles[i][j].data.thread_elements()[0],
//                                                                                    rhs->tiles[i][j].data.thread_elements()[0]);
//    dst->tiles[i][j].data.thread_elements()[1] = op::template op<typename RT::dtype>(lhs->tiles[i][j].data.thread_elements()[1],
//                                                                                    rhs->tiles[i][j].data.thread_elements()[1]);
//    T4 val = op::template op<T4>(T4{src->tiles[i][j].data.thread_elements()[0],
//        src->tiles[i][j].data.thread_elements()[1],
//        src->tiles[i][j+1].data.thread_elements()[0],
//        src->tiles[i][j+1].data.thread_elements()[1]});
//    dst->tiles[i][j].data.thread_elements()[0] = val[0];
//    dst->tiles[i][j].data.thread_elements()[1] = val[1];
//    dst->tiles[i][j+1].data.thread_elements()[0] = val[2];
//    dst->tiles[i][j+1].data.thread_elements()[1] = val[3];
}
}
/**
 * @brief Applies a binary operation element-wise between two tiles.
 *
 * @tparam op Binary operation to apply.
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile for the operation.
 */
template<typename op, typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
bin_map(thread RT &dst, thread const RT &lhs, thread const RT &rhs) {
    using T = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < dst.height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < dst.width; j++) {
//            dst.tiles[i][j].data.thread_elements()[0] = op::template op<typename RT::dtype>(lhs.tiles[i][j].data.thread_elements()[0],
//                                                                                            rhs.tiles[i][j].data.thread_elements()[0]);
//            dst.tiles[i][j].data.thread_elements()[1] = op::template op<typename RT::dtype>(lhs.tiles[i][j].data.thread_elements()[1],
//                                                                                            rhs.tiles[i][j].data.thread_elements()[1]);
//            dst.tiles[i][j].data.thread_elements()[0] = lhs.tiles[i][j].data.thread_elements()[0] + rhs.tiles[i][j].data.thread_elements()[0];
//            dst.tiles[i][j].data.thread_elements()[1] = lhs.tiles[i][j].data.thread_elements()[1] + rhs.tiles[i][j].data.thread_elements()[1];
////
//            T2 vals = op::template op<T2>(T2(lhs.tiles[i][j].data.thread_elements()[0], lhs.tiles[i][j].data.thread_elements()[1]),
//                                                   T2(rhs.tiles[i][j].data.thread_elements()[0], rhs.tiles[i][j].data.thread_elements()[1]));
//            dst.tiles[i][j].data.thread_elements()[0] = vals[0];
//            dst.tiles[i][j].data.thread_elements()[1] = vals[1];
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::binary_map_unroll<op, RT>, &dst, &lhs, &rhs);
}

/* ----------  Row tile maps  ----------*/

namespace meta {
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_map_unroll(int i, int j, thread RT *dst, thread const RT *src, thread const RV *row_values) {
    using T2 = typename RT::T2;
    T2 val = op::template op<T2>({src->tiles[i][j].data.thread_elements()[0], src->tiles[i][j].data.thread_elements()[1]}, {(*row_values)[i][0], (*row_values)[i][0]});
    dst->tiles[i][j].data.thread_elements()[0] = val[0];
    dst->tiles[i][j].data.thread_elements()[1] = val[1];
}

}
/**
 * @brief Applies an operation across the rows of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_map(thread RT &dst, thread const RT &src, thread const RV &row_values) {
    static_assert(ducks::is_ortho_layout<typename RV::layout>(), "RV must be otho layout (col vec for row rt)");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rt and rv must be of same type"); // compatible type
    static_assert(RV::outer_dim == RT::height, "RV outer dim and RT height do not match"); // compatible size
    using T4 = typename base_types::packing<typename RT::dtype>::packed_four;
    using T2 = typename RT::T2;
    using T = typename RT::dtype;
    

//    #pragma clang loop unroll(full)
//    for(int i = 0; i < RT::height; i++) {
//        T row_val = row_values[i][0];
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < RT::width; j++) {
//            T2 val = op::template op<T2>({src.tiles[i][j].data.thread_elements()[0], src.tiles[i][j].data.thread_elements()[1]}, {row_val, row_val});
//            dst.tiles[i][j].data.thread_elements()[0] = val[0];
//            dst.tiles[i][j].data.thread_elements()[1] = val[1];
////            dst.tiles[i][j].data.thread_elements()[0] = op::template op<T>(src.tiles[i][j].data.thread_elements()[0], row_values[i][0]);
////            dst.tiles[i][j].data.thread_elements()[1] = op::template op<T>(src.tiles[i][j].data.thread_elements()[1], row_values[i][0]);
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::row_map_unroll<op, RT, RV>, &dst, &src, &row_values);
    
    
//    meta::unroll_i_j_in_range<0, RT::height, 1,
//                              0, (RT::width / 2) * 2, 2>::run(meta::row_map_unroll<op, RT, RV, 0, 1>, &dst, &src, &row_values);
//    meta::unroll_i_j_in_range<0, (RT::height / 2) * 2, 2,
//                             (RT::width / 2) * 2, RT::width, 1>::run(meta::row_map_unroll<op, RT, RV, 1, 0>, &dst, &src, &row_values);
//    
//    meta::unroll_i_j_in_range<(RT::height / 2) * 2, RT::height, 1,
//                              (RT::width / 2) * 2,  RT::width,  1>::run(meta::row_map_unroll<op, RT, RV>, &dst, &src, &row_values);
}
    
/**
 * @brief Applies an operation across the rows of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_map(thread RT &dst, thread const RT &src, thread const RV &row_values) {
    static_assert(ducks::is_align_layout<typename RV::layout>(), "RV must be align layout (col vec for col rt)");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rt and rv must be of same type"); // compatible type
    static_assert(RV::outer_dim == RT::height, "RV outer dim and RT height do not match"); // compatible size
    using T4 = typename base_types::packing<typename RT::dtype>::packed_four;
    using T2 = typename RT::T2;
    using T = typename RT::dtype;
    

    #pragma clang loop unroll(full)
    for(int i = 0; i < RT::height; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < RT::width; j++) {
            dst.tiles[i][j].data.thread_elements()[0] = op::template op<T>(src.tiles[i][j].data.thread_elements()[0], row_values[i][0]);
            dst.tiles[i][j].data.thread_elements()[1] = op::template op<T>(src.tiles[i][j].data.thread_elements()[1], row_values[i][1]);
        }
    }
//
//    meta::unroll_i_j_in_range<0, RT::height, 1,
//                              0, (RT::width / 2) * 2, 2>::run(meta::row_map_unroll<op, RT, RV, 0, 1>, &dst, &src, &row_values);
//    meta::unroll_i_j_in_range<0, (RT::height / 2) * 2, 2,
//                             (RT::width / 2) * 2, RT::width, 1>::run(meta::row_map_unroll<op, RT, RV, 1, 0>, &dst, &src, &row_values);
//
//    meta::unroll_i_j_in_range<(RT::height / 2) * 2, RT::height, 1,
//                              (RT::width / 2) * 2,  RT::width,  1>::run(meta::row_map_unroll<op, RT, RV>, &dst, &src, &row_values);
}

// Three-operand row map. Mostly useful for FMA instructions.

/**
 * @brief Applies an operation across the rows of two tiles in a row-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_map(thread RT &dst, thread const RT &a, thread const RT &b, thread const RV &row_values) {
    static_assert(ducks::is_ortho_layout<RV::layout>(), "rv must be ortho layout for row rt");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rt and rv must be same type"); // compatible type
    static_assert(RV::outer_dim == RT::height, "rv and rt dimensions don't match"); // compatible size
    

    using dtype = typename RT::dtype;

    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        dtype vec_val = row_values[i][0];
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            dst.tiles[i][j].data.thread_elements()[0] = op::template op<dtype>(a.tiles[i][j].data.thread_elements()[0], b.tiles[i][j].data.thread_elements()[0], vec_val);
            
            dst.tiles[i][j].data.thread_elements()[1] = op::template op<dtype>(a.tiles[i][j].data.thread_elements()[1], b.tiles[i][j].data.thread_elements()[1], vec_val);
        }
    }
}
    
/**
 * @brief Applies an operation across the rows of two tiles in a column-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with column-major layout.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param row_values[in] Column vector containing values to apply across each row.
 */
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
row_map(thread RT &dst, thread const RT &a, thread const RT &b, thread const RV &row_values) {
    static_assert(ducks::is_align_layout<RV::layout>(), "rv must be align layout for row rt");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rt and rv must be same type"); // compatible type
    static_assert(RV::outer_dim == RT::height, "rv and rt dimensions don't match"); // compatible size
    

    using dtype = typename RT::dtype;

    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            dst.tiles[i][j].data.thread_elements()[0] = op::template op<dtype>(a.tiles[i][j].data.thread_elements()[0], b.tiles[i][j].data.thread_elements()[0], row_values[i][0]);
            
            dst.tiles[i][j].data.thread_elements()[1] = op::template op<dtype>(a.tiles[i][j].data.thread_elements()[1], b.tiles[i][j].data.thread_elements()[1], row_values[i][1]);
        }
    }
}

/* ----------  Col major tile maps  ----------*/

/**
 * @brief Applies an operation across the columns of a tile in a row-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_map(thread RT &dst, thread const RT &src, thread const RV &col_values) {
    static_assert(ducks::is_align_layout<typename RV::layout>(), "rv must be align layout for row rt"); // compatible type
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rv and rt must be of the same type"); // compatible type
    static_assert(RV::outer_dim == RT::width, "rv and rt dimensions do not match"); // compatible size
    
    using dtype = typename RT::dtype;

    #pragma clang loop unroll(full)
    for(int j = 0; j < dst.width; j++) {
        #pragma clang loop unroll(full)
        for(int i = 0; i < dst.height; i++) {
            dst.tiles[i][j].data.thread_elements()[0] = op::template op<dtype>(src.tiles[i][j].data.thread_elements()[0], col_values[j][0]);
            dst.tiles[i][j].data.thread_elements()[1] = op::template op<dtype>(src.tiles[i][j].data.thread_elements()[1], col_values[j][1]);
        }
    }
}
    
/**
 * @brief Applies an operation across the columns of a tile in a col-major layout.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_map(thread RT &dst, thread const RT &src, thread const RV &col_values) {
    static_assert(ducks::is_ortho_layout<typename RV::layout>(), "rv must be ortho layout for row rt"); // compatible type
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rv and rt must be of the same type"); // compatible type
    static_assert(RV::outer_dim == RT::width, "rv and rt dimensions do not match"); // compatible size

    using dtype = typename RT::dtype;

    #pragma clang loop unroll(full)
    for(int j = 0; j < dst.width; j++) {
        #pragma clang loop unroll(full)
        for(int i = 0; i < dst.height; i++) {
            dst.tiles[i][j].data.thread_elements()[0] = op::template op<dtype>(src.tiles[i][j].data.thread_elements()[0], col_values[j][0]);
            dst.tiles[i][j].data.thread_elements()[1] = op::template op<dtype>(src.tiles[i][j].data.thread_elements()[1], col_values[j][0]);
        }
    }
}

// Three-operand col map
/**
 * @brief Applies an operation across the columns of two tiles in a row-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_map(thread RT &dst, thread const RT &a, thread const RT &b, thread const RV &col_values) {
    static_assert(ducks::is_align_layout<RV::layout>(), "rv must be align layout");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rv and rt must be of the same type"); // compatible type
    static_assert(RV::outer_dim == RT::width, "rv and rt dims don't match"); // compatible size
    

    using dtype = typename RT::dtype;

    #pragma clang loop unroll(full)
    for(int j = 0; j < dst.width; j++) {
        #pragma clang loop unroll(full)
        for(int i = 0; i < dst.height; i++) {
            dst.tiles[i][j].data.thread_elements()[0] = op::template op<dtype>(a.tiles[i][j].data.thread_elements()[0], b.tiles[i][j].data.thread_elements()[0],  col_values[j][0]);
            dst.tiles[i][j].data.thread_elements()[1] = op::template op<dtype>(a.tiles[i][j].data.thread_elements()[1], b.tiles[i][j].data.thread_elements()[1],  col_values[j][1]);
        }
    }
}
    
/**
 * @brief Applies an operation across the columns of two tiles in a row-major layout, using a third operand.
 *
 * @tparam op Operation to apply.
 * @tparam T Tile type with row-major layout.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param a[in] First source tile to apply the operation on.
 * @param b[in] Second source tile to apply the operation on.
 * @param col_values[in] Row vector containing values to apply across each column.
 */
template<typename op, typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
col_map(thread RT &dst, thread const RT &a, thread const RT &b, thread const RV &col_values) {
    static_assert(ducks::is_ortho_layout<RV::layout>(), "rv must be ortho layout");
    static_assert(metal::is_same_v<typename RV::dtype, typename RT::dtype>, "rv and rt must be of the same type"); // compatible type
    static_assert(RV::outer_dim == RT::width, "rv and rt dims don't match"); // compatible size
    

    using dtype = typename RT::dtype;

    #pragma clang loop unroll(full)
    for(int j = 0; j < dst.width; j++) {
        #pragma clang loop unroll(full)
        for(int i = 0; i < dst.height; i++) {
            dst.tiles[i][j].data.thread_elements()[0] = op::template op<dtype>(a.tiles[i][j].data.thread_elements()[0], b.tiles[i][j].data.thread_elements()[0],  col_values[j][0]);
            dst.tiles[i][j].data.thread_elements()[1] = op::template op<dtype>(a.tiles[i][j].data.thread_elements()[1], b.tiles[i][j].data.thread_elements()[1],  col_values[j][0]);
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be mittens::add_row(tile, colvec);

/**
 * @brief Sets all elements of a tile to zero.
 *
 * @tparam RT Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
zero(thread RT &dst) {
    unary_map<base_ops::zero, RT>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to one.
 *
 * @tparam RT Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
one(thread RT &dst) {
    unary_map<base_ops::one, RT>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to positive infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
pos_infty(thread RT &dst) {
    unary_map<base_ops::pos_infty, RT>(dst, dst);
}
/**
 * @brief Sets all elements of a tile to negative infinity.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
neg_infty(thread RT &dst) {
    unary_map<base_ops::neg_infty, RT>(dst, dst);
}

/**
 * @brief Applies the exponential function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
exp(thread RT &dst, thread const RT &src) {
    unary_map<base_ops::exp, RT>(dst, src);
}
/**
 * @brief Applies the exponential function to each element of a tile, in base 2.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
exp2(thread RT &dst, thread const RT &src) {
    unary_map<base_ops::exp2, RT>(dst, src);
}
/**
 * @brief Applies the natural logarithm function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the natural logarithm function on.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
log(thread RT &dst, thread const RT &src) {
    unary_map<base_ops::log, RT>(dst, src);
}
/**
 * @brief Applies the absolute value function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the absolute value function on.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
abs(thread RT &dst, thread const RT &src) {
    unary_map<base_ops::abs, RT>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function to each element of a tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the ReLU function on.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
relu(thread RT &dst, thread const RT &src) {
    unary_map<base_ops::relu, RT>(dst, src);
}
/**
 * @brief Copies the elements from one tile to another.
 *
 * @tparam T Destination tile type.
 * @tparam U Source tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to copy from.
 */
template<typename RT, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
copy(thread RT &dst, thread const U &src) {
    bin_map<base_ops::copy2, RT>(dst, dst, src);
} 

/**
 * @brief Applies the max operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template<typename RT, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
max(thread RT &dst, thread const RT &lhs, thread const U &rhs) {
    bin_map<base_ops::max, RT>(dst, lhs, rhs);
}
/**
 * @brief Applies the min operation element-wise between two tiles or a tile and a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the operation.
 * @param rhs[in] Right-hand side source tile or scalar for the operation.
 */
template<typename RT, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
min(thread RT &dst, thread const RT &lhs, thread const U &rhs) {
    bin_map<base_ops::min, RT>(dst, lhs, rhs);
}
/**
 * @brief Adds two tiles element-wise or adds a scalar to each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the addition.
 * @param rhs[in] Right-hand side source tile or scalar for the addition.
 */
template<typename RT, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
add(thread RT &dst, thread const RT &lhs, thread const U &rhs) {
    bin_map<base_ops::sum, RT>(dst, lhs, rhs);
}
/**
 * @brief Subtracts two tiles element-wise or subtracts a scalar from each element of a tile.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the subtraction.
 * @param rhs[in] Right-hand side source tile or scalar for the subtraction.
 */
template<typename RT, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
sub(thread RT &dst, const thread RT &lhs, thread const U &rhs) {
    bin_map<base_ops::sub, RT>(dst, lhs, rhs);
}
/**
 * @brief Multiplies two tiles element-wise or multiplies each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the multiplication.
 * @param rhs[in] Right-hand side source tile or scalar for the multiplication.
 */
template<typename RT, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
mul(thread RT &dst, thread const RT &lhs, thread const U &rhs) {
    bin_map<base_ops::mul, RT>(dst, lhs, rhs);
}
/**
 * @brief Divides two tiles element-wise or divides each element of a tile by a scalar.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the division.
 * @param rhs[in] Right-hand side source tile or scalar for the division.
 */
template<typename RT, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
div(thread RT &dst, thread const RT &lhs, thread const U &rhs) {
    bin_map<base_ops::div, RT>(dst, lhs, rhs);
}

/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param row_values[in] Column vector containing values to add to each row.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
add_row(thread RT &dst, thread const RT &src, thread const RV &row_values) {
    row_map<base_ops::sum, RT, RV>(dst, src, row_values);
}
/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each row.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
sub_row(thread RT &dst, thread const RT &src, thread const RV &row_values) {
    row_map<base_ops::sub, RT, RV>(dst, src, row_values);
//    using T4 = typename base_types::packing<typename RT::dtype>::packed_four;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < RT::height; i++) {
//    //        #pragma clang loop unroll(full)
//    //        for(int j = 0; j < RT::width; j+=2) {
//    //            T4 val = op::template op<T4>({src.tiles[i][j].data.thread_elements()[0],
//    //                                        src.tiles[i][j].data.thread_elements()[1],
//    //                                        src.tiles[i][j+1].data.thread_elements()[0],
//    //                                        src.tiles[i][j+1].data.thread_elements()[1],},
//    //                                         {row_values[i][0], row_values[i][0],row_values[i][0], row_values[i][0]});
//    //
//    //            dst.tiles[i][j].data.thread_elements()[0] = val[0];
//    //            dst.tiles[i][j].data.thread_elements()[1] = val[1];
//    //            dst.tiles[i][j+1].data.thread_elements()[0] = val[2];
//    //            dst.tiles[i][j+1].data.thread_elements()[1] = val[3];
//    //        }
//        
//    //        #pragma clang loop unroll(full)
//    //        for(int j = 0; j < RT::width; j++) {
//    //            T2 val = op::template op<T2>({src.tiles[i][j].data.thread_elements()[0],
//    //                                        src.tiles[i][j].data.thread_elements()[1]},
//    //                                         {row_values[i][0], row_values[i][0]});
//    //
//    //            dst.tiles[i][j].data.thread_elements()[0] = val[0];
//    //            dst.tiles[i][j].data.thread_elements()[1] = val[1];
//    //        }
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < RT::width; j+=2) {
//            T4 val = T4(src.tiles[i][j].data.thread_elements()[0],
//                        src.tiles[i][j].data.thread_elements()[1],
//                        src.tiles[i][j+1].data.thread_elements()[0],
//                        src.tiles[i][j+1].data.thread_elements()[1]) - T4(row_values[i][0], row_values[i][0], row_values[i][0], row_values[i][0]);
//            dst.tiles[i][j].data.thread_elements()[0] = val[0];
//            dst.tiles[i][j].data.thread_elements()[1] = val[1];
//            dst.tiles[i][j+1].data.thread_elements()[0] = val[2];
//            dst.tiles[i][j+1].data.thread_elements()[1] = val[3];
//        }
//    }
}
/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param row_values[in] Column vector containing values to multiply each row by.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
mul_row(thread RT &dst, thread const RT &src, thread const RV &row_values) {
//    using T = typename RT::T;
//    using T2 = typename RT::T2;
//    #pragma clang loop unroll(full)
//    for(int i = 0; i < RT::height; i++) {
//        #pragma clang loop unroll(full)
//        for(int j = 0; j < RT::width; j++) {
////            T s1 = src.tiles[i][j].data.thread_elements()[0];
////            T v1 = row_values[i][0];
////            dst.tiles[i][j].data.thread_elements()[0] = s1 * v1;
////            T s2 = src.tiles[i][j].data.thread_elements()[1];
////            T v2 = row_values[i][1];
////            dst.tiles[i][j].data.thread_elements()[1] = s2 * v2;
//            
//            
////            dst.tiles[i][j].data.thread_elements()[0] = op::template op<T>(src.tiles[i][j].data.thread_elements()[0], row_values[i][0]);
////            dst.tiles[i][j].data.thread_elements()[1] = op::template op<T>(src.tiles[i][j].data.thread_elements()[1], row_values[i][0]);
//            T2 val = op::template op<T2>({src.tiles[i][j].data.thread_elements()[0], row_values[i][0]);
//            dst.tiles[i][j].data.thread_elements()[0] = op::template op<T>(src.tiles[i][j].data.thread_elements()[0], row_values[i][0]);
//            dst.tiles[i][j].data.thread_elements()[1] = op::template op<T>(src.tiles[i][j].data.thread_elements()[1], row_values[i][0]);
//        }
//    }
    row_map<base_ops::mul, RT, RV>(dst, src, row_values);
}
/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param row_values[in] Column vector containing values to divide each row by.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
div_row(thread RT &dst, thread const RT &src, thread const RV &row_values) {
    row_map<base_ops::div, RT, RV>(dst, src, row_values);
}
/**
 * @brief Broadcast a vector into into a tile's rows.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Column vector containing values to broadcast into rows.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
broadcast_row(thread RT &dst, thread const RV &row_values) {
    row_map<base_ops::copy2, RT, RV>(dst, dst, row_values);
}


// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param col_values[in] Row vector containing values to add to each column.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
add_col(thread RT &dst, thread const RT &src, thread const RV &col_values) {
    col_map<base_ops::sum, RT, RV>(dst, src, col_values);
}
/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param col_values[in] Row vector containing values to subtract from each column.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
sub_col(thread RT &dst, thread const RT &src, thread const RV &col_values) {
    col_map<base_ops::sub, RT, RV>(dst, src, col_values);
}
/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param col_values[in] Row vector containing values to multiply each column by.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
mul_col(thread RT &dst, thread const RT &src, thread const RV &col_values) {
    col_map<base_ops::mul, RT, RV>(dst, src, col_values);
}
/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param col_values[in] Row vector containing values to divide each column by.
 */
template<typename RT, typename RV>
static METAL_FUNC void div_col(thread RT &dst, thread const RT &src, thread const RV &col_values) {
    col_map<base_ops::div, RT, RV>(dst, src, col_values);
}
/**
 * @brief Broadcast a vector into into a tile's columns.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Row vector containing values to broadcast into cols.
 */
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
broadcast_col(thread RT &dst, thread const RV &col_values) {
    col_map<base_ops::copy2, RT, RV>(dst, dst, col_values);
}
        

}
