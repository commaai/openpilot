/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */

#pragma once // done

#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {

namespace detail {
    static METAL_FUNC int colstart_from_laneid(const int laneid) { // rowvec
        return (laneid % 2) * 2 + ((laneid / 8) % 2) * 4;
    }
    // 0,1,2,3,4,5,6,7 -> 0,2,1,3,8,10,9,11
    static METAL_FUNC int leader_from_col(const int col) { // rowvec
        return (col / 4) * 8 + (col / 2) % 2 + (col % 2) * 2;
    }
    // 0,2,1,3,8,10,9,11 -> 0,1,0,1,0,1,0,1
    static METAL_FUNC int idx_from_colleader(const int laneid) { // rowvec
        return ((laneid % 8) / 2) % 2; // % 2 to protect against non-leaders
    }
    
    static METAL_FUNC int row_from_laneid(const int laneid) { // rowvec
        return (laneid / 2) % 4 + (laneid / 16) * 4;
    }
    // 0,1,2,3,4,5,6,7 -> 0, 2, 4, 6, 16, 18, 20, 22
    static METAL_FUNC int leader_from_row(const int row) { // rowvec
        return (row/4) * 16 + (row % 4) * 2;
    }
    
    
    /* ----- ducks::is_align_register_vector<RV1>() && ducks::is_naive_register_vector<RV2>() -----*/
    static METAL_FUNC int col_leader_from_naive_laneid(const int laneid) { // rowvec
        int tile_col = laneid % 8;
        int base_leader = (tile_col / 4) * 8 + (tile_col / 2) % 2 + (tile_col % 2) * 16;
        return base_leader + 2 * (laneid / 8);
    }
    
    static METAL_FUNC int local_send_idx_from_col(const int laneid) {
        return laneid >= 16;
    }
    
    static METAL_FUNC int src_basetile_from_laneid(const int laneid) { // rowvec
        return (laneid/ 2) % 4;
    }
    
    /* ----- ducks::is_ortho_register_vector<RV1>() && ducks::is_naive_register_vector<RV2>() -----*/
    static METAL_FUNC int row_leader_from_naive_laneid(const int laneid) { // rowvec
        int row = laneid % 8;
        int base_row = (row/4) * 16 + (row % 4) * 2;
        return base_row + (laneid / 8) % 2 + (laneid >= 16) * 8;
    }
    
    static METAL_FUNC int ortho_send_tile_from_laneid(const int laneid) { // rowvec
//        uint32_t MASK_1 = 0b00000000010101010000000001010101;
        uint32_t MASK_2 = 0b00000000101010100000000010101010;
        uint32_t MASK_3 = 0b01010101000000000101010100000000;
        uint32_t MASK_4 = 0b10101010000000001010101000000000;
        return ((MASK_2 >> laneid) & 1) + ((MASK_3 >> laneid) & 1) * 2 + ((MASK_4 >> laneid) & 1) * 3;
    }
    
    

    
}
/**
 * @brief Copies data from one register vector to another.
 *
 * @tparam RV1 The type of the destination register vector.
 * @tparam RV2 The type of the source register vector.
 * @param dst[out] The destination register vector.
 * @param src[in] The source register vector to copy from.
 */
template<typename RV2, typename RV1>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV1>() && ducks::is_register_vector<RV2>(), void>::type
copy(thread RV2 &dst, thread const RV1 &src, const ushort laneid) {
    static_assert(RV1::length == RV2::length, "Outer dimensions of the register vectors must be the same.");
    using D1 = typename RV1::dtype;
    using D2 = typename RV2::dtype;
    if (metal::is_same_v<typename RV1::layout, typename RV2::layout>) {
        #pragma clang loop unroll(full)
        for(int i = 0; i < RV1::outer_dim; i++) {
            #pragma clang loop unroll(full)
            for(int j = 0; j < RV1::inner_dim; j++) {
                dst[i][j] = base_types::convertor<D1, D2>::convert(src[i][j]);
            }
        }
    } else if (ducks::is_align_register_vector<RV1>() && ducks::is_ortho_register_vector<RV2>()) { // align vector -> ortho vector
        const int row        = detail::row_from_laneid(laneid);
        const int laneid_src = detail::leader_from_col(row);
        const int send_idx   = detail::idx_from_colleader(laneid);
        #pragma clang loop unroll(full)
        for(int i = 0; i < RV1::outer_dim; i++) {
            dst[i][0] = base_types::convertor<D1,D2>::convert(shfl_sync<D2>(src[i][send_idx], laneid_src));
//            dst[i][0] = 1;
        }
    } else if (ducks::is_ortho_register_vector<RV1>() && ducks::is_align_register_vector<RV2>()) { // ortho vector -> align vector
        const int col1 = detail::colstart_from_laneid(laneid);
        const int col2 = col1 + 1;
        const int laneid_src1 = detail::leader_from_row(col1);
        const int laneid_src2 = detail::leader_from_row(col2);
        #pragma clang loop unroll(full)
        for(int i = 0; i < RV1::outer_dim; i++) {
            dst[i][0] = base_types::convertor<D2,D1>::convert(shfl_sync<D1>(src[i][0], laneid_src1));
            dst[i][1] = base_types::convertor<D2,D1>::convert(shfl_sync<D1>(src[i][0], laneid_src2));
        }
    } else if (ducks::is_align_register_vector<RV1>() && ducks::is_naive_register_vector<RV2>()) {
        const int src_laneid = detail::col_leader_from_naive_laneid(laneid);
        int align_send_tile = detail::src_basetile_from_laneid(laneid);
        int align_local_send_idx = detail::local_send_idx_from_col(laneid);
        int naive_tile_idx = 0;
        for (int l_idx = 0;
             l_idx < RV2::length;
             l_idx += 32, naive_tile_idx++, align_send_tile += 4)
        {
            D1 send_val = 0;
            if (align_send_tile < RV1::outer_dim) send_val = src[align_send_tile][align_local_send_idx];
            D1 recieve_val = shfl_sync<D1>(send_val, src_laneid);
            if (l_idx + laneid < RV2::length) dst[l_idx / 32][0] = base_types::convertor<D2,D1>::convert(recieve_val);
        }
    } else if (ducks::is_naive_register_vector<RV1>() && ducks::is_align_register_vector<RV2>()) {
        int col1 = detail::colstart_from_laneid(laneid);
        int col2 = col1 + 1;
        for (int i = 0; i < RV2::outer_dim; i++) {
            int src1 = (i%4) * 8 + col1;
            int src2 = (i%4) * 8 + col2;
            D1 send_val = src[i / 4][0];
            D1 recieve_val1 = shfl_sync<D1>(send_val, src1);
            D1 recieve_val2 = shfl_sync<D1>(send_val, src2);
            dst[i][0] = recieve_val1;
            dst[i][1] = recieve_val2;
        }
    } else if (ducks::is_ortho_register_vector<RV1>() && ducks::is_naive_register_vector<RV2>()) {
        const int src_laneid = detail::row_leader_from_naive_laneid(laneid);
        int ortho_send_tile = detail::ortho_send_tile_from_laneid(laneid);
        int naive_tile_idx = 0;
        for (int l_idx = 0; l_idx < RV2::length;
             l_idx += 32, naive_tile_idx++, ortho_send_tile += 4)
        {
            D1 send_val = 10;
            if (ortho_send_tile < RV1::outer_dim) send_val = src[ortho_send_tile][0];
            D1 recieve_val = shfl_sync<D1>(send_val, src_laneid);
            if (l_idx + laneid < RV2::length) dst[l_idx / 32][0] = base_types::convertor<D2,D1>::convert(recieve_val);
        }
    } else if (ducks::is_naive_register_vector<RV1>() && ducks::is_ortho_register_vector<RV2>()) {
        int row = detail::row_from_laneid(laneid);
        for (int i = 0; i < RV2::outer_dim; i++) {
            int src_laneid = (i%4) * 8 + row;
            D1 send_val = src[i / 4][0];
            D1 recieve_val = shfl_sync<D1>(send_val, src_laneid);
            dst[i][0] = recieve_val;
        }
    }
    else {
//        static_assert(RV1::inner_dim == RV2::inner_dim, "Something has gone deeply wrong with how register vectors were instantiated.");
    }
}

}
