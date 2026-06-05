/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
*/
#pragma once // not done
/*
 TODO:
    change loads/stores, prevent unnecessary
 */
#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {
/**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<typename RV, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>() && ducks::is_global_layout<GL>(), void>::type
load(thread RV &dst, thread const GL &src, thread const coord &idx, const short laneid) {
    using RV_T  = typename RV::dtype;
    using RV_T2 = typename base_types::packing<RV_T>::packed_type;
    using U     = typename GL::dtype;
    using U2    = typename base_types::packing<U>::packed_type;
    device U *src_ptr = (device U*)&src.template get<RV>(idx);
    if (ducks::is_align_layout<typename RV::layout>()) {
        constexpr const uint32_t MASK_1 = 0x00AA00AA; // kitty bit magic
        constexpr const uint32_t MASK_2 = 0x55005500;
        constexpr const uint32_t MASK_3 = 0xAA00AA00;
        unsigned offset = ((MASK_1 >> laneid) & 1u) * 2 + ((MASK_2 >> laneid) & 1u) * 4 + ((MASK_3 >> laneid) & 1u) * 6;
        #pragma clang loop unroll(full)
        for (int t = 0; t < RV::outer_dim; offset+=8, t++) {
            RV_T2 src2 = base_types::convertor<RV_T2, U2>::convert(*(device U2*)(&src_ptr[offset]));
            dst.data[t][0] = src2[0];
            dst.data[t][1] = src2[1];
        }
    } else if (ducks::is_ortho_layout<typename RV::layout>()) { // RV::inner_dim == 1
        const short laneid_div2 = laneid / 2;
        unsigned offset = laneid_div2 % 4 + (laneid_div2 / 8) * 4;
        #pragma clang loop unroll(full)
        for (int t = 0; t < RV::outer_dim; offset+=8, t++) {
            dst.data[t][0] = base_types::convertor<RV_T, U>::convert(src_ptr[offset]);
        }
    } else if (ducks::is_naive_layout<typename RV::layout>()) {
        #pragma clang loop unroll(full)
        for(auto w = 0; w < RV::outer_dim; w++) {
//            if(w < dst.outer_dim-1 || dst.length%32 == 0 || laneid<16) {
            if (w * SIMD_THREADS + laneid < RV::length) {
                dst[w][0] = base_types::convertor<RV_T, U>::convert(src_ptr[w * SIMD_THREADS + laneid]);
            }
        }
    }
}

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<typename RV, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>() && ducks::is_global_layout<GL>(), void>::type
store(thread GL &dst, thread const RV &src, thread const coord &idx, const short laneid) {
    using RV_T  = typename RV::dtype;
    using RV_T2 = typename base_types::packing<RV_T>::packed_type;
    using U     = typename GL::dtype;
    using U2    = typename base_types::packing<U>::packed_type;
    device U *dst_ptr = (device U*)&(dst.template get<RV>(idx));
    if (ducks::is_align_layout<typename RV::layout>()) {
        constexpr const uint32_t MASK_1 = 0x00AA00AA; // kitty bit magic
        constexpr const uint32_t MASK_2 = 0x55005500;
        constexpr const uint32_t MASK_3 = 0xAA00AA00;
        unsigned offset = ((MASK_1 >> laneid) & 1u) * 2 + ((MASK_2 >> laneid) & 1u) * 4 + ((MASK_3 >> laneid) & 1u) * 6;
        #pragma clang loop unroll(full)
        for (int t = 0; t < RV::outer_dim; offset+=8, t++) {
            U2 src2 = base_types::convertor<U2, RV_T2>::convert({src.data[t][0], src.data[t][1]});
            *(device U2*)(&dst_ptr[offset]) = src2;
        }
    } else if (ducks::is_ortho_layout<typename RV::layout>()){ // RV::inner_dim == 1
        const short laneid_div2 = laneid / 2;
        unsigned offset = laneid_div2 % 4 + (laneid_div2 / 8) * 4;
        #pragma clang loop unroll(full)
        for (int t = 0; t < RV::outer_dim; offset+=8, t++) {
            dst_ptr[offset] = base_types::convertor<U, RV_T>::convert(src.data[t][0]);
        }
    } else {
        #pragma clang loop unroll(full)
        for(auto w = 0; w < RV::outer_dim; w++) {
        //            if(w < dst.outer_dim-1 || dst.length%32 == 0 || laneid<16) {
            if (w * SIMD_THREADS + laneid < RV::length) {
                dst_ptr[w * SIMD_THREADS + laneid] = base_types::convertor<U, RV_T>::convert(src.data[w][0]);
            }
        }
    }
}

}
