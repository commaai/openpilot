/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once // not done
/*
 TODO:
    prevent unnecesary memory back forth
 
 */
#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
    
/*
 "For row-vectors:
 0,2,4,6,16,18,20,22       holds %8+0 & %8 +1
 1,3,5,7,17,19,21,23       holds %8+2 & %8+3
 00000000101010100000000010101010 = 0x00AA00AA
 8,10,12,14,24,26,28,30 holds %8+4 & %8+5
 01010101000000000101010100000000 = 0x55005500
 9,11,13,15,25,27,29,31 holds %8+6 & %8+7"
 10101010000000001010101000000000 = 0xAA00AA00
 
 "For colum-vectors:
 0,1,8,9     holds %8+0
 2,3,10,11   holds %8+1
 4,5,12,13   holds %8+2
 6,7,14,15   holds %8+3
 16,17,24,25 holds %8+4
 18,19,26,27 holds %8+5
 20,21,28,29 holds %8+6
 22,23,30,31 holds %8+7
 
 0,0,4,4     holds %8+0
 1,1,5,5     holds %8+1
 2,2,6,6     holds %8+2
 3,3,7,7     holds %8+3
 8,8,12,12   holds %8+4
 9,9,13,13   holds %8+5
 10,10,14,14 holds %8+6
 11,11,15,15 holds %8+7
 "
 
 0  0  1  1  8  8  9  9
 2  2  3  3  10 10 11 11
 4  4  5  5  12 12 13 13
 6  6  7  7  14 14 15 15
 16 16 17 17 24 24 25 25
 18 18 19 19 26 26 27 27
 20 20 21 21 28 28 29 29 
 22 22 23 23 30 30 31 31
 */
// optimize later
template<typename RV, typename SV>
METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>() && ducks::is_shared_vector<SV>(), void>::type
load(thread RV &dst, threadgroup const SV &src, const short laneid) {
    using RV_T  = typename RV::dtype;
    using RV_T2 = typename base_types::packing<RV_T>::packed_type;
    using SV_T  = typename SV::dtype;
    using SV_T2 = typename base_types::packing<SV_T>::packed_type;
    
    
    static_assert(SV::tiles == RV::tiles, "RV and SV dimensions must match");
    
    if (ducks::is_align_layout<typename RV::layout>()) {
        constexpr const uint32_t MASK_1 = 0x00AA00AA; // kitty bit magic
        constexpr const uint32_t MASK_2 = 0x55005500;
        constexpr const uint32_t MASK_3 = 0xAA00AA00;
        unsigned offset = ((MASK_1 >> laneid) & 1u) * 2 + ((MASK_2 >> laneid) & 1u) * 4 + ((MASK_3 >> laneid) & 1u) * 6;
        #pragma clang loop unroll(full)
        for (int t = 0; t < SV::tiles; offset+=8, t++) {
            RV_T2 src2 = base_types::convertor<RV_T2, SV_T2>::convert(*(threadgroup SV_T2*)(&src.data[offset]));
            dst.data[t][0] = src2[0];
            dst.data[t][1] = src2[1];
//            dst.data[t][0] = 7.f;
//            dst.data[t][1] = 7.f;
        }
    } else if (ducks::is_ortho_layout<typename RV::layout>()) {
        const short laneid_div2 = laneid / 2;
        unsigned offset = laneid_div2 % 4 + (laneid_div2 / 8) * 4;
        #pragma clang loop unroll(full)
        for (int t = 0; t < SV::tiles; offset+=8, t++) {
            dst.data[t][0] = base_types::convertor<RV_T, SV_T>::convert(src[offset]);
        }
    } else if (ducks::is_naive_layout<typename RV::layout>()) {
        #pragma clang loop unroll(full)
        for(auto w = 0; w < RV::outer_dim; w++) {
            if (w * SIMD_THREADS + laneid < RV::length) {
                dst.data[w][0] = base_types::convertor<RV_T, SV_T>::convert(src[w * SIMD_THREADS + laneid]);
            }
        }
    }
}

    
/**
 * @brief Store data into a shared vector from a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
    // optimize later
template<typename SV, typename RV>
METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>() && ducks::is_shared_vector<SV>(), void>::type
store(threadgroup SV &dst, thread const RV &src, const short laneid) {
    ducks::assert_shared_vector<SV>();
    ducks::assert_register_vector<RV>();
    using RV_T  = typename RV::dtype;
    using RV_T2 = typename base_types::packing<RV_T>::packed_type;
    using SV_T  = typename SV::dtype;
    using SV_T2 = typename base_types::packing<SV_T>::packed_type;
    

    static_assert(SV::tiles == RV::tiles, "RV and SV dimensions must match");
    
    if (ducks::is_align_layout<typename RV::layout>()) {
        constexpr const uint32_t MASK_1 = 0x00AA00AA; // kitty bit magic
        constexpr const uint32_t MASK_2 = 0x55005500;
        constexpr const uint32_t MASK_3 = 0xAA00AA00;
        unsigned offset = ((MASK_1 >> laneid) & 1u) * 2 + ((MASK_2 >> laneid) & 1u) * 4 + ((MASK_3 >> laneid) & 1u) * 6;
        #pragma clang loop unroll(full)
        for (int t = 0; t < SV::tiles; offset+=8, t++) {
            SV_T2 src2 = base_types::convertor<SV_T2, RV_T2>::convert({src.data[t][0], src.data[t][1]});
            *(threadgroup SV_T2*)(&dst.data[offset]) = src2;
            
//            *(threadgroup SV_T2*)(&dst.data[offset]) = (SV_T2)1.f;
        }
    } else if (ducks::is_ortho_layout<typename RV::layout>()) {
        const short laneid_div2 = laneid / 2;
        unsigned offset = laneid_div2 % 4 + (laneid_div2 / 8) * 4;
        #pragma clang loop unroll(full)
        for (int t = 0; t < SV::tiles; offset+=8, t++) {
            dst[offset] = base_types::convertor<SV_T, RV_T>::convert(src.data[t][0]);
        }
    } else if (ducks::is_naive_layout<typename RV::layout>()) {
        #pragma clang loop unroll(full)
        for(auto w = 0; w < RV::outer_dim; w++) {
            if (w * SIMD_THREADS + laneid < RV::length) {
                dst[w * SIMD_THREADS + laneid] = base_types::convertor<SV_T, RV_T>::convert(src.data[w][0]);
            }
        }
    }
}

}



///// TRASH CAN

/*
 template<typename RV, typename SV>
 METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>() && ducks::is_shared_vector<SV>(), void>::type
 load(thread RV &dst, threadgroup const SV &src, const short laneid, const int start_tile, const int size_tile) {
     using RV_T  = typename RV::dtype;
     using RV_T2 = typename base_types::packing<RV_T>::packed_type;
     using SV_T  = typename SV::dtype;
     using SV_T2 = typename base_types::packing<SV_T>::packed_type;
     
     
 //    static_assert(RV::tiles == size_tile , "RV and SV dimensions must match");
     
     if (ducks::is_align_layout<typename RV::layout>()) {
         constexpr const uint32_t MASK_1 = 0x00AA00AA; // kitty bit magic
         constexpr const uint32_t MASK_2 = 0x55005500;
         constexpr const uint32_t MASK_3 = 0xAA00AA00;
         unsigned offset = ((MASK_1 >> laneid) & 1u) * 2 + ((MASK_2 >> laneid) & 1u) * 4 + ((MASK_3 >> laneid) & 1u) * 6
                           + 8 * start_tile;
         #pragma clang loop unroll(full)
         for (int t = start_tile; t < start_tile + size_tile; offset+=8, t++) {
 //            RV_T2 src2 = base_types::convertor<RV_T2, SV_T2>::convert(*(threadgroup SV_T2*)(&src.data[offset]));
 //            dst.data[t][0] = src2[0];
 //            dst.data[t][1] = src2[1];
         }
     } else if (ducks::is_ortho_layout<typename RV::layout>()) {
         const short laneid_div2 = laneid / 2;
         unsigned offset = laneid_div2 % 4 + (laneid_div2 / 8) * 4
                           + 8 * start_tile;
         #pragma clang loop unroll(full)
         for (int t = start_tile; t < start_tile + size_tile; offset+=8, t++) {
             dst.data[t][0] = base_types::convertor<RV_T, SV_T>::convert(src[offset]);
         }
     }
 //    else if (ducks::is_naive_layout<typename RV::layout>()) {
 //        #pragma clang loop unroll(full)
 //        for(auto w = 0; w < RV::outer_dim; w++) {
 //            if (w * SIMD_THREADS + laneid < RV::length) {
 //                dst.data[w][0] = base_types::convertor<RV_T, SV_T>::convert(src[w * SIMD_THREADS + laneid]);
 //            }
 //        }
 //    }
 }
     
 */
