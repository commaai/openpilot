/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once // done!
#include "../../../../types/types.metal"

namespace mittens {

template<typename SV, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_shared_vector<SV>() && ducks::is_global_layout<GL>(), void>::type
load(threadgroup SV &dst, thread const GL &src, thread const coord &idx, const unsigned laneid) {
    using read_type = float4;
    using U = typename GL::dtype;
    constexpr int elem_per_transfer = sizeof(read_type) / sizeof(typename SV::dtype);
    constexpr int total_calls = SV::length / elem_per_transfer; // guaranteed to divide
    device U *src_ptr = (device U*)&src.template get<SV>(idx);
    #pragma clang loop unroll(full)
    for (int i = laneid; i < total_calls; i += mittens::SIMD_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            *(threadgroup read_type*)&dst[i*elem_per_transfer] = *(device read_type*)&src_ptr[i*elem_per_transfer];
        }
    }
}

template<typename SV, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_shared_vector<SV>() && ducks::is_global_layout<GL>(), void>::type
store(thread const GL &dst, threadgroup const SV &src, thread const coord &idx, const unsigned laneid) {
    using read_type = float4;
    using U = typename GL::dtype;
    constexpr int elem_per_transfer = sizeof(read_type) / sizeof(typename SV::dtype);
    constexpr int total_calls = SV::length / elem_per_transfer; // guaranteed to divide
    device U *dst_ptr = (device U*)&dst.template get<SV>(idx);
    #pragma clang loop unroll(full)
    for (int i = laneid; i < total_calls; i += mittens::SIMD_THREADS) {
        if(i * elem_per_transfer < src.length) {
            *(device read_type*)&dst_ptr[i*elem_per_transfer] = *(threadgroup read_type*)&src[i*elem_per_transfer];
        }
    }
}

}
 
