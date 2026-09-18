/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<U, fp8e4m3>, "Unsupported type for load");
    static_assert(SV::length == RV::length);
    
    int laneid = ::kittens::laneid();
    
    // TODO: this uses no inter-thread communication and is therefore not optimal.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for (int w = 0; w < dst.outer_dim; w++) {
            int idx = w * RV::reductions + RV::stride*(laneid/RV::aligned_threads);

            #pragma unroll
            for (int i = 0; i < RV::strides_per_tile; i++) {
                #pragma unroll
                for (int j = 0; j < RV::packed_per_stride; j++) {
                    dst[w][i * RV::packed_per_stride + j] = base_types::convertor<T2, U2>::convert(*(U2*)&src.data[idx + i * RV::elements_per_stride_group + j * RV::packing]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < RV::outer_dim; w++) {
            int idx = w * RV::reductions + (laneid % RV::reductions);
            // this should be a maximally coalesced load.
            dst[w][0] = base_types::convertor<T, U>::convert(src.data[idx]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        const int offset = laneid * RV::inner_dim;
        if (offset >= RV::length) return;
        #pragma unroll
        for(int i = 0; i < RV::inner_dim; i++) {
            int idx = offset + i;
            dst[0][i] = base_types::convertor<T, U>::convert(src.data[idx]);
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
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);
    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<U, fp8e4m3>, "Unsupported type for store");

    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < RV::outer_dim; w++) {
            int idx = w * RV::reductions + RV::stride*(laneid/RV::aligned_threads);
            // this should be a maximally coalesced store. I hope!
            #pragma unroll
            for(int i = 0; i < RV::strides_per_tile; i++) {
                #pragma unroll
                for(int j = 0; j < RV::packed_per_stride; j++) {
                    *(U2*)&dst.data[idx + i * RV::elements_per_stride_group + j * RV::packing] = base_types::convertor<U2, T2>::convert(src[w][i * RV::packed_per_stride + j]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < RV::outer_dim; w++) {
            int idx = w * RV::reductions + (laneid % RV::reductions);
            // this should be a maximally coalesced store. I hope!
            dst.data[idx] = base_types::convertor<U, T>::convert(src[w][0]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        const int offset = laneid * RV::inner_dim;
        if (offset >= RV::length) return;
        #pragma unroll
        for(int i = 0; i < RV::inner_dim; i++) {
            int idx = offset + i;
            dst.data[idx] = base_types::convertor<U, T>::convert(src[0][i]);
        }
    }
}
}