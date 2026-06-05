/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Copies data from one register vector to another.
 *
 * @tparam RV1 The type of the destination register vector.
 * @tparam RV2 The type of the source register vector.
 * @param dst[out] The destination register vector.
 * @param src[in] The source register vector to copy from.
 */
template<ducks::rv::all RV1, ducks::rv::all RV2>
__device__ static inline void copy(RV1 &dst, const RV2 &src) {
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;

    using D1_1 = base_types::packing<D1>::unpacked_type;
    using D1_2 = base_types::packing<D1_1>::packed_type;

    using D2_1 = base_types::packing<D2>::unpacked_type;
    using D2_2 = base_types::packing<D2_1>::packed_type;

    static_assert(!(std::is_same_v<D1_1, fp8e4m3> ^ std::is_same_v<D2_1, fp8e4m3>),
                  "If either D1_1 or D2_1 is fp8e4m3, both must be fp8e4m3.");

    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) { // just a simple copy / typecast
        #pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < RV1::inner_dim; j++) {
                dst[i][j] = base_types::convertor<D1, D2>::convert(src[i][j]);
            }
        }
    }
    else { // Inner dimensions are not the same, this is really a layout conversion.
        static_assert(false, "Vector layout conversion not implemented");
    }
}
} // namespace kittens