/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../../common/common.cuh"
#include "../../../../../types/types.cuh"
#include "../../util/util.cuh"

namespace kittens {

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<int GPR, ducks::sv::all SV>
__device__ inline static void load(const SV &src) {
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(std::is_same_v<U, float>, "shared_to_register only supports float");
    
    int laneid = ::kittens::laneid();
    
    const int lane_offset = 4*(laneid/16) + laneid%4;
    const uint32_t addr = reinterpret_cast<uintptr_t>(&src.data[0]) + lane_offset * sizeof(U);

    if constexpr (GPR >= 256) {
        asm volatile(
            "ds_read_b32 a[%0], %1 offset:%2\n"
            : 
            : "n"(GPR - 256), "v"(addr), "i"(0)
            : "memory"
        );
    } else {
        asm volatile(
            "ds_read_b32 v[%0], %1 offset:%2\n"
            : 
            : "n"(GPR), "v"(addr), "i"(0)
            : "memory"
        );
    }
}


}