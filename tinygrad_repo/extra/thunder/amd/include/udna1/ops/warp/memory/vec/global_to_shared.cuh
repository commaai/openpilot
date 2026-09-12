/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Stores data from a shared memory vector into global memory.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vector.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const SV &src, const COORD &idx) {
    using T = typename SV::dtype;
    using U = typename GL::dtype;

    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<U, fp8e4m3>, "Unsupported type for store");

    constexpr int bytes_per_thread = 4;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);
    constexpr int num_memcpys = (SV::length * sizeof(T)) / (N_THREADS*bytes_per_thread);

    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int elem_per_warp = bytes_per_warp / sizeof(T);
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    const T* lds_base = &src.data[0];

    if constexpr (num_memcpys > 0) {

        #pragma unroll
        for (int i = 0; i < num_memcpys; i++) {
            const int lane_elem_offset = ((warpid + i * num_warps) * elem_per_warp) + laneid * elems_per_thread;
            U* dst_elem_ptr = (U*)(dst_ptr + lane_elem_offset);
            const T* src_elem_ptr = (T*)(lds_base + lane_elem_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }

    if constexpr (num_memcpys * (N_THREADS*bytes_per_thread) != SV::length * sizeof(T)) {
        constexpr int leftover_bytes = SV::length * sizeof(T) - num_memcpys * (N_THREADS*bytes_per_thread);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;
        
        if (warpid < leftover_warps) {
            const int lane_elem_offset = ((warpid + num_memcpys * num_warps) * elem_per_warp) + laneid * elems_per_thread;

            U* dst_elem_ptr = (U*)(dst_ptr + lane_elem_offset);
            T* src_elem_ptr = (T*)(lds_base + lane_elem_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }
}
}