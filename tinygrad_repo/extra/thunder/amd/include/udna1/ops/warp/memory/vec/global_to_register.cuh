/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<U, fp8e4m3>, "Unsupported type for load");

    U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();

    uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    
    // TODO: this uses no inter-thread communication and is therefore not optimal.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*RV::reductions + RV::stride*(laneid/RV::aligned_threads);
            // this should be a maximally coalesced load.
            #pragma unroll
            for(int i = 0; i < RV::strides_per_tile; i++) {
                #pragma unroll
                for(int j = 0; j < RV::packed_per_stride; j++) {
                    dst[w][i * RV::packed_per_stride + j] = 
                        base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx + i * RV::elements_per_stride_group + j * RV::packing]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < RV::outer_dim; w++) {
            int idx = w * RV::reductions + (laneid % RV::reductions);
            // this should be a maximally coalesced load.
            dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        const int offset = laneid * RV::inner_dim;
        if (offset >= RV::length) return;

        constexpr int inner_dim_bytes = RV::inner_dim * sizeof(U);
        // Use buffer_load_dwordx4
        if constexpr (inner_dim_bytes % 16 == 0) {
            constexpr int elements_per_load = 16 / sizeof(U);
            #pragma unroll
            for (int i = 0; i < inner_dim_bytes / 16; i++) {
                float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                    std::bit_cast<i32x4>(br),
                    (offset * sizeof(U)) + i * 16,
                    0,
                    0
                ));
                U* tmp = reinterpret_cast<U*>(&loaded);
                #pragma unroll
                for (int j = 0; j < elements_per_load; j++) {
                    dst[0][i * elements_per_load + j] = base_types::convertor<T, U>::convert(tmp[j]);
                }
            }
        // Use buffer_load_dwordx2
        } else if constexpr (inner_dim_bytes % 8 == 0) {
            constexpr int elements_per_load = 8 / sizeof(U);
            #pragma unroll
            for (int i = 0; i < inner_dim_bytes / 8; i++) {
                float2 loaded = std::bit_cast<float2>(llvm_amdgcn_raw_buffer_load_b64(
                    std::bit_cast<i32x4>(br),
                    (offset * sizeof(U)) + i * 8,
                    0,
                    0
                ));
                U* tmp = reinterpret_cast<U*>(&loaded);
                #pragma unroll
                for (int j = 0; j < elements_per_load; j++) {
                    dst[0][i * elements_per_load + j] = base_types::convertor<T, U>::convert(tmp[j]);
                }
            }
        // Use buffer_load_dword
        } else if constexpr (inner_dim_bytes % 4 == 0) {
            constexpr int elements_per_load = 4 / sizeof(U);
            #pragma unroll
            for (int i = 0; i < inner_dim_bytes / 4; i++) {
                float loaded = std::bit_cast<float>(llvm_amdgcn_raw_buffer_load_b32(
                    std::bit_cast<i32x4>(br),
                    (offset * sizeof(U)) + i * 4,
                    0,
                    0
                ));
                U* tmp = reinterpret_cast<U*>(&loaded);
                #pragma unroll
                for (int j = 0; j < elements_per_load; j++) {
                    dst[0][i * elements_per_load + j] = base_types::convertor<T, U>::convert(tmp[j]);
                }
            }
        // fall back to direct load
        } else {
            #pragma unroll
            for (int i = 0; i < RV::inner_dim; i++) {
                dst[0][i] = base_types::convertor<T, U>::convert(src_ptr[offset + i]);
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
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(!std::is_same_v<T, fp8e4m3> && !std::is_same_v<U, fp8e4m3>, "Unsupported type for store");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();

    uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(U);
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < RV::outer_dim; w++) {
            int idx = w*RV::reductions + RV::stride*(laneid/RV::aligned_threads);
            // this should be a maximally coalesced store. I hope!
            #pragma unroll
            for (int i = 0; i < RV::strides_per_tile; i++) {
                #pragma unroll
                for (int j = 0; j < RV::packed_per_stride; j++) {
                    *(U2*)&dst_ptr[idx + i * RV::elements_per_stride_group + j * RV::packing] = base_types::convertor<U2, T2>::convert(src[w][i * RV::packed_per_stride + j]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w * RV::reductions + (laneid % RV::reductions);
            // this should be a maximally coalesced load.
            dst_ptr[idx] = base_types::convertor<U, T>::convert(src[w][0]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        const int offset = laneid * src.inner_dim;
        if (offset >= src.length) return;

        constexpr int inner_dim_bytes = RV::inner_dim * sizeof(U);

        // Use buffer_store_dwordx4
        if constexpr (inner_dim_bytes % 16 == 0) {
            constexpr int elements_per_store = 16 / sizeof(U);
            U tmp[elements_per_store];
            #pragma unroll
            for (int i = 0; i < inner_dim_bytes / 16; i++) {
                #pragma unroll
                for (int j = 0; j < elements_per_store; j++) {
                    tmp[j] = base_types::convertor<U, T>::convert(src[0][i * elements_per_store + j]);
                }
                __uint128_t val = *reinterpret_cast<__uint128_t*>(tmp);
                llvm_amdgcn_raw_buffer_store_b128(
                    val,
                    std::bit_cast<i32x4>(br),
                    (offset * sizeof(U)) + i * 16,
                    0,
                    0
                );
            }
        } else if constexpr (inner_dim_bytes % 8 == 0) {
            constexpr int elements_per_store = 8 / sizeof(U);
            U tmp[elements_per_store];
            #pragma unroll
            for (int i = 0; i < inner_dim_bytes / 8; i++) {
                #pragma unroll
                for (int j = 0; j < elements_per_store; j++) {
                    tmp[j] = base_types::convertor<U, T>::convert(src[0][i * elements_per_store + j]);
                }
                uint64_t val = *reinterpret_cast<uint64_t*>(tmp);
                llvm_amdgcn_raw_buffer_store_b64(
                    val,
                    std::bit_cast<i32x4>(br),
                    (offset * sizeof(U)) + i * 8,
                    0,
                    0
                );
            }
        } else if constexpr (inner_dim_bytes % 4 == 0) {
            constexpr int elements_per_store = 4 / sizeof(U);
            U tmp[elements_per_store];
            #pragma unroll
            for (int i = 0; i < inner_dim_bytes / 4; i++) {
                #pragma unroll
                for (int j = 0; j < elements_per_store; j++) {
                    tmp[j] = base_types::convertor<U, T>::convert(src[0][i * elements_per_store + j]);
                }
                uint32_t val = *reinterpret_cast<uint32_t*>(tmp);
                llvm_amdgcn_raw_buffer_store_b32(
                    val,
                    std::bit_cast<i32x4>(br),
                    (offset * sizeof(U)) + i * 4,
                    0,
                    0
                );
            }
        } else {
            #pragma unroll
            for (int i = 0; i < RV::inner_dim; i++) {
                dst_ptr[offset + i] = base_types::convertor<U, T>::convert(src[0][i]);
            }
        }
    }
}

} // namespace kittens