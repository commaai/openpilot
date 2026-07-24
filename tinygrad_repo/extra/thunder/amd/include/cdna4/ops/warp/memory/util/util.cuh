/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */
#pragma once

#include "../../../../common/common.cuh"
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/hip_ldg.h>

namespace kittens {

enum class coherency {
    cache_all = 0,
    cache_global = 1,
    cache_stream = 2,
    non_temporal = 3
};

/* ----------   Shared memory utilities  ---------- */
__device__ inline float2 load_shared_vec(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result)              // Output: store result in float2
        : "v"(lds_off)              // Input: LDS offset to read from
        : "memory"
    );
    return result;
}

__device__ inline void store_shared_vec(uint32_t lds_off, float2 val) {
    asm volatile(
        "ds_write_b64 %0, %1\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
}

__device__ inline float2 load_global_vec2(const float2* gptr) {
    float2 v;
    // Use global_load_dwordx2 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx2 %0, %1, off\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(v) 
        : "v"(gptr)
        : "memory"
    );
    return v;   
}

__device__ inline float4 load_global_vec4(const float4* gptr) {
    float4 v;
    // Use global_load_dwordx4 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(v) 
        : "v"(gptr)
        : "memory"
    );
    return v;   
}

__device__ inline buffer_resource make_buffer_resource(uint64_t ptr, uint32_t range, uint32_t config) {
    return {ptr, range, config};
}
__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes, uint32_t row_stride_bytes = 0) {
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(ptr);   // width = sizeof(void*)
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource rsrc = make_buffer_resource(as_u64, range_bytes, 0x110000);

    row_stride_bytes &= 0x3FFF;
    if (row_stride_bytes) {
        // - The swizzle stride lives in bits 13:0 of word2.
        //   Max value = 0x3FFF (8 KiB – one cache line per bank).
        uint64_t stride_field = row_stride_bytes;
        stride_field = stride_field | 0x4000;         // Cache swizzle
        stride_field = stride_field | 0x8000;         // Swizzle enable
        rsrc.ptr |= stride_field << 48;
    }

    return *reinterpret_cast<const i32x4*>(&rsrc);
}

__device__ uint32_t llvm_amdgcn_raw_buffer_load_b32(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b8(uint8_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i8");

__device__ void llvm_amdgcn_raw_buffer_store_b16(uint16_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ void llvm_amdgcn_raw_buffer_store_b32(uint32_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ void llvm_amdgcn_raw_buffer_store_b64(uint64_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ void llvm_amdgcn_raw_buffer_store_b128(__uint128_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i128");

using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));

extern "C" __device__ void 
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                as3_uint32_ptr lds_ptr,
                                int size,
                                int voffset, 
                                int soffset, 
                                int offset,  // does not change (0); instruction offset
                                int aux) __asm("llvm.amdgcn.raw.buffer.load.lds"); // cache coherency

/* ----------   To prevent generic addressing  ---------- */

template<typename T> struct move {
    __device__ static inline void lds(T& dst, uint32_t src);
    __device__ static inline void sts(uint32_t dst, const T& src);
    __device__ static inline void ldg(T& dst, T* src);
    __device__ static inline void stg(T* dst, const T& src);
};

// meant to be used only with shared tiles and shared vectors
namespace detail {
template<typename T> struct size_info {
    static constexpr uint32_t bytes    = sizeof(std::remove_reference_t<T>);
};
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements;
    static constexpr uint32_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;
    static constexpr uint32_t bytes    = SV::length * sizeof(typename SV::dtype);
};
}
template<typename... Args>                       inline constexpr uint32_t size_bytes             = 0; // base case
template<typename T, typename... Args>           inline constexpr uint32_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

} // namespace kittens
