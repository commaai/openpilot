/**
 * @file
 * @brief General utilities for ThunderKittens.
 */

#pragma once

#include <stdint.h>
#include <type_traits>
#include <concepts>
#include <memory>

#include <hip/hip_runtime.h>

#include "base_types.cuh"

#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif

/**
 * @namespace kittens
 *
 * @brief The main namespace of ThunderKittens.
 */
namespace kittens {

/* ----------  GENERAL CONSTANTS FOR KITTENS  ---------- */
/**
 * @brief Constant representing number of threads in a warp.
 *
 * gfx1250 (UDNA1) is wave-32.
 */
constexpr int WARP_THREADS{32};

/**

 * @brief Get the warp ID of the current thread.
 * @return The warp ID.
 */
__device__ __forceinline__ int warpid() { return threadIdx.x >> 5; }

/**
 * @brief Get the number of warps in the threadblock.
 * @return The number of warps in the threadblock.
 */
 __device__ __forceinline__ int num_warps() { return blockDim.x / WARP_THREADS; }

/**
 * @brief Get the lane ID of the current thread within its warp.
 * @return The lane ID.
 */
__device__ __forceinline__ int laneid() { return threadIdx.x & 0x1f; }

using i32x2 = int32_t __attribute__((ext_vector_type(2)));
using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
using i32x3 = int32_t __attribute__((ext_vector_type(3)));
using u32x3 = uint32_t __attribute__((ext_vector_type(3)));
using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

/**
 * @brief Compute the ceiling division of two integers.
 * @param a The dividend.
 * @param b The divisor.
 * @return The ceiling division result.
 */
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
  }

/**
   * @brief Transform a workgroup ID to a new workgroup ID based on the chunk size and number of XCDs.
   * @param workgroup_id The original workgroup ID.
   * @param num_workgroups The total number of workgroups.
   * @param num_xcds The number of XCDs.
   * @param chunk_size The chunk size.
   * @return The new workgroup ID.
   */
   __host__ __device__ inline int chiplet_transform_chunked(
    int workgroup_id, 
    int num_workgroups,
    int num_xcds,
    int chunk_size 
) {
    // Current XCD
    int xcd = workgroup_id % num_xcds;

    // Largest full (NUM_XCDS*CHUNK_SIZE)-aligned block
    int block = num_xcds * chunk_size;
    int limit = (num_workgroups / block) * block;

    // If pid beyond the last full block, leave unchanged
    if (workgroup_id > limit) return workgroup_id;

    // Local PID (within round-robin assignment)
    int local_pid    = workgroup_id / num_xcds;
    int chunk_idx    = local_pid / chunk_size;
    int pos_in_chunk = local_pid % chunk_size;

    // New PID
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk;
}


/**
 * @brief gfx1250 LDS capacity constants.
 *
 * On gfx1250, the **LDS scratchpad and the L1 data cache are one 384 KB SRAM
 * pool per Compute Unit (CU)**, partitioned into six 64 KB segments. 
 * At least one segment must remain L1, leaving up to five segments
 * (320 KB) addressable as LDS.
 *
 * `MAX_SHARED_MEMORY_PER_SEGMENT` is one 64 KB segment; `MAX_SHARED_MEMORY` is
 * the full addressable LDS across all five segments. A kernel that fits in one
 * segment requests `MAX_SHARED_MEMORY_PER_SEGMENT`; one that needs more requests
 * a larger dynamic shared-memory size at launch via `hipFuncSetAttribute`.
 */
constexpr int MAX_SHARED_MEMORY_PER_SEGMENT = 65536;
constexpr int SHARED_MEMORY_NUM_SEGMENTS    = 5;
constexpr int MAX_SHARED_MEMORY             = MAX_SHARED_MEMORY_PER_SEGMENT * SHARED_MEMORY_NUM_SEGMENTS;
constexpr int NUM_XCDS = 1;
constexpr int CUS_PER_XCD = 64;
constexpr int NUM_CUS = CUS_PER_XCD * NUM_XCDS;

/* ----------  CUSTOM TYPES  ---------- */
typedef uint32_t      uint2_t __attribute__((ext_vector_type(2)));

/* ----------  TYPE HELPERS  ---------- */

/**
 * @namespace ducks
 *
 * @brief ThunderKittens' namespace for template metaprogramming..
 * 
 * This includes primarily dummy types and concept wrappers, along
 * with a few additional utilities.
 */
namespace ducks {

/**
 * @brief A type representing an empty default for a template.
 */
struct default_type {};

// This macro can't be done as a template, so it doesn't really have a location in kittens.
#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type

}

/* ----------  SHUFFLE UTILS  ---------- */

/**
 * @brief Mask constant for all active threads in a warp.
 */
static constexpr uint64_t MASK_ALL = 0xFFFFFFFFFFFFFFFF;

/**
 * @brief Perform a shuffle down operation on a packed type synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param delta[in] The number of positions to shuffle down.
 * @return The result of the shuffle operation.
 */
template<typename T>
__device__ static inline T packed_shfl_down(uint64_t mask, const T &f, int delta) {

    if constexpr (std::is_same_v<T, bf16_2> || std::is_same_v<T, bf16>) {
        static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
        union {
          __hip_bfloat162 bf162;
          unsigned int ui;
        } u;

        if constexpr (std::is_same_v<T, bf16_2>) {
            u.bf162 = *reinterpret_cast<const __hip_bfloat162*>(&f);
        } else {
            u.bf162 = __hip_bfloat162{*reinterpret_cast<const __hip_bfloat16*>(&f), 
                                       *reinterpret_cast<const __hip_bfloat16*>(&f)};
        }

        u.ui = __shfl_down_sync<unsigned long long, unsigned int>(mask, u.ui, delta, 64);
        if constexpr (std::is_same_v<T, bf16>) {
            return *reinterpret_cast<const T*>(&u.bf162.x);  // Extract single bf16 from the .x component
        } else {
            return u.bf162;  // Return full bf162 for bf16_2 case
        }
    } else {
        return __shfl_down(f, delta);
    }
}
template<>
__device__ inline float2 packed_shfl_down<float2>(uint64_t mask, const float2 &f, int delta) {
    float2 r;
    r.x = __shfl_down(f.x, delta);
    r.y = __shfl_down(f.y, delta);
    return r;
}
/**
 * @brief Perform a packed shuffle operation synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param src[in] The source lane from which to shuffle.
 * @return The result of the shuffle operation.
 */
template<typename T>
__device__ static inline T packed_shfl(uint64_t mask, const T &f, int src) {
    return __shfl(f, src);
}
template<>
__device__ inline bf16 packed_shfl(uint64_t mask, const bf16 &f, int src) {
    float r = __shfl(base_types::convertor<float, bf16>::convert(f), src);
    return base_types::convertor<bf16, float>::convert(r);
}
template<>
__device__ inline bf16_2 packed_shfl(uint64_t mask, const bf16_2 &f, int src) {
    float2 r;
    r.x = __shfl(base_types::convertor<float, bf16>::convert(f.x), src);
    r.y = __shfl(base_types::convertor<float, bf16>::convert(f.y), src);
    return base_types::convertor<bf16_2, float2>::convert(r);
}
template<>
__device__ inline half packed_shfl(uint64_t mask, const half &f, int src) {
    float r = __shfl(base_types::convertor<float, half>::convert(f), src);
    return base_types::convertor<half, float>::convert(r);
}
template<>
__device__ inline half_2 packed_shfl(uint64_t mask, const half_2 &f, int src) {
    float2 r;
    r.x = __shfl(base_types::convertor<float, half>::convert(f.x), src);
    r.y = __shfl(base_types::convertor<float, half>::convert(f.y), src);
    return base_types::convertor<half_2, float2>::convert(r);
}
template<>
__device__ inline float2 packed_shfl<float2>(uint64_t mask, const float2 &f, int src) {
    float2 r;
    r.x = __shfl(f.x, src);
    r.y = __shfl(f.y, src);
    return r;
}

using bytes_4  = HIP_vector_type<float, 1>;
using bytes_8  = HIP_vector_type<float, 2>;
using bytes_16 = HIP_vector_type<float, 4>;

/* ----------  SHARED MEMORY UTILS  ---------- */

// namespace ducks {
// namespace sb {
// struct identifier {};
// }
// }

// template<typename Args...>
// struct sb {
//     using identifier = ducks::sb::identifier;
//     Args... args;
// };

// namespace ducks {
// namespace sb {
// template<typename T> concept all = requires {
//     typename T::identifier;
// } && std::is_same_v<T::identifier, identifier>;
// }
// }

#define KITTENS_ALIGN_AS(n) alignas(n)
#define KITTENS_DEFAULT_ALIGN KITTENS_ALIGN_AS(16)

/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 */
struct KITTENS_DEFAULT_ALIGN alignment_dummy { int dummy; };

namespace detail {
/// @brief 16B (`int4`) vector types tagged with the address spaces the gfx1250
///        `*_load_async_to_lds_b128` builtins require (AS1 = global, AS3 = LDS).
using i32x4_vec   = int __attribute__((__vector_size__(16)));
using i32x4_gvec  = int __attribute__((__vector_size__(16))) __attribute__((address_space(1)));
using i32x4_lvec  = int __attribute__((__vector_size__(16))) __attribute__((address_space(3)));
} // namespace detail

/**
 * @brief Compile-time tag selecting an LDS segment for tile placement on gfx1250.
 *
 * Background. LDS and L1 share one 384 KB SRAM pool per Compute Unit (CU),
 * partitioned at dispatch into six 64 KB segments (see `MAX_SHARED_MEMORY`
 * above). Up to five segments (indices 0..4, total 320 KB) are addressable as
 * LDS scratchpad; at least one segment must remain L1. By convention we leave
 * segment 5 as L1, so LDS-tile placement uses indices 0..4.
 *
 * Why segments matter. The LDS half of the pool is fronted by two read ports
 * delivering 256 B/cycle each. The two ports can issue in the same cycle only
 * when they target **different** segments, so placing operand `A` in
 * `segment<0>` and operand `B` in `segment<1>` lets the hardware satisfy both
 * reads in parallel and reach the full 512 B/cycle peak. Co-locating `A` and
 * `B` in the same segment serialises them at 256 B/cycle.
 *
 * @tparam IDX 0..4 -- segment index. The allocator aligns the allocation start
 * to `IDX * 64 KB` so multiple tiles can share a single segment.
 */
template<int IDX>
struct segment {
    static_assert(IDX >= 0 && IDX < SHARED_MEMORY_NUM_SEGMENTS,
                  "segment index must be in [0, 5)");
    static constexpr int index       = IDX;
    static constexpr int byte_offset = IDX * MAX_SHARED_MEMORY_PER_SEGMENT;
};

namespace ducks {
namespace segment_tag {
template<typename T> struct is_segment : std::false_type {};
template<int I>      struct is_segment<::kittens::segment<I>> : std::true_type {};
template<typename T> concept all = is_segment<T>::value;
} // namespace segment_tag
} // namespace ducks
/**
 * @brief Very simple allocator for dynamic shared memory. Advances pointer and tracks alignments.
 *
 * Maintains a bump cursor `ptr` that advances on every `allocate*()` call. On
 * gfx1250 the allocator also remembers `base` -- the unmoved origin of the
 * shared-memory region captured at construction -- so segment-aware
 * allocations (`allocate_in<segment<IDX>>`) can jump to `base + IDX * 64 KB`
 * regardless of how far the bump cursor has already advanced.
 *
 * @tparam default_alignment The default alignment this allocator will enforce. If <=0 (default -1) it will not align.
 */
template<int default_alignment=16> 
struct shared_allocator {
    int *ptr;   ///< Bump cursor; advances on every allocate*() call.
    int *base;  ///< Frozen origin captured at construction; never moves.
                ///< Reference point for `allocate_in<segment<IDX>>` segment starts.

    private:
        // Recursive template to generate N-dimensional array type
        template<typename A, size_t... dims>
        struct variadic_array;
        template<typename A, size_t first_dim, size_t... rest_dims>
        struct variadic_array<A, first_dim, rest_dims...> {
            using type = typename variadic_array<A, rest_dims...>::type[first_dim];
        };
        template<typename A>
        struct variadic_array<A> {
            using type = A;
        };
        template<typename A, size_t... dims> 
        using variadic_array_t = typename variadic_array<A, dims...>::type;

        template<int alignment>
        __device__ inline void align_ptr() {
            if constexpr (alignment > 0) {
                uint64_t p = reinterpret_cast<uint64_t>(ptr);
                if(p % alignment != 0) {
                    ptr = (int*)(p + (alignment-(p%alignment)));
                }
            }
        }

    public:
        /**
        * @brief Construct a new shared allocator using a pointer to extern shared memory.
        *
        * `_ptr` is captured into the bump cursor `ptr`; on gfx1250 it is also
        * stashed into `base` so segment-aware allocations can recover the
        * original origin regardless of how far the cursor has advanced.
        *
        * @param[in] _ptr Pointer to the start of the extern shared memory.
        */
        __device__ shared_allocator(int *_ptr): ptr(_ptr), base(_ptr) {}
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % default_alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<default_alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }
        /**
        * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
        * @tparam alignment An alignment to enforce for this particular object.
        * @tparam A The type of the object to allocate.
        * @tparam dims... A list of dimensions for the N-dimensional array.
        * @return Reference to the allocated object.
        */
        template<int alignment, typename A, size_t... dims> 
        __device__ inline variadic_array_t<A, dims...>& allocate() {
            // static_assert(sizeof(A) % alignment == 0, "Type is not aligned properly for array allocation");
            align_ptr<alignment>();
            using at = variadic_array_t<A, dims...>;
            at*p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at)/sizeof(int);
            return *p;
        }

        /**
        * @brief Allocate shared memory inside a specific LDS segment on gfx1250.
        *
        * Positions the allocator pointer at `base + IDX * 64KB` (where `base`
        * is the dynamic-shared-memory pointer this allocator was constructed
        * with), then allocates the requested type there. Multiple
        * `allocate_in<segment<IDX>>` calls into the same segment pack tightly.
        *
        * @tparam SEG    A `kittens::segment<IDX>` tag.
        * @tparam A      The type of the object to allocate.
        * @tparam dims   Optional array dimensions.
        */
        template<typename SEG, typename A, size_t... dims>
            requires ducks::segment_tag::all<SEG>
        __device__ inline variadic_array_t<A, dims...>& allocate_in() {
            int* target = base + (SEG::byte_offset / sizeof(int));
            // If we've already allocated past the requested segment, keep
            // packing where we are; otherwise jump forward to the segment.
            if (ptr < target) ptr = target;
            using at = variadic_array_t<A, dims...>;
            at* p = reinterpret_cast<at*>(ptr);
            ptr += sizeof(at) / sizeof(int);
            return *p;
        }
};

} // namespace kittens