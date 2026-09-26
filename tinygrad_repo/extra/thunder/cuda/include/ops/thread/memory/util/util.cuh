/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */

#pragma once

namespace kittens {

/* ----------   To prevent generic addressing, PTX  ---------- */

template<typename T> struct move {
    __device__ static inline void lds(T& dst, uint32_t src);
    __device__ static inline void sts(uint32_t dst, const T& src);
    __device__ static inline void ldg(T& dst, T* src);
    __device__ static inline void stg(T* dst, const T& src);
};
// unpacked types
template<> struct move<bf16> {
    __device__ static inline void lds(bf16& dst, uint32_t src) {
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const bf16& src) {
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "r"(dst));
    }
    __device__ static inline void ldg(bf16& dst, bf16* src) {
        asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
    }
    __device__ static inline void stg(bf16* dst, const bf16& src) {
        asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
    }
};
template<> struct move<half> {
    __device__ static inline void lds(half& dst, uint32_t src) {
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const half& src) {
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "r"(dst));
    }
    __device__ static inline void ldg(half& dst, half* src) {
        asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src));
    }
    __device__ static inline void stg(half* dst, const half& src) {
        asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst));
    }
};
template<> struct move<float> {
    __device__ static inline void lds(float& dst, uint32_t src) {
        asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(dst) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const float& src) {
        asm volatile("st.shared.f32 [%1], %0;\n" : : "f"(src), "r"(dst));
    }
    __device__ static inline void ldg(float& dst, float* src) {
        asm volatile("ld.global.f32 %0, [%1];\n" : "=f"(dst) : "l"(src));
    }
    __device__ static inline void stg(float* dst, const float& src) {
        asm volatile("st.global.f32 [%1], %0;\n" : : "f"(src), "l"(dst));
    }
};
template<> struct move<int> {
    __device__ static inline void lds(int& dst, uint32_t src) {
        asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(dst) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const int& src) {
        asm volatile("st.shared.u32 [%1], %0;\n" : : "r"(src), "r"(dst));
    }
    __device__ static inline void ldg(int& dst, int* src) {
        asm volatile("ld.global.u32 %0, [%1];\n" : "=r"(dst) : "l"(src));
    }
    __device__ static inline void stg(int* dst, const int& src) {
        asm volatile("st.global.u32 [%1], %0;\n" : : "r"(src), "l"(dst));
    }
};
// packed types
template<> struct move<bf16_2> {
    __device__ static inline void lds(bf16_2& dst, uint32_t src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const bf16_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "r"(dst));
    }
    __device__ static inline void ldg(bf16_2& dst, bf16_2* src) {
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
    }
    __device__ static inline void stg(bf16_2* dst, const bf16_2& src) {
        asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
    }
    __device__ static inline void ldsm4(bf16_2& dst1, bf16_2& dst2, bf16_2& dst3, bf16_2& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    __device__ static inline void ldsm4t(bf16_2& dst1, bf16_2& dst2, bf16_2& dst3, bf16_2& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    __device__ static inline void stsm4(uint32_t dst, bf16_2& src1, bf16_2& src2, bf16_2& src3, bf16_2& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
    __device__ static inline void stsm4t(uint32_t dst, bf16_2& src1, bf16_2& src2, bf16_2& src3, bf16_2& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
};
template<> struct move<half_2> {
    __device__ static inline void lds(half_2& dst, uint32_t src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const half_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "r"(dst));
    }
    __device__ static inline void ldg(half_2& dst, half_2* src) {
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src));
    }
    __device__ static inline void stg(half_2* dst, const half_2& src) {
        asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst));
    }
    __device__ static inline void ldsm4(half_2& dst1, half_2& dst2, half_2& dst3, half_2& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    __device__ static inline void ldsm4t(half_2& dst1, half_2& dst2, half_2& dst3, half_2& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1), "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    __device__ static inline void stsm4(uint32_t dst, half_2& src1, half_2& src2, half_2& src3, half_2& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
    __device__ static inline void stsm4t(uint32_t dst, half_2& src1, half_2& src2, half_2& src3, half_2& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
};
template<> struct move<float2> {
    __device__ static inline void lds(float2& dst, uint32_t src) {
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const float2& src) {
        asm volatile("st.shared.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "r"(dst));
    }
    __device__ static inline void ldg(float2& dst, float2* src) {
        asm volatile("ld.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "l"(src));
    }
    __device__ static inline void stg(float2* dst, const float2& src) {
        asm volatile("st.global.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "l"(dst));
    }
};
template<> struct move<float4> {
    __device__ static inline void lds(float4& dst, uint32_t src) {
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "r"(src));
    }
    __device__ static inline void sts(uint32_t dst, const float4& src) {
        asm volatile("st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "r"(dst));
    }
    __device__ static inline void ldg(float4& dst, float4* src) {
        asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "l"(src));
    }
    __device__ static inline void stg(float4* dst, const float4& src) {
        asm volatile("st.global.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "l"(dst));
    }
};
#ifdef KITTENS_HOPPER
template<> struct move<fp8e4m3_4> {
    __device__ static inline void ldsm4(fp8e4m3_4& dst1, fp8e4m3_4& dst2, fp8e4m3_4& dst3, fp8e4m3_4& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1),  "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    __device__ static inline void stsm4(uint32_t dst, fp8e4m3_4& src1, fp8e4m3_4& src2, fp8e4m3_4& src3, fp8e4m3_4& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }

};
template<> struct move<fp8e5m2_4> {
    __device__ static inline void ldsm4(fp8e5m2_4& dst1, fp8e5m2_4& dst2, fp8e5m2_4& dst3, fp8e5m2_4& dst4, uint32_t src) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n" :
                     "=r"(*(uint32_t*)&dst1),  "=r"(*(uint32_t*)&dst2), "=r"(*(uint32_t*)&dst3), "=r"(*(uint32_t*)&dst4) : "r"(src));
    }
    __device__ static inline void stsm4(uint32_t dst, fp8e5m2_4& src1, fp8e5m2_4& src2, fp8e5m2_4& src3, fp8e5m2_4& src4) {
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3};\n" ::
                     "r"(*(uint32_t*)&src1), "r"(*(uint32_t*)&src2), "r"(*(uint32_t*)&src3), "r"(*(uint32_t*)&src4), "r"(dst));
    }
};
#endif

/* ----------   Constants for Cache policies  ---------- */

enum cache_policy {
    NORMAL = 0,
    EVICT_FIRST = 1,
    EVICT_LAST = 2
};
template<cache_policy policy> __device__ inline uint64_t make_cache_policy() {
    uint64_t cache_policy_val;
    constexpr float fraction = 1.0f;
    static_assert(policy == cache_policy::EVICT_FIRST || policy == cache_policy::EVICT_LAST, "Unexpected cache policy");
    if constexpr (policy == cache_policy::EVICT_FIRST) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;\n" : "=l"(cache_policy_val) : "f"(fraction));
    }
    else {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;\n" : "=l"(cache_policy_val) : "f"(fraction));
    }
    return cache_policy_val;
}
/* ----------   Generic (non-Hopper specific) semaphore functions  ---------- */

struct semaphore {
private:
    uint64_t value;
}; // note that this is an opaque type, so the value should not be accessed directly.
template<int num_warps> struct barrier {
    int barrier_id;
    __device__ __forceinline__ barrier(int _id) : barrier_id(_id) {}
    __device__ __forceinline__ barrier operator[](int i) {
        return barrier(barrier_id + i);
    }
};

/**
 * @brief Initializes a synchronization semaphore with a transaction count and sets the expected number of bytes.
 *
 * This function sets up a semaphore that is used to synchronize threads within a block during asynchronous operations.
 * It initializes the semaphore with a thread count semaphore.
 *
 * Additionally, if it is given a shared tile type, it will also call `set_bytes` to prepare for the memory transaction.
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
__device__ static inline void init_semaphore(semaphore& bar, int thread_count, int transaction_count=0) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}
/**
 * @brief Invalidate an mbarrier
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
__device__ static inline void invalidate_semaphore(semaphore& bar) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    asm volatile (
        "mbarrier.inval.shared::cta.b64 [%0];\n"
        :: "r"(bar_ptr)
    );
}

/**
* @brief Arrives at a semaphore.
*
* Marks a warp arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void arrive(semaphore& sem) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :
        : "r"(mbar_ptr)
        : "memory"
    );
}
template<int num_warps> __device__ static inline void arrive(barrier<num_warps> bar) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(bar.barrier_id), "n"(num_warps*WARP_THREADS) : "memory");
}

#ifdef KITTENS_HOPPER
/**
* @brief Arrives at a semaphore.
*
* Marks a warp arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void arrive(semaphore& sem, uint32_t count) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}
#endif

/**
* @brief Waits for the requested semaphore phase.
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

#ifdef KITTENS_HOPPER
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#else
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures to save instruction issue slots
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#endif
}

__device__ static inline void careful_wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

#ifdef KITTENS_HOPPER
    asm volatile (
        "{\n"
        ".reg .b64                 start_clock, current_clock;\n"
        "mov.b64                   start_clock, %clock64;\n"
        ".reg .pred                P_CLOCK;\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "mov.b64                   current_clock, %clock64;\n"
        "sub.u64                   current_clock, current_clock, start_clock;\n"
        "setp.ge.u64               P_CLOCK, current_clock, 1000000;\n"
        "@P_CLOCK                  trap;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#else
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures to save instruction issue slots
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#endif
}

/**
* @brief Checks if the requested semaphore phase is ready.
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline int test_wait(semaphore& sem, int kPhaseBit) {
    void const* const ptr = &sem;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    int result;
    asm volatile (
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.u32 %0,1,0,P1;"
        "}\n"
        : "=r"(result)
        : "r"(mbar_ptr), "r"(kPhaseBit)
    );
    return result;
}

__device__ static inline void arrive_and_wait(semaphore& sem, int kPhaseBit) {
    arrive(sem);
    wait(sem, kPhaseBit);
}
template<int num_warps> __device__ static inline void arrive_and_wait(barrier<num_warps> bar) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar.barrier_id), "n"(num_warps*WARP_THREADS) : "memory");
}

template<int N=0> __device__ static inline void load_async_wait() { // for completing (non-TMA) async loads
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_all;\n" ::);
    } else {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
    }
    __syncwarp();
}

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
template<typename... Args>             inline constexpr uint32_t size_bytes             = 0; // base case
template<typename T, typename... Args> inline constexpr uint32_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

} // namespace kittens

#ifdef KITTENS_HOPPER
#include "multimem.cuh"
#include "tma.cuh"
#endif

#ifdef KITTENS_BLACKWELL
#include "tensor.cuh"
#endif