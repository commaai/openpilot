#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {
/**
 * @brief A namespace for all of ThunderKittens' TMA functionality.
*/
namespace tma {

/* ----------   Barrier functions for async load  ---------- */

/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the semaphore for the first thread in the warp.
* It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @param semaphore Reference to the semaphore variable.
* @param bytes The number of bytes expected at the semaphore.
*/
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
    void const* const ptr = &bar;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(bytes));
}
/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the mbarrier before the transaction arrives.
*/
template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>);
}

/* ----------   Synchronization functions for async store  ---------- */

/**
 * @brief Commits previous asynchronous TMA stores to a group and performs them.
*/
__device__ static inline void store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;");
}
/**
 * @brief Waits for previous committed TMA store groups to complete.
 *
 * @tparam N The maximum number of remaining TMA store groups. Defaults to 0.
*/
template <int N=0>
__device__ static inline void store_async_wait() {
    asm volatile (
        "cp.async.bulk.wait_group %0;"
        :
        : "n"(N)
        : "memory"
    );
}
/**
 * @brief Waits for previous committed TMA store groups to finish reading from shared memory.
 *
 * @tparam N The maximum number of remaining TMA store groups. Defaults to 0.
*/
template <int N=0>
__device__ static inline void store_async_read_wait() {
    asm volatile (
        "cp.async.bulk.wait_group.read %0;"
        :
        : "n"(N)
        : "memory"
    );
}

/* ----------   Cluster-scope operations  ---------- */

namespace cluster {

/**
* @brief Waits for the requested semaphore phase, at cluster scope
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ static inline void careful_wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    asm volatile (
        "{\n"
        ".reg .b64                 start_clock, current_clock;\n"
        "mov.b64                   start_clock, %clock64;\n"
        ".reg .pred                P_CLOCK;\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
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
}

/**
* @brief Sets the number of bytes expected at the semaphore, assuming a multicast instruction.
*
* This function sets the number of bytes expected at the semaphore for the first thread in the warp.
* It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
* 
* It's worth being aware that this function is particularly necessary for multicast loads, and
* distributed shared memory can actually be done with a normal tma::expect followed by wait. See
* the unit tests of dsmem for an example.
*
* @param semaphore Reference to the semaphore variable.
* @param bytes The number of bytes expected at the semaphore.
*/
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes, int dst_cta) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
    uint32_t neighbor_mbar_addr;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_mbar_addr)
        : "r"(mbar_addr), "r"(dst_cta)
    );

    asm volatile ("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
        :: "r"(neighbor_mbar_addr), "r"(bytes));
}
/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the semaphore for the first thread in the warp.
* It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
* instruction to set the expected number of bytes.
*
* @tparam T The type of the data to be stored at the semaphore.
* @param semaphore Reference to the semaphore variable.
*/
/**
* @brief Sets the number of bytes expected at the semaphore.
*
* This function sets the number of bytes expected at the mbarrier before the transaction arrives.
*/
template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, int dst_cta, const T& _1, const args&... _2) {
    expect_bytes(bar, size_bytes<T, args...>, dst_cta);
}

/**
* @brief Arrives at a semaphore in cluster scope.
*
* Marks a thread arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void arrive(semaphore& bar, int dst_cta, uint32_t count=1) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
    uint32_t neighbor_mbar_addr;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_mbar_addr)
        : "r"(mbar_addr), "r"(dst_cta)
    );
    asm volatile (
        "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
        :
        : "r"(neighbor_mbar_addr), "r" (count)
        : "memory"
    );
}

// Generic transfer
__device__ static inline void store_async(void *dst, void *src, int dst_cta, uint32_t size_bytes, semaphore& bar) {
    void const* const ptr = &bar;
    uint32_t mbarrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

    // **************************************************
    // load from src to dst in different threadblocks
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    // mapa instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa 
    // find dst addr in neighbor's cta
    uint32_t neighbor_addr_dst;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_addr_dst)
        : "r"(dst_ptr), "r"(dst_cta)
    );
    
    uint32_t neighbor_addr_mbarrier = mbarrier_ptr;
    asm volatile (
        "mapa.shared::cluster.u32  %0, %1, %2;\n"
        : "=r"(neighbor_addr_mbarrier)
        : "r"(mbarrier_ptr), "r"(dst_cta)
    );
    
    // cp.async instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk 
    // copy src into dst in neighbor's cta
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile (
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :
        : "r"(neighbor_addr_dst), "r"(src_ptr), "r"(size_bytes), "r"(neighbor_addr_mbarrier)
        : "memory"
    );
}

// Templated transfer for convenience
template<typename T>
__device__ static inline void store_async(T &dst_, T &src_, int dst_cta, semaphore& bar) {
    store_async((void*)&dst_, (void*)&src_, dst_cta, size_bytes<T>, bar);
}

} // namespace cluster
} // namespace tma
} // namespace kittens