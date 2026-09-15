/**
 * @file
 * @brief Various utilities for group memory operations.
 */


template<int N=0> __device__ static inline void load_async_wait(int bar_id) { // for completing (non-TMA) async loads
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    sync(bar_id);
}
template<int N=0> __device__ static inline void load_async_wait() { // for completing (non-TMA) async loads
    KITTENS_CHECK_WARP
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    __syncwarp();
}

__device__ static inline void arrive(barrier<GROUP_WARPS> bar) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(bar.barrier_id), "n"(GROUP_WARPS*WARP_THREADS) : "memory");
}
__device__ static inline void arrive_and_wait(barrier<GROUP_WARPS> bar) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar.barrier_id), "n"(GROUP_WARPS*WARP_THREADS) : "memory");
}

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
    if (laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(thread_count+transaction_count)
        );
    }
}
/**
 * @brief Invalidate an mbarrier
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
__device__ static inline void invalidate_semaphore(semaphore& bar) {
    if (laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
        asm volatile (
            "mbarrier.inval.shared::cta.b64 [%0];\n"
            :: "r"(bar_ptr)
        );
    }
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
    if(laneid() == 0) {
            uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem)); 
            asm volatile (
                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
            :
            : "r"(mbar_ptr)
            : "memory"
        );
    }
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
    if(laneid() == 0) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile (
            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
            :
            : "r"(mbar_ptr), "r"(count)
            : "memory"
        );
    }
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