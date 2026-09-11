/**
 * @file
 * @brief An aggregate header of all device (multi-GPU) operations defined by ThunderKittens
 */

#pragma once

#include "../../types/types.cuh"

namespace kittens {

template<int _NUM_DEVICES>
struct device {

static_assert(_NUM_DEVICES >= 0 && _NUM_DEVICES <= 72, "Invalid number of devices");
static constexpr int NUM_DEVICES = _NUM_DEVICES;

#ifdef KITTENS_HOPPER

using barrier_t = pgl<gl<int, 1, 1, 1, -1>, NUM_DEVICES, true>;

/**
 * @brief Multi-GPU synchronization barrier for coordinated kernel exit
 * 
 * Performs a synchronization across all devices to ensure all GPUs complete 
 * their work before any kernel exits. Does not synchronize intra-node threads
 * or threadblocks.
 * 
 * @param barrier Pre-allocated barrier structure, must be initialized to 0
 * @param dev_idx Current device index (0 to NUM_DEVICES - 1)
 * @param id Synchronization point identifier (default: 0). 0 is fine for most cases
 * 
 */
__device__ static inline void sync_on_exit(const barrier_t &barrier, const int dev_idx, const int id = 0) {
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
        threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_system> barrier_uc(barrier[dev_idx][{id}]);

        // Inter-note check-in
        multimem<int>::red<reduce_op::ADD>(barrier.mc_ptr_at({id}), 1);
        asm volatile ("{fence.proxy.alias;}" ::: "memory");
        while (barrier_uc.load(cuda::memory_order_acquire) < NUM_DEVICES);
        barrier_uc.fetch_sub(NUM_DEVICES, cuda::memory_order_release);
    }
}

#endif

};

} // namespace kittens
