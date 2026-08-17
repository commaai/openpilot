/**
 * @file
 * @brief Functions for transferring data directly between tensor memory and register memory.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "util.cuh"

namespace kittens {

__device__ static inline void tensor_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;\n");
}
__device__ static inline void tensor_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;\n");
}

__device__ inline static void tensor_load_wait() {
   asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ inline static void tensor_store_wait() {
   asm volatile("tcgen05.wait::st.sync.aligned;"); 
}

}