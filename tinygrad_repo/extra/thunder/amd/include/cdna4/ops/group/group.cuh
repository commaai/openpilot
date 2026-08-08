/**
 * @file
 * @brief An aggregate header of all group (multi-warp) operations defined by ThunderKittens
 */

#pragma once

#include "../../common/common.cuh"
#include "../../types/types.cuh"
#include "../warp/warp.cuh" // several group memory ops rely on underlying warp-scope ops

// A "warpgroup" is a special group of 4 consecutive warps defined by NVIDIA for certain SM_90+ operations.
#define KITTENS_CHECK_WARPGROUP static_assert(N_WARPS==4, "PTX warpgroup (N_WARPS=4) function called from a non-warpgroup group.");

namespace kittens {
/*
This is meant to be used with a `using group_N = kittens::group<NUM_WORKERS>;` at the start of every kernel.
*/
template<int N_WARPS>
struct group {
static constexpr int GROUP_WARPS = N_WARPS; // This alias produces nice parallelism.
static constexpr int GROUP_THREADS = N_WARPS * kittens::WARP_THREADS; // This alias produces nice parallelism.
__device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
__device__ static inline int warpid() { return laneid() / kittens::WARP_THREADS; }
__device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }

#include "memory/memory.cuh"
};

using warpgroup = group<4>; // special scope commonly used by SM_90 and later.

}