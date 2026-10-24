/**
 * @file
 * @brief An aggregate header of all group (multi-warp) operations defined by Thundermittens
 */

#pragma once
#include "../../common/common.metal"
#include "../../types/types.metal"
#include "../warp/warp.metal" // several group memory ops rely on underlying warp-scope ops
namespace mittens {
template<int N_WARPS>
struct group {
    constant static constexpr int GROUP_WARPS = N_WARPS; // This alias produces nice parallelism.
    constant static constexpr int GROUP_THREADS = N_WARPS * mittens::SIMD_THREADS; // This alias produces nice parallelism.
    static METAL_FUNC int simd_laneid(const unsigned threadIdx) { return threadIdx % mittens::SIMD_THREADS;         }
    static METAL_FUNC int laneid     (const unsigned threadIdx) { return threadIdx % GROUP_THREADS;                      }
    static METAL_FUNC int warpid     (const unsigned threadIdx) { return laneid(threadIdx) / mittens::SIMD_THREADS; }
    static METAL_FUNC int groupid    (const unsigned threadIdx) { return threadIdx / GROUP_THREADS;                      }
    #include "memory/memory.metal"
    #include "shared/shared.metal"
};
    

}
