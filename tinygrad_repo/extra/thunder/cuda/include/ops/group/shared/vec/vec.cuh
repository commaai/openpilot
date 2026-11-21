/**
 * @file
 * @brief An aggregate header for group operations on shared vectors.
 */

#include "conversions.cuh"
#include "maps.cuh"
// no group vector reductions as they would require additional shared memory and synchronization, and those side effects just aren't worth it.
// warp vector reductions should be plenty fast in 99.9% of situations.

template<ducks::sv::all SV>
__device__ static inline bool hasnan(const SV &src) {
    KITTENS_CHECK_WARP
    bool nan_detected = false;
    #pragma unroll
    for(int i = laneid(); i < SV::length; i+=GROUP_THREADS) {
        if constexpr (std::is_same_v<typename SV::T, float>) {
            if(isnan(src[i])) {
                nan_detected = true;
            }
        }
        else if constexpr (std::is_same_v<typename SV::T, bf16>) {
            if(isnan(__bfloat162float(src[i]))) {
                nan_detected = true;
            }
        }
        else if constexpr (std::is_same_v<typename SV::T, half>) {
            if(isnan(__half2float(src[i]))) {
                nan_detected = true;
            }
        }
        else {
            static_assert(sizeof(typename SV::T) == 999, "Unsupported dtype");
        }
    }
    // Ballot across the warp to see if any lane detected a nan
    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}