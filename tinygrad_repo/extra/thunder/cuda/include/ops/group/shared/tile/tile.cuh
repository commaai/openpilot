/**
 * @file
 * @brief An aggregate header for group operations on shared tiles.
 */

#include "conversions.cuh"
#include "maps.cuh"
#include "reductions.cuh"

template<ducks::st::all ST>
__device__ static inline bool hasnan(const ST &src) {
    KITTENS_CHECK_WARP
    bool nan_detected = false;
    #pragma unroll
    for(int i = laneid(); i < ST::num_elements; i+=GROUP_THREADS) {
        if constexpr (std::is_same_v<typename ST::T, float>) {
            if(isnan(src[i])) {
                nan_detected = true;
            }
        }
        else if constexpr (std::is_same_v<typename ST::T, bf16>) {
            if(isnan(__bfloat162float(src[i]))) {
                nan_detected = true;
            }
        }
        else if constexpr (std::is_same_v<typename ST::T, half>) {
            if(isnan(__half2float(src[i]))) {
                nan_detected = true;
            }
        }
        else {
            static_assert(sizeof(typename ST::T) == 999, "Unsupported dtype");
        }
    }
    // Ballot across the warp to see if any lane detected a nan
    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}
