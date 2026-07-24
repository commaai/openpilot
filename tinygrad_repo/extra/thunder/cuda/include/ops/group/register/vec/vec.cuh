/**
 * @file
 * @brief An aggregate header for warp operations on register vectors.
 */

#include "conversions.cuh"
#include "maps.cuh"
#include "reductions.cuh"

template<ducks::rv::all RV>
__device__ static inline bool hasnan(const RV &src) {
    KITTENS_CHECK_WARP
    bool nan_detected = false;
    #pragma unroll
    for(int i = 0; i < RV::outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < RV::inner_dim; j++) {
            if constexpr (std::is_same_v<typename RV::dtype, typename RV::T2>) {
                if constexpr (std::is_same_v<typename RV::dtype, float2>) {
                    if(isnan(src[i][j].x) || isnan(src[i][j].y)) {
                        nan_detected = true;
                    }
                }
                else if constexpr (std::is_same_v<typename RV::dtype, bf16_2>) {
                    if(isnan(__bfloat162float(src[i][j].x)) || isnan(__bfloat162float(src[i][j].y))) {
                        nan_detected = true;
                    }
                }
                else if constexpr (std::is_same_v<typename RV::dtype, half_2>) {
                    if(isnan(__half2float(src[i][j].x)) || isnan(__half2float(src[i][j].y))) {
                        nan_detected = true;
                    }
                }
            }
            else if constexpr (std::is_same_v<typename RV::dtype, typename RV::T>) {
                if constexpr (std::is_same_v<typename RV::dtype, float>) {
                    if(isnan(src[i][j])) {
                        nan_detected = true;
                    }
                }
                else if constexpr (std::is_same_v<typename RV::dtype, bf16>) {
                    if(isnan(__bfloat162float(src[i][j]))) {
                        nan_detected = true;
                    }
                }
                else if constexpr (std::is_same_v<typename RV::dtype, half>) {
                    if(isnan(__half2float(src[i][j]))) {
                        nan_detected = true;
                    }
                }
            }
            else {
                static_assert(sizeof(typename RV::dtype) == 999, "Unsupported dtype");
            }
        }
    }
    // Ballot across the warp to see if any lane detected a nan
    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}