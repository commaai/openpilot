/**
 * @file
 * @brief An aggregate header for warp operations on register tiles.
 */

#include "conversions.cuh"
#include "maps.cuh"
#include "reductions.cuh"

template<ducks::rt::all RT>
__device__ static inline bool hasnan(const RT &src) {
    KITTENS_CHECK_WARP
    bool nan_detected = false;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            #pragma unroll
            for(int k = 0; k < RT::packed_per_tile; k++) {
                if constexpr (std::is_same_v<typename RT::T, float>) {
                    if(isnan(src.tiles[i][j].data[k].x) || isnan(src.tiles[i][j].data[k].y)) {
                        nan_detected = true;
                    }
                }
                else if constexpr (std::is_same_v<typename RT::T, bf16>) {
                    if(isnan(__bfloat162float(src.tiles[i][j].data[k].x)) || isnan(__bfloat162float(src.tiles[i][j].data[k].y))) {
                        nan_detected = true;
                    }
                }
                else if constexpr (std::is_same_v<typename RT::T, half>) {
                    if(isnan(__half2float(src.tiles[i][j].data[k].x)) || isnan(__half2float(src.tiles[i][j].data[k].y))) {
                        nan_detected = true;
                    }
                }
                else {
                    static_assert(sizeof(typename RT::T) == 999, "Unsupported dtype");
                }
            }
        }
    }
    // Ballot across the warp to see if any lane detected a nan
    return (__ballot_sync(0xffffffff, nan_detected) != 0);
}

#include "complex/complex_conversions.cuh"
#include "complex/complex_maps.cuh"

