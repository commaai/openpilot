/**
 * @file
 * @brief The ThunderKittens tensor memory struct.
 */

#pragma once

#include "../../common/common.cuh"

/* ----------  MAIN tt STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
namespace ducks {
/**
 * @namespace tt
 * 
 * @brief The namespace where concepts and abstract types for shared tiles live.
 */
namespace tt {
/**
 * @brief A dummy type used to identify tensor memory.
 */
struct identifier {};
/**
* @brief Concept for all tt tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as tt::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::tt::identifier
template<typename T> concept half = all<T> && T::rows ==  64;
template<typename T> concept full = all<T> && T::rows == 128;
} // namespace tt
} // namespace ducks

/**
 * @brief Shared memory tile structure for various data types and layouts.
 *
 * @tparam T The data type of the elements in the tile. Not packed!
 * @tparam _rows The height of the tile.
 * @tparam _cols The width of the tile.
 */
template<typename _T, int _rows, int _cols>
struct tt {
    using identifier = ducks::tt::identifier; ///< Type identifier for shared memory tile.
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    static constexpr int rows    = _rows;
    static constexpr int cols    = _cols;
    static constexpr int height  = rows / kittens::TILE_ROW_DIM<T>;
    static constexpr int width   = cols / kittens::TILE_COL_DIM<T>;
    
    uint32_t addr;

    __device__ inline tt() : addr(0) {}
    __device__ inline tt(uint32_t addr) : addr(addr) {}

    template<ducks::tt::all TT>  __device__ inline TT subtile(int row_offset, int col_offset) const {
#ifndef NDEBUG
        if(row_offset < 0 || row_offset+TT::rows > rows || col_offset < 0 || col_offset+TT::cols > cols) {
            printf("Subtile out of bounds! full tile rows: %d, full tile cols: %d, subtile rows: %d, subtile cols: %d, row_offset: %d, col_offset: %d\n", rows, cols, TT::rows, TT::cols, row_offset, col_offset);
            asm volatile("trap;");
        }
#endif
        return TT(addr + (row_offset<<16) + col_offset/(4/(uint32_t)sizeof(T)));
    }
    template<int transpose> __device__ inline uint32_t chunk_addr(int chunk) const {
        if constexpr (transpose) {
            if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, half> || std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>) {
                return addr + ((16 * chunk) << 16);
            }
            else {
                static_assert(sizeof(T) == 999, "Currently unsupported type for input to an mma.");
            }
        }
        else {
            if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, half>) {
                return addr + (16 * chunk / (4/(uint32_t)sizeof(T)));
            }
            else if constexpr (std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>) {
                return addr + (32 * chunk / (4/(uint32_t)sizeof(T)));
            }
            else {
                static_assert(sizeof(T) == 999, "Currently unsupported type for input to an mma.");
            }
        }
    } 

};

} // namespace kittens
