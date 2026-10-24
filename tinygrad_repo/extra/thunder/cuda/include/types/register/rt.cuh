/**
 * @file
 * @brief The main ThunderKittens register tile struct, where most computation happens.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

#include "rt_layout.cuh"
#include "rt_base.cuh"
#include "rv.cuh"

namespace kittens {

/* ----------  MAIN TILE STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
 * @namespace rt
 * 
 * @brief The namespace where concepts and abstract types for register tiles live.
 */
namespace rt {
/**
 * @brief A dummy type used to identify register tiles.
 * 
 * For a type to quack like an rt, it should define its identifier as ducks::rt::identifier.
 * If a type quacks like ducks::rt::identifier, it will be treated as an rt by compiler checks.
 */
struct identifier {};
/**
* @brief Concept for all register tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rt::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
/**
* @brief Concept for register tiles with row layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has an internal type layout that is ducks::rt_layout::row.
*/
template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::row>;
/**
* @brief Concept for register tiles with col layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has an internal type layout that is ducks::rt_layout::col.
*/
template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::rt_layout::col>;
} // namespace rt
} // namespace ducks

/**
 * @brief Main tile structure for manipulating data in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the internal base tiles, either row-major or column-major.
 *
 * This structure is designed to handle matrix tiles in a flexible manner, allowing
 * for operations on tiles that are composed of smaller subtiles. It supports both
 * row-major and column-major layouts and includes helper structs for type inference
 * in vector maps.
 * 
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 */
template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct rt {
    using identifier = ducks::rt::identifier; ///< Type identifier for the rt structure.
    using layout = _layout; ///< Layout of the matrix tile.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2; ///< Data type of the matrix elements

    static constexpr int rows                = _rows; ///< Total number of rows.
    static_assert(rows % rt_base<T, layout>::tile_size_row == 0, "Rows must be divisible by the tile size");
    static constexpr int cols                = _cols; ///< Total number of columns.
    static_assert(cols % rt_base<T, layout>::tile_size_col == 0, "Columns must be divisible by the tile size");
    static constexpr int height              = rows / rt_base<T, layout>::tile_size_row; ///< Height in subtiles.
    static constexpr int width               = cols / rt_base<T, layout>::tile_size_col; ///< Width in subtiles.
    static constexpr int tile_size_row        = rt_base<T, layout>::tile_size_row;        ///< Size of the base tile.
    static constexpr int tile_size_col        = rt_base<T, layout>::tile_size_col;        ///< Size of the base tile.
    static constexpr int num_elements        = rt_base<T, layout>::num_elements        * width * height; ///< Total number of elements.
    static constexpr int elements_per_thread = rt_base<T, layout>::elements_per_thread * width * height; ///< Elements handled per thread.
    static constexpr int packed_per_thread   = rt_base<T, layout>::packed_per_thread   * width * height; ///< Packed elements per thread.
    static constexpr int packed_per_tile     = rt_base<T, layout>::packed_per_thread; ///< Packed elements per tile.

    rt_base<T, layout> tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.

    using row_vec = rv<T, cols, typename rt_base<T, layout>::row_vec_layout>; ///< A type representing a column vector for this tile.
    using col_vec = rv<T, rows, typename rt_base<T, layout>::col_vec_layout>; ///< A type representing a column vector for this tile.

    __device__ inline void operator=(const T &value) {
        T2 value2 = base_types::packing<T>::pack(value);
        #pragma unroll
        for(int i = 0; i < height; i++) {
            #pragma unroll
            for(int j = 0; j < width; j++) {
                #pragma unroll
                for(int k = 0; k < packed_per_tile; k++) {
                    tiles[i][j].data[k] = value2;
                }
            }
        }
    }
    template<typename U>
    __device__ inline void operator=(const rt<U, rows, cols, layout> &other) {
        using U2 = base_types::packing<U>::packed_type;
        #pragma unroll
        for(int i = 0; i < height; i++) {
            #pragma unroll
            for(int j = 0; j < width; j++) {
                #pragma unroll
                for(int k = 0; k < packed_per_tile; k++) {
                    tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(other.tiles[i][j].data[k]);
                }
            }
        }
    }
};


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// layout and type wrappers

template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fl = rt<float, _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_bf = rt<bf16,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_hf = rt<half,  _r, _c, layout>;
#ifdef KITTENS_HOPPER
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e4m3 = rt<fp8e4m3,  _r, _c, layout>;
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e5m2 = rt<fp8e5m2,  _r, _c, layout>;
#ifdef KITTENS_BLACKWELL
template<int _r, int _c, ducks::rt_layout::all layout=ducks::rt_layout::row> using rt_fp8e8m0 = rt<fp8e8m0,  _r, _c, layout>;
#endif
#endif
} // namespace kittens
