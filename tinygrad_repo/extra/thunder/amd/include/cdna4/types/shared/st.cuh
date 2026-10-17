/**
 * @file
 * @brief The ThunderKittens shared tile struct.
 */

#pragma once

#include "../../common/common.cuh"
#include "sv.cuh"
#include "st_shape.cuh"

/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
namespace ducks {
/**
 * @namespace st
 * 
 * @brief The namespace where concepts and abstract types for shared tiles live.
 */
namespace st {
/**
 * @brief A dummy type used to identify shared tiles.
 * 
 * For a type to quack like an st, it should define its identifier as ducks::st::identifier.
 * If a type quacks like ducks::st::identifier, it will be treated as an st by compiler checks.
 * This is particularly useful for subtiles.
 */
struct identifier {};
}
} // namespace ducks

// Forward declaration of subtile
template<
    typename ST,
    int _subtile_height,
    int _subtile_width
>
struct st_subtile;

/**
 * @brief Shared memory tile structure for various data types and layouts.
 *
 * @tparam T The data type of the elements in the tile. Not packed!
 * @tparam _rows The height of the tile.
 * @tparam _cols The width of the tile.
 */
template<typename _T, int _rows, int _cols, ducks::st_shape::all _shape>
struct KITTENS_DEFAULT_ALIGN st {
    using identifier = ducks::st::identifier; ///< Type identifier for shared memory tile.
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.
    using shape = _shape;

    // define underlying data as same as that projected, to make clear that this is *not* a subtile.
    static constexpr int underlying_rows              = _rows;
    static constexpr int underlying_cols              = _cols;
    static constexpr int underlying_num_elements      = underlying_rows * underlying_cols;

    static constexpr int underlying_subtile_rows      = shape::rows;
    static constexpr int underlying_subtile_cols      = shape::cols;
    static constexpr int underlying_subtile_row_bytes = shape::cols * sizeof(T);
    static constexpr int underlying_subtile_elements  = underlying_subtile_rows * underlying_subtile_cols;
    static constexpr int underlying_subtile_bytes     = underlying_subtile_elements * sizeof(T);
    static constexpr int underlying_subtile_bytes_per_thread = shape::template bytes_per_thread<T>();

    static constexpr int underlying_subtiles_per_row  = underlying_cols / underlying_subtile_cols;
    static constexpr int underlying_subtiles_per_col  = underlying_rows / underlying_subtile_rows;

    static constexpr int rows                = _rows; ///< Total number of rows in the tile.
    static constexpr int cols                = _cols; ///< Total number of cols in the tile.
    static constexpr int num_elements        = rows * cols; ///< Total number of elements in the tile.

    static constexpr int subtiles_per_row    = cols / underlying_subtile_cols;
    static constexpr int subtiles_per_col    = rows / underlying_subtile_rows;

    static_assert(base_types::packing<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)

    dtype data[rows*cols]; ///< Raw data storage for the tile.

    __device__ __forceinline__ static const uint32_t swizzle(int2 coord) {
        return shape::template swizzle<T>(coord);
    }

    // vector types
    using col_vec = sv<dtype, rows>; ///< Column vector type for this tile
    using row_vec = sv<dtype, cols>; ///< Row vector type for this tile

    template<int subtile_rows, int subtile_cols> using subtile = st_subtile<st<_T, _rows, _cols, _shape>, subtile_rows, subtile_cols>;
};


/**
 * @brief A reference into a chunk of shared tile memory.
 *
 * The st_subtile is a drop-in replacement for an st which internally
 * references the appropriate memory while performing minimal address
 * calculations. You should never create this directly, but instead
 * have subtile_inplace return it for you instead. (`auto` is nice.)
 *
 * You can generally just pretend this is an st. But not for wgmma's.
 */
template<
    typename _ST,
    int _subtile_rows,
    int _subtile_cols
>
struct st_subtile {
    using identifier = ducks::st::identifier; // i quack like an st, gcc will never know the difference
    using ST = _ST;
    using T = ST::T;
    using T2 = ST::T2;
    using dtype = T; ///< Data type of the elements in the tile.
    using shape = ST::shape;

    static constexpr int underlying_rows              = ST::underlying_rows;
    static constexpr int underlying_cols              = ST::underlying_cols;
    static constexpr int underlying_num_elements      = ST::underlying_num_elements;

    static constexpr int underlying_subtile_cols      = ST::underlying_subtile_cols;
    static constexpr int underlying_subtile_row_bytes = ST::underlying_subtile_row_bytes;
    static constexpr int underlying_subtile_rows      = ST::underlying_subtile_rows;
    static constexpr int underlying_subtile_elements  = ST::underlying_subtile_elements;
    static constexpr int underlying_subtile_bytes     = ST::underlying_subtile_bytes;
    static constexpr int underlying_subtile_bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    
    static constexpr int underlying_subtiles_per_row  = ST::underlying_subtiles_per_row;
    static constexpr int underlying_subtiles_per_col  = ST::underlying_subtiles_per_col;

    static constexpr int rows                = _subtile_rows;
    static constexpr int cols                = _subtile_cols;
    static constexpr int num_elements        = rows * cols;

    static constexpr int subtiles_per_row    = cols / underlying_subtile_cols;
    static constexpr int subtiles_per_col    = rows / underlying_subtile_rows;

    dtype *data;
    int row_offset, col_offset;

    __device__ st_subtile(ST &src, int2 rowcol) {
        row_offset = rowcol.x * rows;
        col_offset = rowcol.y * cols;
        const int subtile_row_offset = row_offset / underlying_subtile_rows;
        const int subtile_col_offset = col_offset / underlying_subtile_cols;
        const int subtile_id = subtile_row_offset * underlying_subtiles_per_row + subtile_col_offset;
        const int subtile_offset = subtile_id * underlying_subtile_elements;
        data = &src.data[subtile_offset];
    }

    __device__ __forceinline__ static const uint32_t swizzle(int2 coord) {
        return ST::swizzle(coord);
    }

    // vector types
    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace st {

/**
* @brief Concept for all shared tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as st::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::st::identifier

} // namespace st
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _height, int _width, ducks::st_shape::all _shape> using st_bf = st<bf16,  _height, _width, _shape>;
template<int _height, int _width, ducks::st_shape::all _shape> using st_hf = st<half,  _height, _width, _shape>;
template<int _height, int _width, ducks::st_shape::all _shape> using st_fl = st<float, _height, _width, _shape>;
template<int _height, int _width, ducks::st_shape::all _shape> using st_fp8e4m3 = st<fp8e4m3, _height, _width, _shape>;
}
