/**
 * @file
 * @brief The main Thundermittens register tile struct, where most computation happens.
 */
#pragma once // kinda done
/*
 TODO:
  consider if column layout rly rly rly makes no sense and no implement needed, not me being lazy
 */
#include <metal_stdlib>
#include "../../common/common.metal"
#include "rt_base.metal"
#include "rv.metal"

/* ----------  MAIN TILE STRUCT  ---------- */


namespace mittens {
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
    
} // namespace rt
    
} // namespace ducks
    
/**
 * @brief Main tile structure for manipulating data in registers.
 *
 * @tparam _T The data type used for the matrix elements.
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 *
 * This structure is designed to handle matrix tiles in a flexible manner, allowing
 * for operations on tiles that are composed of smaller subtiles.
 */
template<typename _T,  int _rows, int _cols, typename _layout=ducks::rt_layout::row>
struct rt {
    using identifier = ducks::rt::identifier; ///< Type identifier for the rt structure.
    using layout = _layout;
    using T  = typename base_types::packing<_T>::unpacked_type;
    static_assert(ducks::base_types::isT1Type<T>(), "T must be float, bf16, or half");
    static_assert(ducks::is_rt_layout<_layout>(), "T must be float, bf16, or half");
    using T2 = typename base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.
    constant static constexpr int rows                = _rows; ///< Total number of rows.
    static_assert(rows % rt_base<T, _layout>::tile_size == 0, "Rows must be divisible by the tile size");
    constant static constexpr int cols                = _cols; ///< Total number of columns.
    static_assert(cols % rt_base<T, _layout>::tile_size == 0, "Columns must be divisible by the tile size");
    constant static constexpr int height              = rows / rt_base<T, _layout>::tile_size; ///< Height in subtiles.
    constant static constexpr int width               = cols / rt_base<T, _layout>::tile_size; ///< Width in subtiles.
    constant static constexpr int tile_size           = rt_base<T, _layout>::tile_size; ///< Size of the base tile.
    constant static constexpr int num_elements        = rt_base<T, _layout>::num_elements        * width * height; ///< Total number of elements.
    constant static constexpr int elements_per_thread = rt_base<T, _layout>::elements_per_thread * width * height; ///< Elements handled per thread.
    constant static constexpr int packed_per_thread   = rt_base<T, _layout>::packed_per_thread   * width * height; ///< Packed elements per thread.
    constant static constexpr int packed_per_tile     = rt_base<T, _layout>::packed_per_thread; ///< Packed elements per tile.
    
    rt_base<dtype, _layout> tiles[height][width]; ///< The actual storage for the matrix tile, organized in subtiles.
    
    using row_vec = rv<T, cols, typename rt_base<T, _layout>::row_vec_layout>; ///< A type representing a column vector for this tile.
    using col_vec = rv<T, rows, typename rt_base<T, _layout>::col_vec_layout>; ///< A type representing a column vector for this tile.
};

 
    
namespace ducks{
template <typename T>
struct has_rt_identifier {
    static constant constexpr bool value = false; // Default case
    static constant constexpr bool is_row = false;
    static constant constexpr bool is_col = false;
};

template <typename _T, int _rows, int _cols>
struct has_rt_identifier<mittens::rt<_T, _rows, _cols, rt_layout::row>> {
    static constant constexpr bool value = true;
    static constant constexpr bool is_row = true;  // Row-specific indicator
    static constant constexpr bool is_col = false;
};

template <typename _T, int _rows, int _cols>
struct has_rt_identifier<mittens::rt<_T, _rows, _cols, rt_layout::col>> {
    static constant constexpr bool value = true;
    static constant constexpr bool is_row = false;
    static constant constexpr bool is_col = true;  // Col-specific indicator
};

template <typename RT>
static constexpr bool is_register_tile() {
    return has_rt_identifier<RT>::value;
}
    
template <typename RT>
static constexpr bool is_row_register_tile() {
    return has_rt_identifier<RT>::is_row;
}

template <typename RT>
static constexpr bool is_col_register_tile() {
    return has_rt_identifier<RT>::is_col;
}
    
    
template <typename RT>
static constexpr void assert_register_tile() {
    static_assert(is_register_tile<RT>(), "T must be a rt");
}
}
   
/* ----------  WRAPPERS FOR PRETTINESS  ---------- */
// layout and type wrappers

template<int _r, int _c, typename layout=ducks::rt_layout::row> using rt_fl = rt<float, _r, _c, layout>;
template<int _r, int _c, typename layout=ducks::rt_layout::row> using rt_bf = rt<bf16,  _r, _c, layout>;
template<int _r, int _c, typename layout=ducks::rt_layout::row> using rt_hf = rt<half,  _r, _c, layout>;
} // namespace mittens
