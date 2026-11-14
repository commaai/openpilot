/**
* @file
* @brief Layouts and their manipulations for register tiles.
*/

#pragma once


namespace mittens {
namespace ducks {
/**
 * @namespace rt_layout
 *
 * @brief A namespace for template metaprogramming with register tile layouts.
 */
namespace rt_layout {
    
/**
 * @brief A dummy type used to identify a row-major layout for a register tile.
 */
struct row {}; // for most matrices
/**
 * @brief A dummy type used to identify a col-major layout for a register tile.
 */
struct col {}; // for the B-matrix of MMA ops.

template<typename l> struct transpose      { using type = rt_layout::col; };
template<>           struct transpose<rt_layout::col> { using type = rt_layout::row; };
} // namespace rt_layout
template <typename _layout>
METAL_FUNC static constexpr bool is_row_layout() {
    return metal::is_same_v<_layout, rt_layout::row>;
}
template <typename _layout>
METAL_FUNC static constexpr bool is_col_layout() {
    return metal::is_same_v<_layout, rt_layout::col>;
}
template <typename _layout>
METAL_FUNC static constexpr bool is_rt_layout() {
    return is_row_layout<_layout>() || is_col_layout<_layout>();
}

    
} // namespace ducks
} // namespace mittens
