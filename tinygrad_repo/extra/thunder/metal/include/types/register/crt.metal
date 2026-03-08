/**
* @file
* @brief Abstraction for a complex register tile composed of real and imaginary tiles
*/

#pragma once

#include "rt.metal"
#include "crv.metal"

namespace mittens {

namespace ducks {
namespace crt {
/**
 * @brief A dummy type used to identify complex register tiles.
 *
 * For a type to quack like an rt_cmplx, it should define its identifier as ducks::rt::cmplx_identifier.
 * If a type quacks like ducks::rt::cmplx_identifier, it will be treated as an rt_cmplx by compiler checks.
 */
struct identifier {};
} // namespace rt
} // namespace ducks

/**
* @brief Complex tile structure
*
* @tparam T2 The packed data type used for the matrix elements.
* @tparam _rows The height of the tile in terms of the number of subtiles.
* @tparam _cols The width of the tile in terms of the number of subtiles.
* @tparam _layout The layout of the internal register tiles, either row-major or column-major.
*
* This structure is designed to abstract complex number operations internally to the real and imaginary
* register tiles, respectively
*
* In general, you probably want a row-major tile, unless you specifically want to call mma
*/
template<typename _T, int _rows, int _cols, typename _layout>
struct crt {
    using identifier = ducks::crt::identifier;
    static_assert(ducks::is_rt_layout<_layout>(), "crt was given invalid layout");
    using component  = rt<_T, _rows, _cols, _layout>; /// Data type of each internal tile.
    using layout     = typename component::layout; ///< Layout of the matrix tile, ensures compatibility with the rt concepts
    using T          = typename component::T;
    using T2         = typename component::T2;
    using dtype      = typename component::dtype; ///< Data type of the elements in the tile.

    constant static constexpr int rows       = component::rows;
    constant static constexpr int cols       = component::cols;
    constant static constexpr int height     = component::height;
    constant static constexpr int width      = component::width;

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;

    using row_vec = crv<T, cols, typename rt_base<T, layout>::row_vec_layout>; ///< A type representing a column vector for this tile.
    using col_vec = crv<T, rows, typename rt_base<T, layout>::col_vec_layout>; ///< A type representing a column vector for this tile.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
template <typename T>
struct has_crt_identifier {
    static constant constexpr bool value = false; // Default case
};

// Specialize for specific template instantiations of st
template <typename _T, int _rows, int _cols, typename _layout>
struct has_crt_identifier<mittens::crt<_T, _rows, _cols, _layout>> {
    static constant constexpr bool value = true;
};

template <typename CRT>
static constexpr bool is_complex_register_tile() {
    return has_crt_identifier<CRT>::value;
}
template <typename CRT>
static constexpr void assert_complex_register_tile() {
    static_assert(is_register_tile<CRT>(), "T must be a rt");
}
}

template<int _rows, int _cols, typename _layout=ducks::rt_layout::row> using crt_fl = crt<float, _rows, _cols, _layout>;
template<int _rows, int _cols, typename _layout=ducks::rt_layout::row> using crt_bf = crt<bf16, _rows, _cols, _layout>;
template<int _rows, int _cols, typename _layout=ducks::rt_layout::row> using crt_hf = crt<half, _rows, _cols, _layout>;


}

