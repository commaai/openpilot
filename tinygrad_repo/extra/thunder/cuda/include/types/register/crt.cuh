/**
 * @file
 * @brief Abstraction for a complex register tile composed of real and imaginary tiles
 */
 
#pragma once

#include "rt.cuh"
#include "crv.cuh"

namespace kittens {

namespace ducks {
namespace crt {
/**
 * @brief A dummy type used to identify complex register tiles.
 * 
 * For a type to quack like an rt_cmplx, it should define its identifier as ducks::rt::cmplx_identifier.
 * If a type quacks like ducks::rt::cmplx_identifier, it will be treated as an rt_cmplx by compiler checks.
 */
struct identifier {};
/**
* @brief Concept for register tiles that are complex.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a register tile.
* - T has a complex tile identifier.
*/
template <typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier> && ducks::rt::all<typename T::component>;

/*
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
template<typename _T, int _rows, int _cols, ducks::rt_layout::all _layout=ducks::rt_layout::row>
struct crt {
    using identifier = ducks::crt::identifier;
    using component  = rt<_T, _rows, _cols, _layout>; /// Data type of each internal tile.
    using layout     = component::layout; ///< Layout of the matrix tile, ensures compatibility with the rt concepts
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype; ///< Data type of the elements in the tile.

    static constexpr int rows       = component::rows;
    static constexpr int cols       = component::cols;
    static constexpr int height     = component::height;
    static constexpr int width      = component::width;

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;

    using row_vec = crv<T, cols, typename rt_base<T, layout>::row_vec_layout>; ///< A type representing a column vector for this tile.
    using col_vec = crv<T, rows, typename rt_base<T, layout>::col_vec_layout>; ///< A type representing a column vector for this tile.
};

template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_fl = crt<float, _rows, _cols, layout>;
template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_bf = crt<bf16, _rows, _cols, layout>;
template<int _rows, int _cols, ducks::rt_layout::all layout=ducks::rt_layout::row> using crt_hf = crt<half, _rows, _cols, layout>;



}