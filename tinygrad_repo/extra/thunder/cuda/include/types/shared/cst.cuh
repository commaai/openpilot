/**
 * @file
 * @brief Abstraction for a complex register tile composed of real and imaginary tiles
 */
 
#pragma once

#include "st.cuh"

namespace kittens {

namespace ducks {
namespace cst {
/**
 * @brief A dummy type used to identify complex register tiles.
 * 
 * For a type to quack like an st_cmplx, it should define its identifier as ducks::st::cmplx_identifier.
 * If a type quacks like ducks::st::cmplx_identifier, it will be treated as an st_cmplx by compiler checks.
 */
struct identifier {};

/**
* @brief Concept for shared tiles that are complex.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a shared tile.
* - T has a complex tile identifier.
*/
template <typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier> && ducks::st::all<typename T::component>;

} // namespace st
} // namespace ducks

/**
 * @brief Complex tile structure
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _rows The height of the tile in terms of the number of subtiles.
 * @tparam _cols The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the internal register tiles
 *
 * This structure is designed to abstract complex number operations internally to the real and imaginary
 * shared tiles, respectively
 * 
 *
 */
template<typename _T, int _rows, int _cols>
struct cst {
    using identifier = ducks::cst::identifier;
    using component  = st<_T, _rows, _cols>; /// Data type of each internal tile.
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype; ///< Data type of the elements in the tile.

    static constexpr int rows       = component::rows;
    static constexpr int cols       = component::cols;
    static constexpr int height     = component::height;
    static constexpr int width      = component::width;

    // todo: fill in the rest for convenience, but they're all accessible via component so it's not urgent.

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;

    // vector types
    using col_vec = csv<dtype, rows>;
    using row_vec = csv<dtype, cols>;
};

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _rows, int _cols> using cst_bf = cst<bf16,  _rows, _cols>;
template<int _rows, int _cols> using cst_hf = cst<half,  _rows, _cols>;
template<int _rows, int _cols> using cst_fl = cst<float, _rows, _cols>;



}