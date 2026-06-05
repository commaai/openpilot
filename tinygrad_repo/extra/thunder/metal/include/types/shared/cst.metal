/**
* @file
* @brief Abstraction for a complex register tile composed of real and imaginary tiles
*/

#pragma once

#include "st.metal"
#include "csv.metal"
namespace mittens {
namespace ducks {
namespace cst {
/**
 * @brief A dummy type used to identify complex register tiles.
 *
 * For a type to quack like an st_cmplx, it should define its identifier as ducks::st::cmplx_identifier.
 * If a type quacks like ducks::st::cmplx_identifier, it will be treated as an st_cmplx by compiler checks.
 */
struct identifier {};
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
    using T          = typename component::T;
    using T2         = typename component::T2;
    using dtype      = typename component::dtype; ///< Data type of the elements in the tile.
    
    constant static constexpr int rows       = component::rows;
    constant static constexpr int cols       = component::cols;
    constant static constexpr int height     = component::height;
    constant static constexpr int width      = component::width;
    
    // todo: fill in the rest for convenience, but they're all accessible via component so it's not urgent.
    
    // Real/imag tiles have same internal layout and size
    component real;
    component imag;
    
    // vector types
    using col_vec = csv<dtype, rows>;
    using row_vec = csv<dtype, cols>;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
template <typename T>
struct has_cst_identifier {
    static constant constexpr bool value = false; // Default case
};
 
// Specialize for specific template instantiations of st
template <typename _T, int _height, int _width>
struct has_cst_identifier<mittens::cst<_T, _height, _width>> {
    static constant constexpr bool value = true;
};
    
template <typename CST>
static constexpr bool is_complex_shared_tile() {
    return has_cst_identifier<CST>::value;
}
template <typename CST>
static constexpr void assert_complex_shared_tile() {
    static_assert(is_complex_shared_tile<CST>(), "T must be a cst");
}

} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _rows, int _cols> using cst_bf = cst<bf16,  _rows, _cols>;
template<int _rows, int _cols> using cst_hf = cst<half,  _rows, _cols>;
template<int _rows, int _cols> using cst_fl = cst<float, _rows, _cols>;



}
