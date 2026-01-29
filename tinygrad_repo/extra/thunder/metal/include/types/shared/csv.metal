/**
* @file
* @brief Abstraction for a complex register tile composed of real and imaginary tiles
*/

#pragma once

#include "st.metal"

namespace mittens {
namespace ducks {
namespace csv {
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
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the internal register tiles
 *
 * This structure is designed to abstract complex number operations internally to the real and imaginary
 * shared tiles, respectively
 *
 *
 */
template<typename _T, int _length>
struct csv {
    using identifier = ducks::csv::identifier;
    using component  = sv<_T, _length>; /// Data type of each internal tile.
    using T          = typename component::T;
    using T2         = typename component::T2;
    using dtype      = typename component::dtype; ///< Data type of the elements in the tile.
    
    constant static constexpr int length     = component::length;
    constant static constexpr int tiles      = component::tiles;
    
    // todo: fill in the rest for convenience, but they're all accessible via component so it's not urgent.
    
    // Real/imag tiles have same internal layout and size
    component real;
    component imag;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
template <typename T>
struct has_csv_identifier {
    static constant constexpr bool value = false; // Default case
};
 
// Specialize for specific template instantiations of st
template <typename _T, int _length>
struct has_csv_identifier<mittens::csv<_T, _length>> {
    static constant constexpr bool value = true;
};
    
template <typename CSV>
static constexpr bool is_complex_shared_vector() {
    return has_csv_identifier<CSV>::value;
}
template <typename CSV>
static constexpr void assert_complex_shared_vector() {
    static_assert(is_complex_shared_vector<CSV>(), "T must be a csv");
}
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _length> using csv_bf = csv<bf16,  _length>;
template<int _length> using csv_hf = csv<half,  _length>;
template<int _length> using csv_fl = csv<float, _length>;

}

