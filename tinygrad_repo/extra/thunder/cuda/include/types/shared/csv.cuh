/**
 * @file
 * @brief Abstraction for a complex register tile composed of real and imaginary tiles
 */
 
#pragma once

#include "st.cuh"

namespace kittens {

namespace ducks {
namespace csv {
/**
 * @brief A dummy type used to identify complex register tiles.
 * 
 * For a type to quack like an st_cmplx, it should define its identifier as ducks::st::cmplx_identifier.
 * If a type quacks like ducks::st::cmplx_identifier, it will be treated as an st_cmplx by compiler checks.
 */
struct identifier {};
/**
* @brief Concept for shared vectors that are complex.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a shared tile.
* - T has a complex tile identifier.
*/
template <typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier> && ducks::sv::all<typename T::component>;

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
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype; ///< Data type of the elements in the tile.

    static constexpr int length     = component::length;
    static constexpr int tiles      = component::tiles;

    // todo: fill in the rest for convenience, but they're all accessible via component so it's not urgent.

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;
};


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<int _length> using csv_bf = csv<bf16,  _length>;
template<int _length> using csv_hf = csv<half,  _length>;
template<int _length> using csv_fl = csv<float, _length>;

}