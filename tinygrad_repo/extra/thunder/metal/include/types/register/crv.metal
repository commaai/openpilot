/**
* @file
* @brief Register vectors for computations on axes.
*/

#pragma once

#include "../../common/common.metal"
#include "rv_layout.metal"
#include "rv.metal"

namespace mittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/**
* @namespace rt
*
* @brief The namespace where concepts and abstract types for register vectors live.
*/
namespace crv {
/**
 * @brief A dummy type used to identify register vectors.
 *
 * For a type to quack like an rv, it should define its identifier as ducks::rv::identifier.
 * If a type quacks like ducks::rv::identifier, it will be treated as an rv by compiler checks.
 */
struct identifier {};
}
}
/**
* @brief Register vector structure.
*
* @tparam _T The packed data type used for the vector elements.
* @tparam _outer_dim The size of the tile, in units of TILE_DIM (16).
* @tparam _inner_dim This controls the layout of the tile in terms of which axis it maps on the register tile layout.
*
* Register vectors are used to accumulate and map values across tiles. You can do computation
* on them directly if you want, but they're not designed to be maximally efficient vectors
* as they have substantial duplication and strange layouts to help them work efficiently with
* the register layouts used by the tensor cores. Thundermittens wants you working with tiles
* where possible!
*/

template<typename _T, size_t _length, typename _layout=ducks::rv_layout::naive>
struct crv {
    static_assert(ducks::is_rv_layout<_layout>(), "_layout must be a rv layout");
    static_assert(ducks::base_types::isT1Type<_T>(), "T must be float, bf16, or half");
    using identifier = ducks::crv::identifier;
    using component  = rv<_T, _length, _layout>; /// Data type of each internal tile.
    using layout     = typename component::layout; ///< Layout of the matrix tile, ensures compatibility with the rv concepts

    using T          = typename component::T;
    using T2         = typename component::T2;
    using dtype      = typename component::dtype; ///< Data type of the elements in the tile.

    constant static constexpr int length     = component::length;
    constant static constexpr int tiles      = component::tiles;

    // Real/imag tiles have same internal layout and size
    component real;
    component imag;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
template <typename T>
struct has_crv_identifier {
    static constant constexpr bool value = false; // Default case
};

// Specialize for specific template instantiations of st
template <typename _T, int _length, typename _layout>
struct has_crv_identifier<mittens::crv<_T, _length, _layout>> {
    static constant constexpr bool value = true;
};

template <typename CRV>
static constexpr bool is_complex_register_vector() {
    return has_crv_identifier<CRV>::value;
}
template <typename CRV>
static constexpr void assert_complex_register_vector() {
    static_assert(is_complex_register_vector<CRV>(), "T must be a crv");
}
} // namespace ducks

template<int _l, typename layout=ducks::rv_layout::naive> using crv_fl = crv<float, _l, layout>;
template<int _l, typename layout=ducks::rv_layout::naive> using crv_bf = crv<bf16,  _l, layout>;
template<int _l, typename layout=ducks::rv_layout::naive> using crv_hf = crv<half,  _l, layout>;


} // namespace mittens

