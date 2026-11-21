/**
 * @file
 * @brief Register vectors for computations on axes.
 */
#pragma once
#include "../../common/common.metal"
#include "rv_layout.metal"
namespace mittens {
/* ----------  MAIN VECTOR STRUCT  ---------- */

// helper struct for type inference
namespace ducks {
/** 
 * @namespace rt
 *
 * @brief The namespace where concepts and abstract types for register vectors live.
 */
namespace rv {
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
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM (8).
 * @tparam _inner_dim This controls the layout of the tile in terms of which axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can do computation
 * on them directly if you want, but they're not designed to be maximally efficient vectors
 * as they have substantial duplication and strange layouts to help them work efficiently with
 * the register layouts used by the tensor cores. Thundermittens wants you working with tiles
 * where possible!
 */
    
template<typename _T, size_t _length, typename _layout>
struct rv {
    using identifier = ducks::rv::identifier; ///< Type identifier for the rv structure.
    
    static_assert(ducks::is_rv_layout<_layout>(), "_layout must be a rv layout");
    static_assert(ducks::base_types::isT1Type<_T>(), "T must be float, bf16, or half");
    using layout = _layout;
    constant static constexpr bool is_naive = ducks::is_naive_layout<layout>();
    using T = typename mittens::base_types::packing<_T>::unpacked_type;
    using T2 =typename mittens::base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the matrix elements

    constant static constexpr int length = _length; ///< Length in elements.
    static_assert(length % mittens::TILE_DIM == 0, "Length must be divisible by the tile dimension");
    constant static constexpr int tiles  = _length / mittens::TILE_DIM; ///< Length in subtiles, aliased for consistency with sv type
    constant static constexpr int inner_dim = layout::inner_dim; ///< Internal layout within a subtile. Either 1 or 2.
    constant static constexpr int outer_dim = is_naive ? (tiles+3)/4 : tiles; ///< Outer dim (also length in tiles)
    dtype data[outer_dim][inner_dim]; ///< The actual register vector data.

    METAL_FUNC thread       dtype* operator[](size_t idx)       { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    METAL_FUNC thread const dtype* operator[](size_t idx) const { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    METAL_FUNC thread       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
    METAL_FUNC thread const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
};

namespace ducks{
template <typename T>
struct has_rv_align_identifier {
    static constant constexpr bool value = false; // Default case
};
template <typename _T, int _length>
struct has_rv_align_identifier<mittens::rv<_T, _length, ducks::rv_layout::align>> {
    static constant constexpr bool value = true;
};
template <typename RT>
static constexpr bool is_align_register_vector() {
    return has_rv_align_identifier<RT>::value;
}
    
template <typename T>
struct has_rv_ortho_identifier {
    static constant constexpr bool value = false; // Default case
};
template <typename _T, int _length>
struct has_rv_ortho_identifier<mittens::rv<_T, _length, ducks::rv_layout::ortho>> {
    static constant constexpr bool value = true;
};

template <typename RT>
static constexpr bool is_ortho_register_vector() {
    return has_rv_ortho_identifier<RT>::value;
}
    
template <typename T>
struct has_rv_naive_identifier {
    static constant constexpr bool value = false; // Default case
};
template <typename _T, int _length>
struct has_rv_naive_identifier<mittens::rv<_T, _length, ducks::rv_layout::naive>> {
    static constant constexpr bool value = true;
};
template <typename RT>
static constexpr bool is_naive_register_vector() {
    return has_rv_naive_identifier<RT>::value;
}

template <typename RT>
static constexpr bool is_register_vector() {
    return is_align_register_vector<RT>() || is_ortho_register_vector<RT>() || is_naive_register_vector<RT>();
}

template <typename RT>
static constexpr void assert_register_vector() {
    static_assert(is_register_vector<RT>(), "T must be a rv");
}
}
template<int _l, typename layout=ducks::rv_layout::naive> using rv_fl = rv<float, _l, layout>;
template<int _l, typename layout=ducks::rv_layout::naive> using rv_bf = rv<bf16,  _l, layout>;
template<int _l, typename layout=ducks::rv_layout::naive> using rv_hf = rv<half,  _l, layout>;

}
 
