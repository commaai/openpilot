/**
 * @file
 * @brief Register vectors for computations on axes.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"
#include "rv_layout.cuh"

namespace kittens {

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
 * @tparam _outer_dim The size of the tile, in units of TILE_DIM.
 * @tparam _inner_dim This controls the layout of the tile in terms of which axis it maps on the register tile layout.
 *
 * Register vectors are used to accumulate and map values across tiles. You can do computation
 * on them directly if you want, but they're not designed to be maximally efficient vectors
 * as they have substantial duplication and strange layouts to help them work efficiently with
 * the register layouts used by the tensor cores. ThunderKittens wants you working with tiles
 * where possible!
 */
template<typename _T, size_t _length, size_t _tile_length, ducks::rt_shape::all _shape, ducks::rv_layout::all _layout=ducks::rv_layout::naive>
struct rv {
    using identifier = ducks::rv::identifier; ///< Type identifier for the rv structure.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using shape = _shape;
    using layout = _layout;
    static constexpr bool is_naive = std::is_same_v<layout, ducks::rv_layout::naive>;
    static constexpr bool is_ortho = std::is_same_v<layout, ducks::rv_layout::ortho>;
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = std::conditional_t<is_naive || is_ortho, T, T2>;
    static constexpr int packing = kittens::base_types::packing<dtype>::num();

    static constexpr int length = _length; ///< Length in elements.
    static_assert(length % _tile_length == 0, "Length must be divisible by the tile dimension");
    static constexpr int tiles  = _length / _tile_length; ///< Length in subtiles, aliased for consistency with sv type
    static constexpr int inner_dim = is_naive ? ((length + kittens::WARP_THREADS - 1) / kittens::WARP_THREADS) : (is_ortho ? 1 : _shape::elements_per_thread / packing);
    static constexpr int outer_dim = is_naive ? 1 : tiles;

    // For align layout
    static constexpr int elements_per_thread = _shape::elements_per_thread;
    static constexpr int reductions = _tile_length;
    static constexpr int threads_per_reduction = reductions / elements_per_thread;
    static constexpr int aligned_threads = kittens::WARP_THREADS / threads_per_reduction;
    static constexpr int stride = _shape::stride;
    static constexpr int packed_per_stride = stride / packing;
    static constexpr int elements_per_stride_group = threads_per_reduction * stride;
    static constexpr int strides_per_tile = reductions / elements_per_stride_group;

    dtype data[outer_dim][inner_dim]; ///< The actual register vector data.

    __device__ inline       dtype* operator[](size_t idx)       { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype* operator[](size_t idx) const { return &data[idx][0]; } ///< A wrapper for indexing into vector data.
    __device__ inline       dtype& operator[](int2 outin)       { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
    __device__ inline const dtype& operator[](int2 outin) const { return data[outin.x][outin.y]; } ///< A wrapper for indexing into vector data.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rv {
/**
* @brief Concept for all register vectors.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rv::identifier.
*/
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rv::identifier.

template<typename T> concept naive_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::naive>;
template<typename T> concept align_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::align>;
template<typename T> concept ortho_layout = all<T> && std::is_same_v<typename T::layout, ducks::rv_layout::ortho>;
template<typename T> concept tile_layout = align_layout<T> || ortho_layout<T>;

} // namespace rv
} // namespace ducks

template<int _l, int _tile_length, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_fl = rv<float, _l, _tile_length, shape, layout>;
template<int _l, int _tile_length, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_bf = rv<bf16,  _l, _tile_length, shape, layout>;
template<int _l, int _tile_length, ducks::rt_shape::all shape=ducks::rt_shape::rt_16x16, ducks::rv_layout::all layout=ducks::rv_layout::naive> using rv_hf = rv<half,  _l, _tile_length, shape, layout>;

template<typename _T, int _l> using rv_naive = rv<_T,  _l, _l, ducks::rt_shape::rt_16x16, ducks::rv_layout::naive>;

} // namespace kittens