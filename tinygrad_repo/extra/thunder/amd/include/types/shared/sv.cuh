/**
 * @file
 * @brief The ThunderKittens shared vector struct.
 */

#pragma once

#include <concepts>
#include <type_traits>

#include "../../common/common.cuh"

namespace kittens {

/* ----------  MAIN VECTOR STRUCT  ---------- */

namespace ducks {
/**
 * @namespace sv
 * 
 * @brief The namespace where concepts and abstract types for shared vectors live.
 */
namespace sv {
/**
 * @brief A dummy type used to identify shared vectors.
 * 
 * For a type to quack like an sv, it should define its identifier as ducks::sv::identifier.
 * If a type quacks like ducks::sv::identifier, it will be treated as an sv by compiler checks.
 */
struct identifier {};
}
}

/**
 * @brief Shared vector structure.
 *
 * @tparam _T The packed data type used for the vector elements.
 * @tparam _tiles The size of the tile, in units of TILE_ROW_DIM (16 for fp16, bf16, fp32).
 *
 * Shared vectors are used to accumulate and map values across shared tiles.
 * Unlike every other structure present in ThunderKittens, these have a simple
 * uniform layout which is just an array in memory. EZ!
 */
template<typename _T, size_t _length>
struct KITTENS_DEFAULT_ALIGN sv {
    using identifier = ducks::sv::identifier;
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    static constexpr int length = _length; ///< Length in elements.

    static constexpr int num_alloc_elements = length;

    dtype data[num_alloc_elements]; ///< The actual shared vector data.

    __device__ static inline T* idx(T *ptr, int idx) { // useful for computations in shared address space, as silly as it sounds.
        return ptr[idx];
    }

    __device__ inline       dtype& operator[](size_t idx)       { return data[idx]; }
    __device__ inline const dtype& operator[](size_t idx) const { return data[idx]; }

    template<size_t sub_length> using subvec = sv<dtype, sub_length>; ///< A subvector which allows warpgroups and blocks to work cooperatively.
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace sv {
/**
* @brief Concept for all shared vectors.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as sv::identifier.
*/
template<typename T>
concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::sv::identifier

} // namespace sv
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// vector types
template<size_t _length> using sv_bf = sv<bf16,  _length>;
template<size_t _length> using sv_hf = sv<half,  _length>;
template<size_t _length> using sv_fl = sv<float, _length>;

} // namespace kittens