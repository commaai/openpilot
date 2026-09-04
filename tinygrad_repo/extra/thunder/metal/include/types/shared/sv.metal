/**
 * @file
 * @brief The Thundermittens shared vector struct.
 */

#pragma once
#include "../../common/common.metal"
#include <metal_stdlib>
namespace mittens {
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
 * @tparam _tiles The size of the tile, in units of TILE_DIM (16).
 *
 * Shared vectors are used to accumulate and map values across shared tiles.
 * Unlike every other structure present in Thundermittens, these have a simple
 * uniform layout which is just an array in memory. EZ!
 */
template<typename _T, size_t _length>
struct mittens_DEFAULT_ALIGN sv {
    using identifier = ducks::sv::identifier;
    using T  = typename base_types::packing<_T>::unpacked_type;
    using T2 = typename base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    constant static constexpr int length = _length; ///< Length in elements.
    static_assert(length % TILE_DIM == 0, "Length must be divisible by the tile dimension");
    constant static constexpr int tiles  = length / TILE_DIM; ///< Length in subtiles.

    dtype data[length]; ///< The actual shared vector data.

    METAL_FUNC threadgroup dtype& operator[](size_t idx) threadgroup { return data[idx]; }
    METAL_FUNC const threadgroup dtype& operator[](size_t idx) const threadgroup { return data[idx]; }

    template<size_t _len> using subvec = sv<dtype, _len>;
};

    
namespace ducks {
template <typename T>
struct has_sv_identifier {
    static constant constexpr bool value = false; // Default case
};

// Specialize for specific template instantiations of st
template<typename _T, size_t _length>
struct has_sv_identifier<mittens::sv<_T, _length>> {
    static constant constexpr bool value = true;
};
    
template <typename ST>
static constexpr bool is_shared_vector() {
    return has_sv_identifier<ST>::value;
}
template <typename ST>
static constexpr void assert_shared_vector() {
    static_assert(is_shared_vector<ST>(), "T must be a sv");
}
}
        
    
template<size_t _tiles> using sv_bf = sv<bfloat, _tiles>;
template<size_t _tiles> using sv_hf = sv<half  , _tiles>;
template<size_t _tiles> using sv_fl = sv<float , _tiles>;
}


