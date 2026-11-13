/**
 * @file
 * @brief General  utilities not specialized for either tiles or vectors.
 */
#pragma once // done!
#include "../tile/tile.metal"
#include "../../../../types/shared/shared.metal"
namespace mittens {
    
// sizeof() can be unreliable when working with references to objects
// plus, template magic allows arrays of these objects to be copied, too.
namespace detail {

template <typename T, uint32_t... dims>
struct size_info;
    
template <typename T>
struct size_info<T> {
private:
    static_assert(ducks::is_shared_tile<T>() || ducks::is_shared_vector<T>(), "T must be a shared tile or shared vector");
    constant static constexpr uint32_t elements = ducks::is_shared_tile<T>() ? T::num_elements : T::length;
    constant static constexpr uint32_t bytes = elements * sizeof(typename T::dtype);
};

template <typename T, uint32_t dim, uint32_t... rest_dims>
struct size_info<T, dim, rest_dims...> {
    constant static constexpr uint32_t elements = dim * size_info<T, rest_dims...>::elements;
    constant static constexpr uint32_t bytes = dim * size_info<T, rest_dims...>::bytes;
};
}

template<typename T, uint32_t... dims> constant constexpr uint32_t size_elements = detail::size_info<T, dims...>::elements;
template<typename T, uint32_t... dims> constant constexpr uint32_t size_bytes    = detail::size_info<T, dims...>::bytes;

    
        
}
