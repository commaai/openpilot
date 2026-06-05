/**
 * @file
 * @brief Templated layouts for complex global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/cst.cuh"
#include "gl.cuh"
#include "util.cuh"
#ifdef KITTENS_HOPPER
#include "tma.cuh"
#endif

namespace kittens {

/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace cgl {
struct identifier {};
}
}

// namespace detail {
// template<typename T> concept tile = ducks::cst::all<T> || ducks::crt::all<T>;
// template<typename T> concept vec  = ducks::csv::all<T> || ducks::crv::all<T>;
// }

template<kittens::ducks::gl::all _GL>
struct cgl {
    using identifier = ducks::cgl::identifier;
    using component  = _GL;
    using T          = component::T;
    using T2         = component::T2;
    using dtype      = component::dtype;
    component real, imag;
};

namespace ducks {
namespace cgl {
/**
* @brief Concept for all complex global layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::cgl::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::cgl::identifier
}
}

}