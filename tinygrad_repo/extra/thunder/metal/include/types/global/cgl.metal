/**
* @file
* @brief Templated layouts for complex global memory.
*/

#pragma once

#include "../../common/common.metal"
//#include "../shared/cst.metal"
#include "gl.metal"
#include "util.metal"
#ifdef mittens_HOPPER
#include "tma.metal"
#endif

namespace mittens {
/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace cgl {
struct identifier {};
}
}

template<typename GL>
struct cgl {
    static_assert(ducks::is_global_layout<GL>, "GL must satisfy global layout requirements.");

    using identifier = ducks::cgl::identifier;
    using T = typename GL::T;
    using T2 = typename GL::T2;
    using dtype = typename GL::dtype;

    GL real, imag;
};

namespace ducks {
template <typename T>
struct has_cgl_identifier {
    static constant constexpr bool value = false; // Default case
};

//template <typename _T, int b, int d, int r, int c, typename... TMA_Types>
//struct has_cgl_identifier<mittens::gl<_T, b, d, r, c, TMA_Types ...>> {
//    static constant constexpr bool value = true;
//};
template <typename _T, int b, int d, int r, int c>
struct has_cgl_identifier<mittens::gl<_T, b, d, r, c>> {
    static constant constexpr bool value = true;
};

template <typename GL>
static constexpr bool is_complex_global_layout() {
    return has_rt_identifier<GL>::value;
}
template <typename GL>
static constexpr void assert_cgl() {
    static_assert(is_complex_global_layout<GL>(), "T must be a cgl");
}
}

}

