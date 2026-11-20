/**
* @file
* @brief Layouts and their manipulations for register tiles.
*/

#pragma once


namespace mittens {
namespace ducks {
/**
* @namespace rv_layout
*
* @brief A namespace for template metaprogramming with register vector layouts.
*/
namespace rv_layout {

/**
 * @brief A dummy type used to identify an aligned (8x replicated) layout.
 */
struct align { constant constexpr static int inner_dim = 2; };
/**
 * @brief A dummy type used to identify an orthogonal (4x replicated) layout.
 */
struct ortho { constant constexpr static int inner_dim = 1; };
/**
 * @brief A dummy type used to identify an unreplicated layout, for better coalesced loads and vector operations like layernorm.
 */
struct naive { constant constexpr static int inner_dim = 1; };

    
} // namespace rv_layout
    
template <typename _layout>
METAL_FUNC static constexpr bool is_align_layout() {
    return metal::is_same_v<_layout, rv_layout::align>;
}
template <typename _layout>
METAL_FUNC static constexpr bool is_ortho_layout() {
    return metal::is_same_v<_layout, rv_layout::ortho>;
}
template <typename _layout>
METAL_FUNC static constexpr bool is_naive_layout() {
    return metal::is_same_v<_layout, rv_layout::naive>;
}
template <typename _layout>
METAL_FUNC static constexpr bool is_rv_layout() {
    return is_align_layout<_layout>() || is_ortho_layout<_layout>() || is_naive_layout<_layout>();
}
    
    
    
} // namespace ducks
} // namespace mittens
