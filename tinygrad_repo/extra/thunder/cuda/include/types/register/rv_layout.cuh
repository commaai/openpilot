/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
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
struct align { constexpr static int inner_dim = 2; };
/**
 * @brief A dummy type used to identify an orthogonal (4x replicated) layout.
 */
struct ortho { constexpr static int inner_dim = 1; };
/**
 * @brief A dummy type used to identify an unreplicated layout, for better coalesced loads and vector operations like layernorm.
 */
struct naive { constexpr static int inner_dim = 1; };

/**
 * @brief A concept to check if a type is a register tile layout.
 */
template<typename T>
concept all = std::is_same_v<T, align> || std::is_same_v<T, ortho> || std::is_same_v<T, naive>;

} // namespace rv_layout
} // namespace ducks
} // namespace kittens