#pragma once
#include "global/global.metal"
#include "register/register.metal"
#include "shared/shared.metal"


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */
namespace mittens {
/**
 * @brief Row vector type alias.
 *
 * This template alias provides a convenient way to refer to the row vector type
 * associated with a given class or type `T`. It assumes that the class `T` has
 * a nested type named `row_vec`.
 *
 * @tparam T The class or type for which the row vector type is defined.
 *
 * Example usage:
 * @code
 * mittens::row_vec<decltype(some_tile)> row_vector;
 * @endcode
 */
template<typename T>
using row_vec = typename T::row_vec;

/**
 * @brief Column vector type alias.
 *
 * This template alias provides a convenient way to refer to the column vector type
 * associated with a given class or type `T`. It assumes that the class `T` has
 * a nested type named `col_vec`.
 *
 * @tparam T The class or type for which the column vector type is defined.
 *
 * Example usage:
 * @code
 * mittens::col_vec<decltype(some_tile)> col_vector;
 * @endcode
 */
template<typename T>
using col_vec = typename T::col_vec;

// register vector layouts
using align_l = ducks::rv_layout::align;
using ortho_l = ducks::rv_layout::ortho;
using naive_l = ducks::rv_layout::naive;

// ^ this code lives here because it applies to both sv and rv types
}
