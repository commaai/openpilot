/**
 * @file
 * @brief An aggregate header file for all the register and shared types defined by ThunderKittens.
 */

#pragma once

#include "register/register.cuh"
#include "shared/shared.cuh"
#include "global/global.cuh"

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

namespace kittens {

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
 * kittens::row_vec<decltype(some_tile)> row_vector;
 * @endcode
 */
template<typename T>
using row_vec = T::row_vec;

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
 * kittens::col_vec<decltype(some_tile)> col_vector;
 * @endcode
 */
template<typename T>
using col_vec = T::col_vec;

// ^ this code lives here because it applies to both sv and rv types

// register tile layouts
using row_l = ducks::rt_layout::row;
using col_l = ducks::rt_layout::col;

// register vector layouts
using align_l = ducks::rv_layout::align;
using ortho_l = ducks::rv_layout::ortho;
using naive_l = ducks::rv_layout::naive;

// register tile shapes
using rt_16x16_s = ducks::rt_shape::rt_16x16;
using rt_32x32_s = ducks::rt_shape::rt_32x32;
using rt_32x32_8_s = ducks::rt_shape::rt_32x32_8;
using rt_16x32_s = ducks::rt_shape::rt_16x32;
using rt_32x16_s = ducks::rt_shape::rt_32x16;
using rt_32x16_4_s = ducks::rt_shape::rt_32x16_4;
using rt_16x32_4_s = ducks::rt_shape::rt_16x32_4;
using rt_16x128_s = ducks::rt_shape::rt_16x128;

// shared tile shapes
using st_16x16_s = ducks::st_shape::st_16x16;
using st_16x16_swizzled_s = ducks::st_shape::st_16x16_swizzled;
using st_32x32_s = ducks::st_shape::st_32x32;
using st_16x32_s = ducks::st_shape::st_16x32;
using st_32x16_s = ducks::st_shape::st_32x16;
using st_8x32_s = ducks::st_shape::st_8x32;
using st_16x128_s = ducks::st_shape::st_16x128;

}
