/**
 * @file
 * @brief An aggregate header file for all the register and shared types defined by ThunderKittens.
 */

#pragma once

#include "device/device.cuh"
#include "register/register.cuh"
#include "shared/shared.cuh"
#include "global/global.cuh"
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
#include "device/device.cuh"
#endif
#ifdef KITTENS_BLACKWELL
#include "tensor/tensor.cuh"
#endif

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

}
