/**
* @file
* @brief Functions for transferring data directly between global and shared memory and back.
*/

#pragma once

#include "../../../../../common/common.metal"
#include "../../../../../types/types.metal"

#include "../global_to_shared.metal"

namespace mittens {
/**
 * @brief Loads data from global memory into a complex shared memory tile with a row layout.
 *
 * @tparam CST The type of the complex shared tile.
 * @param[out] dst The destination complex shared memory tile.
 * @param[in] resrc The source global memory array for the real component.
 * @param[in] imsrc The source global memory array for the imaginary component.
 * @param re_row_stride[in] The stride between rows in the source real component array.
 * @param im_row_stride[in] The stride between rows in the source imaginary component array.
 */
template<typename CST, typename CGL>
METAL_FUNC static typename metal::enable_if<ducks::is_complex_shared_tile<CST>() && ducks::is_global_layout<CGL>(), void>::type
load(threadgroup CST &dst, thread const CGL &src, thread const coord &idx) {
    load(dst.real, src.real, idx);
    load(dst.imag, src.imag, idx);
}

/**
 * @brief Stores bf16 data from a complex shared memory tile with a row layout into global memory.
 *
 * @tparam CST The type of the complex shared tile.
 * @param[out] redst The destination global memory array for the real component.
 * @param[out] imdst The destination global memory array for the imaginary component.
 * @param[in] src The source complex shared memory tile.
 * @param re_row_stride[in] The stride between rows in the destination real component array.
 * @param im_row_stride[in] The stride between rows in the destination imaginary component array.
 */
template<typename CST, typename CGL>
METAL_FUNC static typename metal::enable_if<ducks::is_complex_shared_tile<CST>() && ducks::is_complex_global_layout<CGL>(), void>::type
store(thread const CGL &dst, threadgroup CST &src, thread const coord &idx) {
    store(dst.real, src.real, idx);
    store(dst.imag, src.imag, idx);
}

}
