/**
* @file
* @brief Functions for transferring data directly between global memory and registers and back.
*/

#pragma once

#include "../../../../../common/common.metal"
#include "../../../../../types/types.metal"

#include "../global_to_register.metal"

namespace mittens {
/**
 * @brief Load data from source arrays into a complex-type tile.
 *
 * @tparam CRT The complex tile type.
 * @tparam U The data type of the source arrays.
 * @param dst[out] The destination tile to load data into.
 * @param resrc[in] The source array to load the real component data from.
 * @param imsrc[in] The source array to load the imaginary component data from.
 * @param re_row_stride[in] The stride in elements between rows in the real component source array.
 * @param im_row_stride[in] The stride in elements between rows in the imaginary component source array.
 */
template<typename CRT, typename CGL>
METAL_FUNC static typename metal::enable_if<ducks::is_complex_register_tile<CRT>() && ducks::is_complex_global_layout<CGL>(), void>::type
load(thread CRT &dst, thread const CGL &src, thread const coord &idx, const short laneid) {
    // Internally will use the correct load() method for row and column types
    load(dst.real, src.real, idx);
    load(dst.imag, src.imag, idx);
}

/**
 * @brief Store data from a complex register tile to destination arrays in global memory.
 *
 * @tparam CRT The complex tile type.
 * @tparam U The data type of the destination arrays.
 * @param redst[out] The destination array in global memory to store the real component data into.
 * @param imdst[out] The destination array in global memory to store the imaginary component data into.
 * @param src[in] The source register tile to store data from.
 * @param re_row_stride[in] The stride in elements between rows in the real component destination array.
 * @param im_row_stride[in] The stride in elements between rows in the imaginary component destination array.
 */
template<typename CRT, typename CGL>
METAL_FUNC static typename metal::enable_if<ducks::is_complex_register_tile<CRT>() && ducks::is_complex_global_layout<CGL>(), void>::type
store(thread CGL &dst, thread const CRT &src, thread const coord &idx) {
    // Internally will use the correct load() method for row and column types
    store(dst.real, src.real, idx);
    store(dst.imag, src.imag, idx);
}
}
