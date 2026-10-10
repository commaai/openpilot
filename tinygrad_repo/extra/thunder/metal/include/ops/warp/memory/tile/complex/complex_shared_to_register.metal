/**
* @file
* @brief Functions for transferring data directly between shared memory and registers and back.
*/

#pragma once


#include "../../../../../common/common.metal"
#include "../../../../../types/types.metal"

#include "../shared_to_register.metal"

namespace mittens {
/**
 * @brief Load data from a complex shared tile into a complex register tile.
 *
 * @tparam CRT The complex register tile type
 * @tparam CST The complex shared tile type
 * @param dst[out] The destination complex register tile.
 * @param src[in]  The source complex shared tile.
 */
template<typename CRT, typename CST>
METAL_FUNC static typename metal::enable_if<ducks::is_complex_shared_tile<CST>() && ducks::is_complex_register_tile<CRT>(), void>::type
load(thread CRT &dst, threadgroup const CST &src) {
    load(dst.real, src.real);
    load(dst.imag, src.imag);
}

/**
 * @brief Store data into a complex shared tile from a complex register tile.
 *
 * @tparam RT The complex register tile type
 * @tparam ST The complex shared tile type
 * @param dst[out] The destination complex shared tile.
 * @param src[in]  The source complex register tile.
 */
template<typename CRT, typename CST>
METAL_FUNC static typename metal::enable_if<ducks::is_complex_shared_tile<CST>() && ducks::is_complex_register_tile<CRT>(), void>::type
store(threadgroup CST &dst, thread const CRT &src) {
    store(dst.real, src.real);
    store(dst.imag, src.imag);
}


}

