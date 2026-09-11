/**
 * @file
 * @brief Warp-scope conversions on shared vectors.
 */

#pragma once // done!

#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {
    

/**
 * @brief Copies data from one shared vector to another, converting data types if necessary.
 *
 * This function copies data from the source shared vector `src` to the destination shared vector `dst`.
 * If the data types of `src` and `dst` are the same, it performs a direct memory copy. Otherwise, it
 * converts each element from the source data type to the destination data type using the appropriate
 * converter before copying.
 *
 * @tparam SV1 The type of the destination shared vector, must satisfy the ducks::sv::all concept.
 * @tparam SV2 The type of the source shared vector, must satisfy the ducks::sv::all concept.
 * @param[out] dst The destination shared vector.
 * @param[in] src The source shared vector.
 * @note The lengths of `src` and `dst` must be equal. This is enforced at compile time.
 */
template<typename SV1, typename SV2>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV1>() && ducks::is_shared_vector<SV2>(), void>::type
copy(threadgroup SV1 &dst, threadgroup const SV2 &src, const ushort laneid) {
    static_assert(SV1::length == SV2::length, "Source and destination vectors must have the same length.");
    #pragma clang loop unroll(full)
    for(int i = laneid; i < dst.length; i+=SIMD_THREADS) {
        dst[i] = base_types::convertor<typename SV1::dtype, typename SV2::dtype>::convert(src[i]);
    }
}

/* ----------  SUBVEC  ---------- */

/**
* @brief Returns a reference to a subvec of a given shared vector
*
* @tparam subvec_tiles The length, in subtiles, of the subvec.
* @tparam SV The type of the input vector, which must satisfy the ducks::sv::all concept.
* @param src The input tile.
* @param vec_idx The index of the subtile, in units of subvec_tiles*16 elements.
* @return A reference to the subvec.
*
* @note The subvec length must evenly divide the vector length.
*/
template<int subvec_tiles, typename SV>
//using subvec = typename SV::template subvec<SV::length>;
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), threadgroup typename SV::template subvec<typename SV::dtype, subvec_tiles>&>::type
subvec_inplace(threadgroup SV &src, int vec_idx) {
    return *(threadgroup typename SV::template subvec<typename SV::dtype, subvec_tiles>*)(&src[vec_idx*TILE_DIM*subvec_tiles]);
}

}


