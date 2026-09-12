/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

#pragma once // not done:
/*
 swaping register layout doesn't exist. no layout to swap
 SUBTILE
 
 */
#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {
/* ----------  TRANSPOSE  ---------- */
METAL_FUNC int compute_laneid(ushort y, ushort x) {
    // Extract bits from simd_y
    ushort b1 = y & 1;
    ushort temp_y = y >> 1;
    ushort b2 = temp_y & 1;
    ushort b4 = temp_y >> 1;

    // Extract bits from simd_x
    ushort b0 = (x >> 1) & 1;
    ushort b3 = x >> 2;

    // Reconstruct laneid
    ushort laneid = (b4 << 4) | (b3 << 3) | (b2 << 2) | (b1 << 1) | b0;
    return laneid;
}
/**
 * @brief Transposes a register base tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register base tile to be transposed.
 */
template<typename T, typename layout>
static METAL_FUNC typename metal::enable_if<ducks::is_rt_layout<layout>(), void>::type
swap_layout(thread rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst,
            thread const rt_base<T, layout> &src,
            const ushort laneid) {
    const ushort qid = laneid / 4;
    const ushort simd_y = (qid & 4) + (laneid / 2) % 4;
    const ushort simd_x = (qid & 2) * 2 + (laneid % 2) * 2;

    const ushort src_laneid_start = compute_laneid(simd_x, simd_y);
    const ushort2 src_laneid = ushort2(src_laneid_start, src_laneid_start+(ushort)2);
    const ushort first_idx = (laneid / 2) % 2;
    
    dst.data.thread_elements()[first_idx] = shfl_sync<T>(src.data.thread_elements()[first_idx], src_laneid[first_idx]);

    dst.data.thread_elements()[1 - first_idx] = shfl_sync<T>(src.data.thread_elements()[1 - first_idx], src_laneid[1 - first_idx]);
}
    
/**
 * @brief Swaps the layout of a register tile.
 *
 * This function swaps the layout of a register tile by iterating over its height and width
 * and performing layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
template<typename T, int _height, int _width, typename layout>
static METAL_FUNC typename metal::enable_if<ducks::is_rt_layout<layout>(), void>::type
swap_layout(thread rt<T, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, thread const rt<T, _height, _width, layout> &src, const short laneid) {
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j], laneid);
        }
    }
}

/**
 * @brief Swaps the layout of a register base tile in place.
 *
 * This function swaps the layout of a register base tile in place by casting it to the
 * transposed layout type and then performing the layout swap.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register base tile to be swapped in place.
 * @return A reference to the swapped register base tile.
 */
template<typename T2, typename layout>
static METAL_FUNC typename metal::enable_if<ducks::is_rt_layout<layout>(), thread rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>&>::type
swap_layout_inplace(thread const rt_base<T2, layout> &src) {
    thread rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(thread rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}
    
/* ----------  TRANSPOSE  ---------- */
   
/**
 * @brief Transposes a register base tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register base tile to be transposed.
 */
template<typename T, typename layout>
static METAL_FUNC typename metal::enable_if<ducks::is_rt_layout<layout>(), void>::type
transpose(thread rt_base<T, layout> &dst, thread const rt_base<T, layout> &src, const ushort laneid) {
    const ushort qid = laneid / 4;
    const ushort simd_y = (qid & 4) + (laneid / 2) % 4;
    const ushort simd_x = (qid & 2) * 2 + (laneid % 2) * 2;

    const ushort src_laneid_start = compute_laneid(simd_x, simd_y);
    const ushort2 src_laneid = ushort2(src_laneid_start, src_laneid_start+(ushort)2);
    const ushort first_idx = (laneid / 2) % 2;
    
    dst.data.thread_elements()[first_idx] = shfl_sync<T>(src.data.thread_elements()[first_idx], src_laneid[first_idx]);

    dst.data.thread_elements()[1 - first_idx] = shfl_sync<T>(src.data.thread_elements()[1 - first_idx], src_laneid[1 - first_idx]);
}
/**
 * @brief Transposes a register tile.
 *
 * This function is marked "sep", which means that the registers underlying dst MUST be separate
 * from the registers underlying src.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the src register tile, and the width of the dst tile.
 * @tparam _width The width of the src register tile, and the height of the dst tile.
 * @tparam layout The layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
transpose_sep(thread RT &dst, thread const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src,
              const int laneid) {
    #pragma clang loop unroll(full)
    for(int i = 0; i < RT::height; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < RT::width; j++) {
            transpose(dst.tiles[i][j], src.tiles[j][i], laneid);
        }
    }
}
    
/**
 * @brief Transposes a register base tile in-place.
 *
 * @tparam T2 The data type of the register base tile elements.
 * @tparam layout The current layout of the register base tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register base tile.
 */
template<typename T2, typename layout>
static METAL_FUNC typename metal::enable_if<ducks::is_rt_layout<layout>(), thread rt_base<T2, layout>&>::type
transpose_inplace(thread rt_base<T2, layout> &src, const ushort laneid) {
    transpose(src, src, laneid);
    return src;
}

template<typename T, typename U, typename layout>
static METAL_FUNC typename metal::enable_if<ducks::is_rt_layout<layout>(), void>::type
copy(thread rt_base<T, layout> &dst, thread const rt_base<U, layout> &src);

/**
 * @brief Transposes a square register tile in-place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height (in units of 16) of the src register tile, and the width of the dst tile. (Must be the same as _width.)
 * @tparam _width The width (in units of 16) of the src register tile, and the height of the dst tile. (Must be the same as _height.)
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register tile.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && RT::cols == RT::rows, thread RT&>::type
transpose_inplace(thread RT &tile, const ushort laneid) {
    #pragma clang loop unroll(full)
    for(int i = 0; i < tile.height; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < i; j++) {
            rt_base<typename RT::T, typename RT::layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i], laneid);
            transpose(tile.tiles[j][i], tmp, laneid);
        }
        transpose_inplace(tile.tiles[i][i], laneid);
    }
    return tile;
}
/* ----------  TYPE SWAPS  ---------- */
/**
 * @brief Copies a register base tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam layout The current layout of the register base tile.
 * @param[out] dst A reference to the destination register base tile.
 * @param[in] src A reference to the source register base tile.
 */
template<typename T, typename U, typename layout>
static METAL_FUNC typename metal::enable_if<ducks::is_rt_layout<layout>(), void>::type
copy(thread rt_base<T, layout> &dst, thread const rt_base<U, layout> &src) {
    using T1 = typename base_types::packing<T>::unpacked_type;
    using U1 = typename base_types::packing<U>::unpacked_type;
    dst.data.thread_elements()[0] = base_types::convertor<T1, U1>::convert(src.data.thread_elements()[0]);
    dst.data.thread_elements()[1] = base_types::convertor<T1, U1>::convert(src.data.thread_elements()[1]);
}

/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 8) of the register tiles.
 * @tparam _width The width (in units of 8) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T, typename U, int _height, int _width, typename layout>
static METAL_FUNC typename metal::enable_if<ducks::is_rt_layout<layout>(), void>::type
copy(thread rt<T, _height, _width, layout> &dst, thread const rt<U, _height, _width, layout> &src) {
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/* ----------  CAUSAL  ---------- */

/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * This function modifies a square register tile in-place to make it causal. All elements
 * above the main diagonal are set to zero, while elements on or below the main diagonal
 * are left unchanged.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam _size The size (height and width) of the square register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be made causal.
 */
template<typename RT>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>(), void>::type
make_causal(thread RT &dst, thread const RT &src, const unsigned laneid, thread const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    ducks::assert_register_tile<RT>();
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            if(j < i) { // below the diagonal, copy
                dst.tiles[i][j].data.thread_elements()[0] = src.tiles[i][j].data.thread_elements()[0];
                dst.tiles[i][j].data.thread_elements()[1] = src.tiles[i][j].data.thread_elements()[1];
            }
            else if(j > i) { // above the diagonal, zero
                dst.tiles[i][j].data.thread_elements()[0] = val;
                dst.tiles[i][j].data.thread_elements()[1] = val;
            }
            else { // on the diagonal
                constexpr uint32_t MASK_0 = (ducks::is_row_register_tile<RT>()) ? 0x0A00FF0A : 0xD4FF00D4;
                constexpr uint32_t MASK_1 = (ducks::is_row_register_tile<RT>()) ? 0x2B00FF2B : 0x50FF0050;
                if((MASK_0 >> laneid) & 1) {
                    dst.tiles[i][j].data.thread_elements()[0] = val;
                }
                else {
                    dst.tiles[i][j].data.thread_elements()[0] = src.tiles[i][j].data.thread_elements()[0];
                }
                if((MASK_1 >> laneid) & 1) {
                    dst.tiles[i][j].data.thread_elements()[1] = val;
                }
                else {
                    dst.tiles[i][j].data.thread_elements()[1] = src.tiles[i][j].data.thread_elements()[1];
                }
            }
        }
    }
}


    
/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The index of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
//template<int subtile_height, ducks::rt::all RT>
//__device__ inline rt<typename RT::T, subtile_height, RT::width, typename RT::layout> &subtile_inplace(RT & src, int idx) {
//    static_assert(RT::height % subtile_height == 0, "subtile height should evenly divide tile height.");
//    return reinterpret_cast<rt<typename RT::T, subtile_height, RT::width, typename RT::layout>&>(
//        src.tiles[idx*subtile_height]
//    );
//}

}
