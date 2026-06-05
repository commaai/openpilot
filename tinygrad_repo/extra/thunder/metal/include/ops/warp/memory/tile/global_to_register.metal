/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once // done!
#include "../../../../types/types.metal"
#include "../../../../common/common.metal"
#include <metal_stdlib>
namespace mittens{

namespace meta {
template<typename RT, typename U>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>(), void>::type
load(int i, int j, thread RT *dst, const device U *src_ptr, const short simd_y, const short simd_x, const int row_stride) {
    using T = typename RT::dtype;
    using T2 = typename RT::T2;
    using U2 = typename base_types::packing<U>::packed_type;
    using layout = typename RT::layout;
    unsigned offset = (simd_y + i * rt_base<T, layout>::tile_size) * row_stride + (simd_x + j * rt_base<T, layout>::tile_size);
    T2 src2 = base_types::convertor<T2, U2>::convert(*((device U2*)(&src_ptr[offset])));
    dst->tiles[i][j].data.thread_elements()[0] = src2[0];
    dst->tiles[i][j].data.thread_elements()[1] = src2[1];
}
    
template<typename RT, typename U>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>(), void>::type
load(int i, int j, thread RT *dst, const device U *src_ptr, const short simd_y, const short simd_x, const int row_stride) {
    using T = typename RT::dtype;
    using T2 = typename RT::T2;
    using U2 = typename base_types::packing<U>::packed_type;
    using layout = typename RT::layout;
    unsigned offset = (simd_y + i * rt_base<T, layout>::tile_size) * row_stride + (simd_x + j * rt_base<T, layout>::tile_size);
    dst->tiles[i][j].data.thread_elements()[0] = base_types::convertor<T, U>::convert(src_ptr[offset]);
    offset += row_stride;
    dst->tiles[i][j].data.thread_elements()[1] = base_types::convertor<T, U>::convert(src_ptr[offset]);
}
    
template<typename RT, typename U>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>(), void>::type
store(int i, int j, device U *dst_ptr, const thread RT *src, const short simd_y, const short simd_x, const int row_stride) {
    using T = typename RT::dtype;
    using T2 = typename RT::T2;
    using U2 = typename base_types::packing<U>::packed_type;
    using layout = typename RT::layout;
    unsigned offset = (simd_y + i * TILE_DIM) * row_stride + (simd_x + j * TILE_DIM);
    U2 src2 = base_types::convertor<U2, T2>::convert(
                                                     T2(src->tiles[i][j].data.thread_elements()[0],
                                                        src->tiles[i][j].data.thread_elements()[1])
                                                     );
    *((device U2*)&dst_ptr[offset]) = src2;
}

template<typename RT, typename U>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>(), void>::type
store(int i, int j, device U *dst_ptr, const thread RT *src, const short simd_y, const short simd_x, const int row_stride) {
    using T = typename RT::dtype;
    using T2 = typename RT::T2;
    using U2 = typename base_types::packing<U>::packed_type;
    using layout = typename RT::layout;
    unsigned offset = (simd_y + i * rt_base<T, layout>::tile_size) * row_stride + (simd_x + j * rt_base<T, layout>::tile_size);
    dst_ptr[offset] = base_types::convertor<U, T>::convert(src->tiles[i][j].data.thread_elements()[0]);
    offset += row_stride;
    dst_ptr[offset] = base_types::convertor<U, T>::convert(src->tiles[i][j].data.thread_elements()[1]);
}

}

/**
 * @brief Load data from a source array into a row-major layout tile.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<typename RT, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_global_layout<GL>(), void>::type
load(thread RT &dst, thread const GL &src, thread const coord &idx, const short laneid) {
    using T = typename RT::dtype;
    using T2 = typename RT::T2;
    using U = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    using layout = typename RT::layout;
    const device U *src_ptr = (device U*)&src.template get<RT>(idx);
    const int row_stride = src.row_stride();
    
    const short qid = laneid / 4;
    const short simd_y = (qid & 4) + (laneid / 2) % 4;
    const short simd_x = (qid & 2) * 2 + (laneid % 2) * 2;
    
//    #pragma clang loop unroll(full)
//    for (int i = 0; i < RT::height; i++) {
//        #pragma clang loop unroll(full)
//        for (int j = 0; j < RT::width; j++) {
//            unsigned offset = (simd_y + i * rt_base<T, layout>::tile_size) * row_stride + (simd_x + j * rt_base<T, layout>::tile_size);
//            T2 src2 = base_types::convertor<T2, U2>::convert(*((device U2*)(&src_ptr[offset])));
//            dst.tiles[i][j].data.thread_elements()[0] = src2[0];
//            dst.tiles[i][j].data.thread_elements()[1] = src2[1];
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::load<RT, U>, &dst, src_ptr, simd_y, simd_x, row_stride);
}
/**
 * @brief Load data from a source array into a col-major layout tile.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<typename RT, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_global_layout<GL>(), void>::type
load(thread RT &dst, thread const GL &src, thread const coord &idx, const short laneid) {
    using T = typename RT::dtype;
    using T2 = typename RT::T2;
    using U = typename GL::dtype;
    using layout = typename RT::layout;
    const device U *src_ptr = (device U*)&(src.template get<RT>(idx));
    const int row_stride = src.row_stride();
    
    const short qid = laneid / 4;
    const short simd_x = (qid & 4) + (laneid / 2) % 4;
    const short simd_y = (qid & 2) * 2 + (laneid % 2) * 2;
    
//    #pragma clang loop unroll(full)
//    for (int i = 0; i < RT::height; i++) {
//        #pragma clang loop unroll(full)
//        for (int j = 0; j < RT::width; j++) {
//            unsigned offset = (simd_y + i * rt_base<T, layout>::tile_size) * row_stride + (simd_x + j * rt_base<T, layout>::tile_size);
//            dst.tiles[i][j].data.thread_elements()[0] = base_types::convertor<T, U>::convert(src_ptr[offset]);
//            offset += row_stride;
//            dst.tiles[i][j].data.thread_elements()[1] = base_types::convertor<T, U>::convert(src_ptr[offset]);
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::load<RT, U>, &dst, src_ptr, simd_y, simd_x, row_stride);
}
    
/**
 * @brief Store data from a register tile to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<typename RT, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_global_layout<GL>(), void>::type
store(thread GL &dst, thread const RT &src, thread const coord &idx, const short laneid) {
    using T  = typename RT::dtype;
    using T2 = typename RT::T2;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    using layout = typename RT::layout;
    device U *dst_ptr = (device U*)&(dst.template get<RT>(idx));
//    device U* dst_ptr = dst.raw_ptr;
    const int row_stride = dst.row_stride();
    const short qid = laneid / 4;
    const short simd_y = (qid & 4) + (laneid / 2) % 4;
    const short simd_x = (qid & 2) * 2 + (laneid % 2) * 2;
    
//    #pragma clang loop unroll(full)
//    for (int i = 0; i < RT::height; i++) {
//        #pragma clang loop unroll(full)
//        for (int j = 0; j < RT::width; j++) {
//            unsigned offset = (simd_y + i * TILE_DIM) * row_stride + (simd_x + j * TILE_DIM);
//            U2 src2 = base_types::convertor<U2, T2>::convert(
//                                                             T2(src.tiles[i][j].data.thread_elements()[0],
//                                                                src.tiles[i][j].data.thread_elements()[1])
//                                                             );
//            *((device U2*)&dst_ptr[offset]) = src2;
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::store<RT, U>, dst_ptr, &src, simd_y, simd_x, row_stride);
}

/**
 * @brief Store data from a register tile to a destination array in global memory with a col-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<typename RT, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_global_layout<GL>(), void>::type
store(thread GL &dst, thread const RT &src, thread const coord &idx, const short laneid) {
    using T  = typename RT::dtype;
    using T2 = typename RT::T2;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    using layout = typename RT::layout;
    device U *dst_ptr = (device U*)&(dst.template get<RT>(idx));
    const int row_stride = dst.row_stride();
    const short qid = laneid / 4;
    const short simd_x = (qid & 4) + (laneid / 2) % 4;
    const short simd_y = (qid & 2) * 2 + (laneid % 2) * 2;
    
//    #pragma clang loop unroll(full)
//    for (int i = 0; i < RT::height; i++) {
//        #pragma clang loop unroll(full)
//        for (int j = 0; j < RT::width; j++) {
//            unsigned offset = (simd_y + i * rt_base<T, layout>::tile_size) * row_stride + (simd_x + j * rt_base<T, layout>::tile_size);
//            dst_ptr[offset] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[0]);
//            offset += row_stride;
//            dst_ptr[offset] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[1]);
//        }
//    }
    meta::unroll_i_j_in_range<0, RT::height, 1, 0, RT::width, 1>::run(meta::store<RT, U>, dst_ptr, &src, simd_y, simd_x, row_stride);
}


}
