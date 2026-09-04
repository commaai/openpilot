
/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly between global memory and registers and back.
 */

/**
 * @brief Collaboratively loads data from a source array into row-major layout tiles.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from. 
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<typename RT, typename GL>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_global_layout<GL>(), void>::type
load(thread RT &dst, thread const GL &_src, thread const coord &idx, const int threadIdx) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    const device U *src = (device U*)&_src.template get<RT>(idx);
    const int row_stride = _src.row_stride();
    
    int warp_laneid = threadIdx % 32;
    const int row_offset = dst.rows * warpid(threadIdx);
    const short qid = warp_laneid / 4;
    const short simd_y = row_offset + (qid & 4) + (warp_laneid / 2) % 4;
    const short simd_x = (qid & 2) * 2 + (warp_laneid % 2) * 2;
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        int row = simd_y + i * RT::tile_size;
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            int col = simd_x + j * RT::tile_size;
            T2 src2 = base_types::convertor<T2, U2>::convert(*((device U2*)(&src[row * row_stride + col])));
            dst.tiles[i][j].data.thread_elements()[0] = src2[0];
            dst.tiles[i][j].data.thread_elements()[1] = src2[1];
        }
    }
}

template<typename RT, typename GL>
static METAL_FUNC typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_global_layout<GL>(), void>::type
load(thread RT &dst, thread const GL &_src, thread const coord &idx, const int threadIdx) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    const device U *src = (device U*)&_src.template get<RT>(idx);
    const int row_stride = _src.row_stride();
    
    int warp_laneid = threadIdx % 32;
    const int row_offset = dst.rows * warpid(threadIdx);
    const short qid = warp_laneid / 4;
    const short simd_y = row_offset + (qid & 2) * 2 + (warp_laneid % 2) * 2;;
    const short simd_x = (qid & 4) + (warp_laneid / 2) % 4;
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        int row = simd_y + i * RT::tile_size;
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            int col = simd_x + j * RT::tile_size;
            T2 src2 = base_types::convertor<T2, U2>::convert(*((device U2*)(&src[row * row_stride + col])));
            dst.tiles[i][j].data.thread_elements()[0] = base_types::convertor<T, U>::convert(src[row * row_stride + col]);
            dst.tiles[i][j].data.thread_elements()[1] = base_types::convertor<T, U>::convert(src[(row + 1) * row_stride + col]);
        }
    }
}
/**
 * @brief Collaboratively stores data from register tiles to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<typename RT, typename GL>
static METAL_FUNC typename metal::enable_if<ducks::is_row_register_tile<RT>(), void>::type
store(thread GL &_dst, thread const RT &src, thread const coord &idx, const int threadIdx) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    device U *dst = (device U*)&(_dst.template get<RT>(idx));
    const int row_stride = _dst.row_stride();
    int warp_laneid = simd_laneid(threadIdx);
    const int row_offset = src.rows * warpid(threadIdx);
    const short qid = warp_laneid / 4;
    const short simd_y = row_offset + (qid & 4) + (warp_laneid / 2) % 4;
    const short simd_x = (qid & 2) * 2 + (warp_laneid % 2) * 2;
    #pragma clang loop unroll(full)
    for(int i = 0; i < src.height; i++) {
        int row = simd_y + i * RT::tile_size;
        #pragma clang loop unroll(full)
        for(int j = 0; j < src.width; j++) {
            int col = simd_x + j * RT::tile_size;
            U2 src2 = base_types::convertor<U2, T2>::convert(T2(src.tiles[i][j].data.thread_elements()[0], src.tiles[i][j].data.thread_elements()[1]));
            *(device U2*)(&dst[row*row_stride + col]) = src2;
        }
    }
}

template<typename RT, typename GL>
static METAL_FUNC typename metal::enable_if<ducks::is_col_register_tile<RT>(), void>::type
store(thread GL &_dst, thread const RT &src, thread const coord &idx, const int threadIdx) {
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    device U *dst = (device U*)&(_dst.template get<RT>(idx));
    const int row_stride = _dst.row_stride();
    int warp_laneid = simd_laneid(threadIdx);
    const int row_offset = src.rows * warpid(threadIdx);
    const short qid = warp_laneid / 4;
//    const short simd_y = row_offset + (qid & 4) + (warp_laneid / 2) % 4;
//    const short simd_x = (qid & 2) * 2 + (warp_laneid % 2) * 2;
    const short simd_y = row_offset + (qid & 2) * 2 + (warp_laneid % 2) * 2;
    const short simd_x = (qid & 4) + (warp_laneid / 2) % 4;
    #pragma clang loop unroll(full)
    for(int i = 0; i < src.height; i++) {
        int row = simd_y + i * RT::tile_size;
        #pragma clang loop unroll(full)
        for(int j = 0; j < src.width; j++) {
            int col = simd_x + j * RT::tile_size;
            dst[row*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[0]);
            dst[(row + 1) * row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[1]);
        }
    }
}
