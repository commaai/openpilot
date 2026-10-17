/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer data directly between shared memory and registers and back.
 */

/**
 * @brief Collaboratively load data from a shared tile into register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
load(thread RT &dst, threadgroup const ST &src, const int threadIdx) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(warp_height * N_WARPS == height, "RT height * N_WARPS must = ST height");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    
    int warp_laneid = simd_laneid(threadIdx);
    const int row_offset = RT::rows * warpid(threadIdx);
    const short qid = warp_laneid / 4;
    const short simd_y = row_offset + (qid & 4) + (warp_laneid / 2) % 4;
    const short simd_x = (qid & 2) * 2 + (warp_laneid % 2) * 2;
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        int row = simd_y + i * mittens::TILE_DIM;
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            int col = simd_x + j * mittens::TILE_DIM;
            T2 src2 = base_types::convertor<T2, U2>::convert(*((threadgroup U2*)(&src[{row, col}])));
            dst.tiles[i][j].data.thread_elements()[0] = src2[0];
            dst.tiles[i][j].data.thread_elements()[1] = src2[1];
        }
    } 
}

template<typename RT, typename ST>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
load(thread RT &dst, threadgroup const ST &src, const int threadIdx) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(warp_height * N_WARPS == height, "RT height * N_WARPS must = ST height");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    
    int warp_laneid = simd_laneid(threadIdx);
    const int row_offset = RT::rows * warpid(threadIdx);
    const short qid = warp_laneid / 4;
    const short simd_y = row_offset + (qid & 2) * 2 + (warp_laneid % 2) * 2;
    const short simd_x = (qid & 4) + (warp_laneid / 2) % 4;
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.height; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.width; j++) {
            int row = simd_y + i * mittens::TILE_DIM;
            int col = simd_x + j * mittens::TILE_DIM;
            dst.tiles[i][j].data.thread_elements()[0] = base_types::convertor<T, U>::convert(src[{row + 0, col}]);
            dst.tiles[i][j].data.thread_elements()[1] = base_types::convertor<T, U>::convert(src[{row + 1, col}]);
        }
    }
}

/**
 * @brief Collaboratively store data into a shared tile from register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<typename ST, typename RT>
METAL_FUNC static typename metal::enable_if<ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
store(threadgroup ST &dst, thread const RT &src, const int threadIdx) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(warp_height * N_WARPS == height, "RT height * N_WARPS must = ST height");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int warp_laneid = simd_laneid(threadIdx);
    const int row_offset = RT::rows * warpid(threadIdx);
    const short qid = warp_laneid / 4;
    const short simd_y = row_offset + (qid & 4) + (warp_laneid / 2) % 4;
    const short simd_x = (qid & 2) * 2 + (warp_laneid % 2) * 2;
    #pragma clang loop unroll(full)
    for(int i = 0; i < RT::height; i++) {
        int row = simd_y + i * mittens::TILE_DIM;
        #pragma clang loop unroll(full)
        for(int j = 0; j < RT::width; j++) {
            int col = simd_x + j * mittens::TILE_DIM;
            U2 src2 = base_types::convertor<U2, T2>::convert(T2(src.tiles[i][j].data.thread_elements()[0],
                                                                src.tiles[i][j].data.thread_elements()[1]));
            *(threadgroup U2*)(&dst[{row, col}]) = src2;
        }
    }
}


template<typename ST, typename RT>
METAL_FUNC static typename metal::enable_if<ducks::is_col_register_tile<RT>() && ducks::is_shared_tile<ST>(), void>::type
store(threadgroup ST &dst, thread const RT &src, const int threadIdx) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%N_WARPS == 0, "Group load / store requires tile height to be a multiple of N_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(warp_height * N_WARPS == height, "RT height * N_WARPS must = ST height");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    using T  = typename RT::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U  = typename ST::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    int warp_laneid = simd_laneid(threadIdx);
    const int row_offset = RT::rows * warpid(threadIdx);
    const short qid = warp_laneid / 4;
//    const short simd_y = row_offset + (qid & 4) + (warp_laneid / 2) % 4;
//    const short simd_x = (qid & 2) * 2 + (warp_laneid % 2) * 2;
    const short simd_y = row_offset + (qid & 2) * 2 + (warp_laneid % 2) * 2;
    const short simd_x = (qid & 4) + (warp_laneid / 2) % 4;
    #pragma clang loop unroll(full)
    for(int i = 0; i < RT::height; i++) {
        
        #pragma clang loop unroll(full)
        for(int j = 0; j < RT::width; j++) {
            int row = simd_y + i * mittens::TILE_DIM;
            int col = simd_x + j * mittens::TILE_DIM;
//            U2 src2 = base_types::convertor<U2, T2>::convert(T2(src.tiles[i][j].data.thread_elements()[0],
//                                                                src.tiles[i][j].data.thread_elements()[1]));
//            *(threadgroup U2*)(&dst[{row, col}]) = src2;
            
            dst[{row + 0, col}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[0]);
            dst[{row + 1, col}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data.thread_elements()[1]);
        }
    }
}
