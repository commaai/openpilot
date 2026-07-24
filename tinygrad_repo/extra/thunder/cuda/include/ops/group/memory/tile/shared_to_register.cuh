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
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%GROUP_WARPS == 0, "Group load / store requires tile height to be a multiple of GROUP_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();

    // convert to shared state space
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            if constexpr (sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout for 16-bit types
                U2 tmp[4];
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + (warp_laneid % 16);
                int col = j*dst.tile_size_col + (warp_laneid / 16) * 8;
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));
                }
                else {
                    move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {row, col}));
                }
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 1) {
                // handle the row-major layout for 8-bit types
                int warp_group_16 = (warp_laneid / 16);  // divide each warp into two groups of 16 threads
                int lane_in_16 = warp_laneid % 16;       // position in group of 16 threads
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + (lane_in_16 % 16); // find base row for warp in warpgroup and then distribute the 16 threads in the warp across the rows
                int col = j*dst.tile_size_col + warp_group_16 * 16; // find base column and then *16 for second half of the warp

                U2 tmp[4];
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row, col}));
                }
                else {
                    move<U2>::ldsm4t(tmp[0], tmp[2], tmp[1], tmp[3], src.idx(shared_addr, {row, col}));
                }
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                // handle the row-major layout for 32-bit types
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + (warp_laneid / 4);
                int col = j*dst.tile_size_col + 2*(warp_laneid % 4);
                if constexpr (ST::rows != ST::underlying_rows || ST::cols != ST::underlying_cols) { // subtile case
                    row += src.row_offset;
                    col += src.col_offset;
                }
                int blit = sizeof(typename ST::dtype) * ((warp_laneid%4) / 2);
                U2 tmp[4];
                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(U);
                const int outer_idx = col/subtile_cols;
                const uint32_t addr_1 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+0)*subtile_cols + col%subtile_cols);
                const uint32_t addr_2 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+8)*subtile_cols + col%subtile_cols);
                const int swizzle_1 = blit ^ ((addr_1 % swizzle_repeat) >> 7) << 4;
                const int swizzle_2 = blit ^ ((addr_2 % swizzle_repeat) >> 7) << 4;
                move<U>::lds(tmp[0].x, (addr_1+ 0)^swizzle_1);
                move<U>::lds(tmp[0].y, (addr_1+ 4)^swizzle_1);
                move<U>::lds(tmp[2].x, (addr_1+32)^swizzle_1);
                move<U>::lds(tmp[2].y, (addr_1+36)^swizzle_1);
                move<U>::lds(tmp[1].x, (addr_2+ 0)^swizzle_2);
                move<U>::lds(tmp[1].y, (addr_2+ 4)^swizzle_2);
                move<U>::lds(tmp[3].x, (addr_2+32)^swizzle_2);
                move<U>::lds(tmp[3].y, (addr_2+36)^swizzle_2);
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        dst.tiles[i][j].data[k] = T2{dst.tiles[i][j].data[k].y, dst.tiles[i][j].data[k].x};
                    }
                }
            }
            else {
                // handle the column-major layout
                int row = (local_warpid*warp_height + i)*dst.tile_size_row + 2*(warp_laneid % 4);
                int col = j*dst.tile_size_col + (warp_laneid / 4);
                U2 tmp[4];
                move<U>::lds(tmp[0].x, src.idx(shared_addr, {row+0, col+0}));
                move<U>::lds(tmp[0].y, src.idx(shared_addr, {row+1, col+0}));
                move<U>::lds(tmp[1].x, src.idx(shared_addr, {row+0, col+8}));
                move<U>::lds(tmp[1].y, src.idx(shared_addr, {row+1, col+8}));
                move<U>::lds(tmp[2].x, src.idx(shared_addr, {row+8, col+0}));
                move<U>::lds(tmp[2].y, src.idx(shared_addr, {row+9, col+0}));
                move<U>::lds(tmp[3].x, src.idx(shared_addr, {row+8, col+8}));
                move<U>::lds(tmp[3].y, src.idx(shared_addr, {row+9, col+8}));
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
                dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            }
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
template<ducks::st::all ST, ducks::rt::all RT>
__device__ inline static void store(ST &dst, const RT &src) {
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(height%GROUP_WARPS == 0, "Group load / store requires tile height to be a multiple of GROUP_WARPS.");
    static_assert(height%warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width==RT::width, "Group load / store requires tile widths to match.");
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();

    // convert to shared state space
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

    #pragma unroll
    for(int i = 0; i < warp_height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            if constexpr (sizeof(typename ST::dtype) == 2) {
                // handle the row-major layout
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
#ifdef KITTENS_HOPPER
                int row = (local_warpid*warp_height + i)*src.tile_size_row + (warp_laneid % 16);
                int col = j*src.tile_size_col + (warp_laneid / 16) * 8;
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::stsm4(dst.idx(shared_addr, {row, col}), tmp[0], tmp[1], tmp[2], tmp[3]);
                }
                else {
                    move<U2>::stsm4t(dst.idx(shared_addr, {row, col}), tmp[0], tmp[2], tmp[1], tmp[3]);
                }
#else
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    int row = (local_warpid*warp_height + i)*src.tile_size_row + (warp_laneid / 4);
                    int col = j*src.tile_size_col + 2*(warp_laneid % 4);
                    move<U2>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0]);
                    move<U2>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[1]);
                    move<U2>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[2]);
                    move<U2>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3]);
                }
                else {
                    int row = (local_warpid*warp_height + i)*src.tile_size_row + 2*(warp_laneid % 4);
                    int col = j*src.tile_size_col + (warp_laneid / 4);
                    move<U>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0].x);
                    move<U>::sts(dst.idx(shared_addr, {row+1, col+0}), tmp[0].y);
                    move<U>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[1].x);
                    move<U>::sts(dst.idx(shared_addr, {row+1, col+8}), tmp[1].y);
                    move<U>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[2].x);
                    move<U>::sts(dst.idx(shared_addr, {row+9, col+0}), tmp[2].y);
                    move<U>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3].x);
                    move<U>::sts(dst.idx(shared_addr, {row+9, col+8}), tmp[3].y);
                }
#endif
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 1) { 
                // handle the row-major layout for 8-bit types

                int warp_group_16 = (warp_laneid / 16);  // divide each warp into two groups of 16 threads
                int lane_in_16 = warp_laneid % 16;       // position in group of 16 threads
                int row = (local_warpid*warp_height + i)*src.tile_size_row + (lane_in_16 % 16); // find base row for warp in warpgroup and then distribute the 16 threads in the warp across the rows
                int col = j*src.tile_size_col + warp_group_16 * 16; // find base column and then *16 for second half of the warp

                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                    move<U2>::stsm4(dst.idx(shared_addr, {row, col}), tmp[0], tmp[1], tmp[2], tmp[3]);
                }
                else {
                    move<U2>::stsm4t(dst.idx(shared_addr, {row, col}), tmp[0], tmp[2], tmp[1], tmp[3]);
                }
            }
            else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row> && sizeof(typename ST::dtype) == 4) {
                // handle the row-major layout for 32-bit types
                int row = (local_warpid*warp_height + i)*src.tile_size_row + (warp_laneid / 4);
                int col = j*src.tile_size_col + 2*(warp_laneid % 4);
                if constexpr (ST::rows != ST::underlying_rows || ST::cols != ST::underlying_cols) { // subtile case
                    row += dst.row_offset;
                    col += dst.col_offset;
                }
                int blit = sizeof(typename ST::dtype) * ((warp_laneid%4) / 2);
                T2 reg_tmp[4];
                if(blit) {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = T2{src.tiles[i][j].data[k].y, src.tiles[i][j].data[k].x};
                    }
                }
                else {
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        reg_tmp[k] = src.tiles[i][j].data[k];
                    }
                }
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(reg_tmp[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(reg_tmp[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(reg_tmp[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(reg_tmp[3]);
                static constexpr int swizzle_repeat = ST::swizzle_bytes * 8;
                static constexpr int subtile_cols   = ST::swizzle_bytes / sizeof(U);
                const int outer_idx = col/subtile_cols;
                const uint32_t addr_1 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+0)*subtile_cols + col%subtile_cols);
                const uint32_t addr_2 = shared_addr + sizeof(U)*(outer_idx*ST::underlying_rows*subtile_cols + (row+8)*subtile_cols + col%subtile_cols);
                const int swizzle_1 = blit ^ ((addr_1 % swizzle_repeat) >> 7) << 4;
                const int swizzle_2 = blit ^ ((addr_2 % swizzle_repeat) >> 7) << 4;
                move<U>::sts((addr_1+ 0)^swizzle_1, tmp[0].x);
                move<U>::sts((addr_1+ 4)^swizzle_1, tmp[0].y);
                move<U>::sts((addr_1+32)^swizzle_1, tmp[2].x);
                move<U>::sts((addr_1+36)^swizzle_1, tmp[2].y);
                move<U>::sts((addr_2+ 0)^swizzle_2, tmp[1].x);
                move<U>::sts((addr_2+ 4)^swizzle_2, tmp[1].y);
                move<U>::sts((addr_2+32)^swizzle_2, tmp[3].x);
                move<U>::sts((addr_2+36)^swizzle_2, tmp[3].y);
            }
            else {
                // handle the column-major layout
                int row = (local_warpid*warp_height + i)*src.tile_size_row + 2*(warp_laneid % 4);
                int col = j*src.tile_size_col + (warp_laneid / 4);
                U2 tmp[4];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                tmp[2] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
                tmp[3] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
                move<U>::sts(dst.idx(shared_addr, {row+0, col+0}), tmp[0].x);
                move<U>::sts(dst.idx(shared_addr, {row+1, col+0}), tmp[0].y);
                move<U>::sts(dst.idx(shared_addr, {row+0, col+8}), tmp[1].x);
                move<U>::sts(dst.idx(shared_addr, {row+1, col+8}), tmp[1].y);
                move<U>::sts(dst.idx(shared_addr, {row+8, col+0}), tmp[2].x);
                move<U>::sts(dst.idx(shared_addr, {row+9, col+0}), tmp[2].y);
                move<U>::sts(dst.idx(shared_addr, {row+8, col+8}), tmp[3].x);
                move<U>::sts(dst.idx(shared_addr, {row+9, col+8}), tmp[3].y);
            }
        }
    }
}

// Load and store of vectors from/to shared tiles.

template<ducks::rv::naive_layout RV, ducks::st::all ST>
__device__ inline static auto load(RV &dst, const ST &src, int2 row_col) {
    KITTENS_CHECK_WARP;
    static_assert(ST::cols>=RV::length, "Shared tile must be at least as wide as the vector.");
    using T = RV::T;
    using U = ST::T;
    int warp_laneid = ::kittens::laneid();

    // convert to shared state space
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));

    #pragma unroll
    for(int col = warp_laneid; col < dst.length; col+=WARP_THREADS) {
        U tmp;
        move<U>::lds(tmp, src.idx(shared_addr, {row_col.x, row_col.y + col}));
        dst.data[col/WARP_THREADS][0] = base_types::convertor<T, U>::convert(tmp);
    }
}

template<ducks::rv::naive_layout RV, ducks::st::all ST>
__device__ inline static auto store(ST &dst, const RV &src, int2 row_col) {
    KITTENS_CHECK_WARP;
    static_assert(ST::cols>=RV::length, "Shared tile must be at least as wide as the vector.");
    using T = RV::T;
    using U = ST::T;
    int warp_laneid = ::kittens::laneid();

    // convert to shared state space
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));

    #pragma unroll
    for(int col = warp_laneid; col < src.length; col+=WARP_THREADS) {
        U tmp = base_types::convertor<U, T>::convert(src.data[col/WARP_THREADS][0]);
        move<U>::sts(dst.idx(shared_addr, {row_col.x, row_col.y + col}), tmp);
    }
}