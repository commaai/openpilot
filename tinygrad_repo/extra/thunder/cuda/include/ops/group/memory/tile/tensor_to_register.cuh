/**
 * @file
 * @brief Group (collaborative warp) ops for loading tensor tiles into register tiles.
 */

/**
 * @brief Load data from a tensor tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam TM The tensor memory tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source tensor tile.
 */
template<ducks::rt::row_layout RT, ducks::tt::all TM>
__device__ inline static void load_async(RT &dst, const TM &src) {
    if constexpr (GROUP_WARPS == 1) {
        static_assert(RT::height == TM::height, "register tile and tensor tile must match height");
        static_assert(RT::width == TM::width, "register tile and tensor tile must match width");

        using T2 = RT::dtype;
        using U  = typename TM::dtype;
        using U2 = base_types::packing<typename TM::dtype>::packed_type;

        if constexpr (sizeof(typename TM::dtype) == 1) {
            #pragma unroll
            for(int i = 0; i < dst.height; i++) {
                #pragma unroll
                for(int j = 0; j < dst.width; j++) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(*(uint32_t*) &dst.tiles[i][j].data[0]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[1]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[2]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[3])
                        : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))
                    );
                }
            }
        } else if constexpr (sizeof(typename TM::dtype) == 2) {
            #pragma unroll
            for(int i = 0; i < dst.height; i++) {
                #pragma unroll
                for(int j = 0; j < dst.width; j++) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(*(uint32_t*) &dst.tiles[i][j].data[0]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[1]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[2]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[3])
                        : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col))
                    );
                }
            }
        }
        else if constexpr (sizeof(typename TM::dtype) == 4) {
            #pragma unroll
            for(int i = 0; i < dst.height; i++) {
                if constexpr (dst.width%4 == 0) {
                    #pragma unroll
                    for(int j = 0; j < dst.width; j+=4) {
                        U2 data[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
                            : "=f"(data[0].x), "=f"(data[0].y),
                            "=f"(data[1].x), "=f"(data[1].y),
                            "=f"(data[2].x), "=f"(data[2].y),
                            "=f"(data[3].x), "=f"(data[3].y),
                            "=f"(data[4].x), "=f"(data[4].y),
                            "=f"(data[5].x), "=f"(data[5].y),
                            "=f"(data[6].x), "=f"(data[6].y),
                            "=f"(data[7].x), "=f"(data[7].y),
                            "=f"(data[8].x), "=f"(data[8].y),
                            "=f"(data[9].x), "=f"(data[9].y),
                            "=f"(data[10].x), "=f"(data[10].y),
                            "=f"(data[11].x), "=f"(data[11].y),
                            "=f"(data[12].x), "=f"(data[12].y),
                            "=f"(data[13].x), "=f"(data[13].y),
                            "=f"(data[14].x), "=f"(data[14].y),
                            "=f"(data[15].x), "=f"(data[15].y)
                            : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))
                        );
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            dst.tiles[i][j+0].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                            dst.tiles[i][j+1].data[k] = base_types::convertor<T2, U2>::convert(data[k+4]);
                            dst.tiles[i][j+2].data[k] = base_types::convertor<T2, U2>::convert(data[k+8]);
                            dst.tiles[i][j+3].data[k] = base_types::convertor<T2, U2>::convert(data[k+12]);
                        }
                    }
                }
                else if constexpr (dst.width%2 == 0) {
                    #pragma unroll
                    for(int j = 0; j < dst.width; j+=2) {
                        U2 data[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
                            : "=f"(data[0].x), "=f"(data[0].y),
                            "=f"(data[1].x), "=f"(data[1].y),
                            "=f"(data[2].x), "=f"(data[2].y),
                            "=f"(data[3].x), "=f"(data[3].y),
                            "=f"(data[4].x), "=f"(data[4].y),
                            "=f"(data[5].x), "=f"(data[5].y),
                            "=f"(data[6].x), "=f"(data[6].y),
                            "=f"(data[7].x), "=f"(data[7].y)
                            : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))
                        );
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            dst.tiles[i][j+0].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                            dst.tiles[i][j+1].data[k] = base_types::convertor<T2, U2>::convert(data[k+4]);
                        }
                    }
                }
                else {
                    #pragma unroll
                    for(int j = 0; j < dst.width; j++) {
                        U2 data[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                            : "=f"(data[0].x), "=f"(data[0].y),
                            "=f"(data[1].x), "=f"(data[1].y),
                            "=f"(data[2].x), "=f"(data[2].y),
                            "=f"(data[3].x), "=f"(data[3].y)
                            : "r"(src.addr + ((i * dst.tile_size_row) << 16) + (j * dst.tile_size_col)/(4/(uint32_t)sizeof(U)))
                        );
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                        }
                    }
                }
            }
        }
    }
    else {
        static_assert(GROUP_WARPS==4 || GROUP_WARPS==8);
        constexpr int warp_rows = TM::rows/GROUP_WARPS;
        static_assert(TM::cols==RT::cols);
        static_assert(warp_rows==RT::rows);
        if constexpr (GROUP_WARPS == 4) {
            auto src_subtile = src.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*warpid(), 0);
            ::kittens::group<1>::load_async(dst, src_subtile);
        }
        else {
            auto src_subtile = src.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*(warpid()%4)+16*(warpid()/4), 0);
            ::kittens::group<1>::load_async(dst, src_subtile);
        }
    }
}


/**
 * @brief Store data into a tensor tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam TM The tensor memory tile type
 * @param dst[out] The destination tensor tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::tt::all TM>
__device__ inline static void store_async(TM &dst, const RT &src) {
    if constexpr (GROUP_WARPS == 1) {
        static_assert(RT::height == TM::height, "register tile and tensor tile must match height");
        static_assert(RT::width == TM::width, "register tile and tensor tile must match width");

        using T2 = RT::dtype;
        using T = base_types::packing<T2>::unpacked_type;
        using U = TM::dtype;
        using U2 = base_types::packing<U>::packed_type;

        if constexpr (sizeof(typename TM::dtype) == 2) {
            #pragma unroll
            for(int i = 0; i < src.height; i++) {
                if constexpr (src.width%4 == 0) {
                    #pragma unroll
                    for(int j = 0; j < src.width; j+=4) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[3])
                        );
                    }
                }
                else if constexpr (src.width%2 == 0) {
                    #pragma unroll
                    for(int j = 0; j < src.width; j+=2) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[3])
                        );
                    }
                }
                else {
                    #pragma unroll
                    for(int j = 0; j < src.width; j++) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[3])
                        );
                    }
                }
            }
        }
        else if constexpr (sizeof(typename TM::dtype) == 4) {
            #pragma unroll
            for(int i = 0; i < src.height; i++) {
                if constexpr(src.width%4 == 0) {
                    #pragma unroll
                    for(int j = 0; j < src.width; j+=4) {
                        U2 data[16];
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                            data[k+4] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+1].data[k]);
                            data[k+8] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+2].data[k]);
                            data[k+12] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+3].data[k]);
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "f"(data[0].x), "f"(data[0].y),
                            "f"(data[1].x), "f"(data[1].y),
                            "f"(data[2].x), "f"(data[2].y),
                            "f"(data[3].x), "f"(data[3].y),
                            "f"(data[4].x), "f"(data[4].y),
                            "f"(data[5].x), "f"(data[5].y),
                            "f"(data[6].x), "f"(data[6].y),
                            "f"(data[7].x), "f"(data[7].y),
                            "f"(data[8].x), "f"(data[8].y),
                            "f"(data[9].x), "f"(data[9].y),
                            "f"(data[10].x), "f"(data[10].y),
                            "f"(data[11].x), "f"(data[11].y),
                            "f"(data[12].x), "f"(data[12].y),
                            "f"(data[13].x), "f"(data[13].y),
                            "f"(data[14].x), "f"(data[14].y),
                            "f"(data[15].x), "f"(data[15].y)
                        );
                    }
                }
                else if constexpr(src.width%2 == 0) {
                    #pragma unroll
                    for(int j = 0; j < src.width; j+=2) {
                        U2 data[8];
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                            data[k+4] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+1].data[k]);
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "f"(data[0].x), "f"(data[0].y),
                            "f"(data[1].x), "f"(data[1].y),
                            "f"(data[2].x), "f"(data[2].y),
                            "f"(data[3].x), "f"(data[3].y),
                            "f"(data[4].x), "f"(data[4].y),
                            "f"(data[5].x), "f"(data[5].y),
                            "f"(data[6].x), "f"(data[6].y),
                            "f"(data[7].x), "f"(data[7].y)
                        );
                    }
                }
                else {
                    #pragma unroll
                    for(int j = 0; j < src.width; j++) {
                        U2 data[4];
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                            :: "r"(dst.addr + ((i * src.tile_size_row) << 16) + (j * src.tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "f"(data[0].x), "f"(data[0].y),
                            "f"(data[1].x), "f"(data[1].y),
                            "f"(data[2].x), "f"(data[2].y),
                            "f"(data[3].x), "f"(data[3].y)
                        );
                    }
                }
            }
        }
    }
    else {
        static_assert(GROUP_WARPS==4 || GROUP_WARPS==8);
        constexpr int warp_rows = TM::rows/GROUP_WARPS;
        static_assert(TM::cols==RT::cols);
        static_assert(warp_rows==RT::rows);
        if constexpr (GROUP_WARPS == 4) {
            auto dst_subtile = dst.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*warpid(), 0);
            ::kittens::group<1>::store_async(dst_subtile, src);
        }
        else {
            auto dst_subtile = dst.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*(warpid()%4)+16*(warpid()/4), 0);
            ::kittens::group<1>::store_async(dst_subtile, src);
        }
    }
}