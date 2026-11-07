/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

/* ----------  LAYOUT SWAPS  ---------- */

/**
 * @brief Perform a matrix transpose on a block of 8 bf16_2 elements using inline assembly.
 *
 * This low-level operation is utilized by higher-level layout swap functions to transpose
 * the layout of bf16_2 elements within a register tile. The function leverages inline PTX
 * assembly to efficiently swap the layout of the given block.
 *
 * @param[out] dst A reference to the destination bf16_2 element where the transposed result is stored.
 * @param[in] src A reference to the source bf16_2 element to be transposed.
 */
__device__ static inline void swap_layout_8(bf16_2 &dst, const bf16_2 &src) {
    KITTENS_CHECK_WARP
    asm volatile (
        "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n"
    :   "+r"(*(uint32_t*)(&dst))
    :   "r"(*(uint32_t*)(&src))
    );
}
/**
 * @brief Swaps the layout of a register base tile.
 *
 * This function swaps the layout of a register base tile by performing a series of layout swaps
 * on its constituent bf16_2 elements. It is used to change the data layout within a register tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register base tile where the result will be stored.
 * @param src[in] Reference to the source register base tile to be swapped.
 */
template<typename T, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);
    // technically this swap can be eliminated if we simply reinterpret the layout of the registers
    // everywhere else in the code, but that feels... very likely to cause bugs and not worth it. 
    typename rt_base<T, layout>::T2 data1_cache = src.data[1]; // important for swap!
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
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
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
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
template<typename T2, ducks::rt_layout::all layout>
__device__ static inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}
/**
 * @brief Swaps the layout of a register tile in place.
 *
 * This function swaps the layout of a register tile in place by iterating over its height and width
 * and performing in-place layout swaps on each of its base elements.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _rows, _cols, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
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
template<typename T, ducks::rt_layout::all layout>
__device__ static inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    swap_layout_8(dst.data[0], src.data[0]);
    // technically this swap can be eliminated if we simply reinterpret the layout of the registers
    // everywhere else in the code, but that feels... very likely to cause bugs and not worth it. 
    typename rt_base<T, layout>::T2 data1_cache = src.data[1]; // important for swap!
    swap_layout_8(dst.data[1], src.data[2]);
    swap_layout_8(dst.data[2], data1_cache);
    swap_layout_8(dst.data[3], src.data[3]);
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
template<ducks::rt::all RT>
__device__ static inline void transpose_sep(RT &dst, const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src) {
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            transpose(dst.tiles[i][j], src.tiles[j][i]);
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
template<typename T2, ducks::rt_layout::all layout>
__device__ static inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}
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
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, layout>& transpose_inplace(rt<T2, _rows, _cols, layout> &tile) {
    static_assert(_cols == _rows, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
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
template<typename T, typename U, ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;
    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
    }
}
#ifdef KITTENS_HOPPER
/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {

    if constexpr (
        (std::is_same_v<U2, float> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, float> && std::is_same_v<T2, fp8e5m2>) ||
        (std::is_same_v<U2, kittens::bf16> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, kittens::bf16> && std::is_same_v<T2, fp8e5m2>) ||
        (std::is_same_v<U2, half> && std::is_same_v<T2, fp8e4m3>) ||
        (std::is_same_v<U2, half> && std::is_same_v<T2, fp8e5m2>)
    ) {
        // FLOAT (SRC -- 1H x 2W) to FP8 (DST -- 1H x 1W)
        int laneid = threadIdx.x % 32;

        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                #pragma unroll
                for(int k = 0; k < dst.tiles[0][0].packed_per_thread; k++) {
                    
                    // check for half, float, bf16
                    using src_t = std::conditional_t<std::is_same_v<U2, float>, float2, std::conditional_t<std::is_same_v<U2, kittens::bf16>, bf16_2, half2>>;
                    src_t val1, val2;

                    // Put something up for adoption
                    if (laneid % 2 == 0) { 
                        // put up src left core matrix first as 0, 2
                        val1 = src.tiles[i][2*j + k/2].data[(k%2)+0];
                        val2 = src.tiles[i][2*j + k/2].data[(k%2)+2];
                    } else { 
                        // put up src right core matrix first as 1, 3
                        val1 = src.tiles[i][2*j + k/2].data[(k%2)+2];
                        val2 = src.tiles[i][2*j + k/2].data[(k%2)+0];
                    }

                    // Shuffle first 4 floats
                    int row_mask = 4 * ( laneid / 4 );
                    int row_offset = row_mask + ( (laneid-row_mask) / 2 ) + ( laneid % 2 );
                    int src_offset = (laneid % 2 == 0 ) ? row_offset + 0 : ( row_offset + 1 );
                    src_t val01 = packed_shfl_sync(MASK_ALL, val1, src_offset);  // Get from even thread

                    int src_offset2 = (laneid % 4 < 2 ) ? src_offset + 1 : (src_offset - 1);
                    src_t val23 = packed_shfl_sync(MASK_ALL, val2, src_offset2);  // Get from odd thread
                    
                    // Convert to fp8e4m3_4
                    float4 f4;
                    using fp8_4_t = std::conditional_t<std::is_same_v<T2, fp8e4m3>, fp8e4m3_4, fp8e5m2_4>;
                    fp8_4_t f4_fp8;
                    if ( laneid % 4 < 2 ) { 
                        f4.x = val01.x;  // Thread 2N's first value
                        f4.y = val01.y;  // Thread 2N's second value
                        f4.z = val23.x;  // Thread 2N+1's first value
                        f4.w = val23.y;  // Thread 2N+1's second value
                        f4_fp8 = base_types::convertor<fp8_4_t, float4>::convert(f4);
                        dst.tiles[i][j].data[k] = f4_fp8;
                    } else {
                        f4.x = val23.x;  // Thread 2N+1's first value
                        f4.y = val23.y;  // Thread 2N+1's second value
                        f4.z = val01.x;  // Thread 2N's first value
                        f4.w = val01.y;  // Thread 2N's second value
                        f4_fp8 = base_types::convertor<fp8_4_t, float4>::convert(f4);
                        dst.tiles[i][j].data[k] = f4_fp8;
                    }
                }
            }
        }
    }
    else if constexpr (
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, float>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, float>) ||
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, kittens::bf16>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, kittens::bf16>) ||
        (std::is_same_v<U2, fp8e4m3> && std::is_same_v<T2, half>) ||
        (std::is_same_v<U2, fp8e5m2> && std::is_same_v<T2, half>)
    ) {
        // FP8 (SRC -- 1H x 1W) to FLOAT (DST -- 1H x 2W)
        int laneid = threadIdx.x % 32;

        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                #pragma unroll
                for(int k = 0; k < src.tiles[0][0].packed_per_thread; k++) {
                    int dst_j = 2*j + k/2;

                    // Put something up for adoption
                    using fp8_4_t = std::conditional_t<std::is_same_v<U2, fp8e4m3>, fp8e4m3_4, fp8e5m2_4>;
                    fp8_4_t val = src.tiles[i][j].data[k];
                    float4 f4 = base_types::convertor<float4, fp8_4_t>::convert(val);
                    float2 f2_0, f2_1;
                    if ( laneid % 4 < 2 ) { // src 0 and 1 should put up .x and .y first
                        f2_0 = make_float2(f4.x, f4.y);
                        f2_1 = make_float2(f4.z, f4.w);
                    }
                    else { // src 2 and 3 should put up .z and .w first
                        f2_0 = make_float2(f4.z, f4.w);
                        f2_1 = make_float2(f4.x, f4.y);
                    }

                    int row_offset = 4 * (laneid/4) + (laneid%2) * 2 + (laneid%4) / 2;
                    float2 f2_0_shfl = packed_shfl_sync(MASK_ALL, f2_0, row_offset);
                    float2 f2_1_shfl = packed_shfl_sync(MASK_ALL, f2_1, row_offset^2);

                    // convert to dst type if needed
                    using dst_t = std::conditional_t<std::is_same_v<T2, float>, float2, std::conditional_t<std::is_same_v<T2, kittens::bf16>, bf16_2, half2>>;
                    if constexpr (!(std::is_same_v<T2, float>)) {
                        dst_t f2_0_shfl_t = base_types::convertor<dst_t, float2>::convert(f2_0_shfl);
                        dst_t f2_1_shfl_t = base_types::convertor<dst_t, float2>::convert(f2_1_shfl);
                        if (laneid % 2 == 0) {  
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_0_shfl_t;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_1_shfl_t;
                        } else {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_1_shfl_t;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_0_shfl_t;
                        }
                    } else {
                        if (laneid % 2 == 0) {  
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_0_shfl;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_1_shfl;
                        } else {
                            dst.tiles[i][dst_j].data[(k%2)+0] = f2_1_shfl;
                            dst.tiles[i][dst_j].data[(k%2)+2] = f2_0_shfl;
                        }
                    }
                }
            }
        }
    }
    // default case where the layouts map 1:1 in thread ownership logic
    else {
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                copy(dst.tiles[i][j], src.tiles[i][j]);
            }
        }
    }
}
#else
/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}
#endif

/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The coord of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
template<int subtile_rows, ducks::rt::all RT>
__device__ static inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    KITTENS_CHECK_WARP
    using T = typename RT::T;
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx*(subtile_rows / TILE_ROW_DIM<T>)]
    );
}