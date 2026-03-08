/**
 * @file
 * @brief Group conversions between different shared memory tile types.
 */

/* ----------  COPIES  ---------- */

/**
 * @brief Copies data from one shared memory tile to another, potentially with different data types and layouts.
 *
 * @tparam T The data type of the destination tile.
 * @tparam U The data type of the source tile.
 * @tparam _height The height of the tile.
 * @tparam _width The width of the tile.
 * @tparam L1 The layout of the destination tile.
 * @tparam L2 The layout of the source tile.
 * @param[out] dst The destination tile.
 * @param[in] src The source tile.
 */
template<typename T, typename U, int _height, int _width>
static METAL_FUNC void copy(threadgroup st<T, _height, _width> &dst, threadgroup const st<U, _height, _width> &src, const int threadIdx) {
    #pragma clang loop unroll(full)
    for(int i = laneid(threadIdx); i < dst.num_elements; i+=GROUP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = base_types::convertor<T, U>::convert(src[{row, col}]);
    }
}
