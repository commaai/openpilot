/**
 * @file
 * @brief Conversions between data layouts and types for complex register tiles.
 */

/* ----------  LAYOUT SWAPS  ---------- */

/**
 * @brief Swaps the layout of a complex register tile.
 *
 * This function swaps the layout of a complex register tile by 
 * swapping the real and imaginary component tiles' layouts
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(crt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const crt<T2, _height, _width, layout> &src) {
    swap_layout(dst.real, src.real);
    swap_layout(dst.real, src.real);
}
/**
 * @brief Swaps the layout of a complex register tile in place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline crt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(crt<T2, _height, _width, layout> &tile) {
    tile.real = swap_layout_inplace(tile.real);
    tile.imag = swap_layout_inplace(tile.imag);
    return tile;
}

/* ----------  TRANSPOSE  ---------- */

/**
 * @brief Transposes a complex register tile.
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
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void transpose_sep(crt<T2, _width, _height, layout> &dst, const crt<T2, _height, _width, layout> &src) {
    transpose_sep(dst.real, src.real);
    transpose_sep(dst.imag, src.imag);
}
/**
 * @brief Transposes a square complex register tile in-place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height (in units of 16) of the src register tile, and the width of the dst tile. (Must be the same as _width.)
 * @tparam _width The width (in units of 16) of the src register tile, and the height of the dst tile. (Must be the same as _height.)
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register tile.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline crt<T2, _height, _width, layout>& transpose_inplace(crt<T2, _height, _width, layout> &tile) {
    tile.real = transpose_inplace(tile.real);
    tile.imag = transpose_inplace(tile.imag);

    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief Copies a complex register tile, converting the underlying type if necessary.
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
__device__ static inline void copy(crt<T2, _height, _width, layout> &dst, const crt<U2, _height, _width, layout> &src) {
    copy(dst.real, src.real);
    copy(dst.imag, src.imag);
}