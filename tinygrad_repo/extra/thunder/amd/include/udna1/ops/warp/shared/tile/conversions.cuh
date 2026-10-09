/**
 * @file
 * @brief Conversions between shared tile types.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given shared tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam subtile_width The width of the subtile.
* @tparam ST The type of the input tile, which must satisfy the ducks::st::all concept.
* @param src The input tile.
* @param row_idx The row coord of the subtile, in units of subtile_height*16 elements.
* @param col_idx The col coord of the subtile, in units of subtile_width*16 elements.
* @return A reference to the subtile.
*
* @note The subtile {height, width} must evenly divide the tile {height, width}.
*/
template<int subtile_rows, int subtile_cols, ducks::st::all ST>
__device__ inline st_subtile<ST, subtile_rows, subtile_cols> subtile_inplace(ST &src, int2 rowcol) {
    using T = typename ST::dtype;
    static_assert(ST::rows % subtile_rows == 0);
    static_assert(ST::cols % subtile_cols == 0);
    static_assert(ST::rows == ST::underlying_rows && ST::cols == ST::underlying_cols); // must be a real ST, no recursive subtiles.
    return st_subtile<ST, subtile_rows, subtile_cols>(src, rowcol);
}

} // namespace kittens