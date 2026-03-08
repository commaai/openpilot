/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly between global memory and registers and back.
 */

/**
 * @brief Collaboratively loads data from a source array into register tiles.
 *
 * @tparam RT The register tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    load<axis, CRT::component, CGL::component, COORD>(dst.real, src.real, idx);
    load<axis, CRT::component, CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void load(CRT &dst, const CGL &src, const COORD &idx) {
    load<2, CRT, CGL>(dst, src, idx);
}

/**
 * @brief Collaboratively stores data from register tiles to a destination array in global memory.
 *
 * @tparam RT The register tile type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    store<axis, typename CRT::component, typename CGL::component>(dst.real, src.real, idx);
    store<axis, typename CRT::component, typename CGL::component>(dst.imag, src.imag, idx);
}
template<ducks::crt::all CRT, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<crt<typename CRT::T, GROUP_WARPS*CRT::rows, CRT::cols, typename CRT::layout>>>
__device__ inline static void store(CGL &dst, const CRT &src, const COORD &idx) {
    store<2, CRT, CGL>(dst, src, idx);
}
