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
template<ducks::crt::all RT, ducks::cst::all ST>
__device__ inline static void load(RT &dst, const ST &src) {
    load(dst.real, src.real);
    load(dst.imag, src.imag);
}


/**
 * @brief Collaboratively store data into a shared tile from register tiles split across a warpgroup.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::cst::all ST, ducks::crt::all RT>
__device__ inline static void store(ST &dst, const RT &src) {
    store(dst.real, src.real);
    store(dst.imag, src.imag);
}

