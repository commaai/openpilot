/**
 * @file
 * @brief Map operations between complex tiles.
 */

/**
 * @brief Sets all elements of a complex tile to zero.
 *
 * @tparam T Complex tile type.
 * @param dst[out] Destination tile where the result is stored.
 */
template<ducks::crt::all T>
__device__ static inline void zero(T &dst) {
    zero(dst.real);
    zero(dst.imag);
}
/**
 * @brief Applies the exponential function to each element of a complex tile.
 *
 * @tparam T Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the exponential function on.
 */
template<ducks::crt::all T>
__device__ static inline void exp(T &dst, const T &src) {
    using dtype = T::dtype;
    dtype tmp;
    // out of place storage
    dtype rdst;
    dtype idst;

    // exp(a)
    exp(rdst, src.real);
    copy(idst, rdst);
    // exp(a)cos(b) + exp(a)sin(b)i
    cos(tmp, src.imag);
    mul(rdst, rdst, tmp);
    sin(tmp, src.imag);
    mul(idst, idst, tmp);

    copy(dst.real, rdst);
    copy(dst.imag, idst);
}
/**
 * @brief Adds two complex tiles element-wise.
 *
 * @tparam T Complex Tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the addition.
 * @param rhs[in] Right-hand side source tile for the addition.
 */
template<ducks::crt::all T>
__device__ static inline void add(T &dst, const T &lhs, const T &rhs) {
    add(dst.real, lhs.real, rhs.real);
    add(dst.imag, lhs.imag, rhs.imag);
}
/**
 * @brief Subtracts two tiles element-wise.
 *
 * @tparam T Tile type.
 * @tparam U Second operand type, which can be a tile or a scalar.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the subtraction.
 * @param rhs[in] Right-hand side source tile for the subtraction.
 */
template<ducks::crt::all T>
__device__ static inline void sub(T &dst, const T &lhs, const T &rhs) {
    sub(dst.real, lhs.real, rhs.real);
    sub(dst.imag, lhs.imag, rhs.imag);
}
/**
 * @brief Multiplies two tiles element-wise.
 *
 * @tparam T Complex tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the multiplication.
 * @param rhs[in] Right-hand side source tile for the multiplication.
 */
template<ducks::crt::all T>
__device__ static inline void mul(T &dst, const T &lhs, const T &rhs) {
    using dtype = T::component;
    dtype tmp;
    // out of place storage regs
    dtype rdst;
    dtype idst;
    
    // (a + bi) * (c + di) --> (ac - bd) + (ad + bc)i
    // Real component
    mul(rdst, lhs.real, rhs.real);  
    mul(tmp, lhs.imag, rhs.imag);    
    sub(rdst, rdst, tmp);

    // Imag component
    mul(idst, lhs.imag, rhs.real);
    mul(tmp, lhs.real, rhs.imag);
    add(idst, idst, tmp);

    copy(dst.real, rdst);
    copy(dst.imag, idst);
}
/**
 * @brief Divides two tiles element-wise.
 *
 * @tparam T Complex tile type.
 * @param dst[out] Destination tile where the result is stored.
 * @param lhs[in] Left-hand side source tile for the division.
 * @param rhs[in] Right-hand side source tile or scalar for the division.
 */
template<ducks::crt::all T>
__device__ static inline void div(T &dst, const T &lhs, const T &rhs) {
    using dtype = T::dtype;
    dtype tmp;
    dtype denom;
    // out of place storage regs
    dtype rdst;
    dtype idst;

    // Calculate denom - square of b terms
    mul(tmp, rhs.real, rhs.real);
    mul(denom, rhs.imag, rhs.imag);
    add(denom, tmp, denom);
    // Real component
    mul(rdst, lhs.real, rhs.real);
    mul(tmp, lhs.imag, rhs.imag);
    add(rdst, rdst, tmp);
    // Imag component
    mul(dst.imag, lhs.imag, rhs.real);
    mul(tmp, lhs.real, rhs.imag);
    sub(idst, idst, tmp);
    // Divide components by denom
    div(rdst, rdst, denom);
    div(idst, idst, denom);
    copy(dst.real, rdst);
    copy(dst.imag, idst);
}


