/**
 * @file
 * @brief Group reductions on shared tiles.
 */

/**
 * Performs row-wise reduction on a matrix using a specified operation.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type with row layout.
 * @param row_accum The accumulator where the result of the reduction is stored.
 * @param src The source matrix on which to perform the reduction.
 * @param src_accum The initial value of the accumulator, used when reset is false.
 * @param reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 */
template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    using dtype = typename V::dtype;
    for (int row = laneid(); row < src.rows; row += GROUP_THREADS) {
        dtype accum = src[{row, 0}];
        #pragma unroll
        for (int col = 1; col < src.cols; col++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            row_accum[row] = accum;
        } else {
            row_accum[row] = op::template op<dtype>(src_accum[row], accum);
        }
    }
}

/**
 * Performs column-wise reduction on a matrix using a specified operation.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The shared vector type for the column accumulator.
 * @tparam T The shared matrix type with column layout.
 * @param col_accum The accumulator where the result of the reduction is stored.
 * @param src The source matrix on which to perform the reduction.
 * @param src_accum The initial value of the accumulator, used when reset is false.
 * @param reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 */
template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    using dtype = typename V::dtype;
    for (int col = laneid(); col < src.cols; col += GROUP_THREADS) {
        dtype accum = src[{0, col}];
        #pragma unroll
        for (int row = 1; row < src.rows; row++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        if (reset) {
            col_accum[col] = accum;
        } else {
            col_accum[col] = op::template op<dtype>(src_accum[col], accum);
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Store the maximum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the minimum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the sum of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the product of each row of the src shared matrix in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}

/**
 * @brief Store the maximum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the sum of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the product of each row of the src shared matrix, as well as the src_accum shared vector, in the row_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

/**
 * @brief Store the maximum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the minimum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the sum of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the product of each column of the src shared matrix in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}

/**
 * @brief Store the maximum of each column of the src shared matrix, as well as the src_accum shared vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each column of the src shared matrix, as well as the src_accum shared vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the sum of each column of the src shared tile, as well as the src_accum row vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the product of each column of the src shared tile, as well as the src_accum row vector, in the col_accum shared vector.
 *
 * @tparam V The shared vector type for the row accumulator.
 * @tparam T The shared matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::sv::all V, ducks::st::all T>
__device__ static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

// templated versions of each

template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void max(V &dst, const T &src, const V &src_accum) {
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline auto max(const T &src, const V &src_accum) {
    V dst;
    if constexpr (ax == axis::COL) row_max(dst, src, src_accum);
    else col_max(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void max(V &dst, const T &src) {
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
}
template<int ax, ducks::st::all T>
__device__ static inline auto max(const T &src) {
    using V = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    V dst;
    if constexpr (ax == axis::COL) row_max(dst, src);
    else col_max(dst, src);
    return dst;
}

template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void min(V &dst, const T &src, const V &src_accum) {
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline auto min(const T &src, const V &src_accum) {
    V dst;
    if constexpr (ax == axis::COL) row_min(dst, src, src_accum);
    else col_min(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void min(V &dst, const T &src) {
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
}
template<int ax, ducks::st::all T>
__device__ static inline auto min(const T &src) {
    using V = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    V dst;
    if constexpr (ax == axis::COL) row_min(dst, src);
    else col_min(dst, src);
    return dst;
}

template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void sum(V &dst, const T &src, const V &src_accum) {
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline auto sum(const T &src, const V &src_accum) {
    V dst;
    if constexpr (ax == axis::COL) row_sum(dst, src, src_accum);
    else col_sum(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void sum(V &dst, const T &src) {
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
}
template<int ax, ducks::st::all T>
__device__ static inline auto sum(const T &src) {
    using V = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    V dst;
    if constexpr (ax == axis::COL) row_sum(dst, src);
    else col_sum(dst, src);
    return dst;
}

template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void prod(V &dst, const T &src, const V &src_accum) {
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline auto prod(const T &src, const V &src_accum) {
    V dst;
    if constexpr (ax == axis::COL) row_prod(dst, src, src_accum);
    else col_prod(dst, src, src_accum);
    return dst;
}
template<int ax, ducks::sv::all V, ducks::st::all T>
__device__ static inline void prod(V &dst, const T &src) {
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
}
template<int ax, ducks::st::all T>
__device__ static inline auto prod(const T &src) {
    using V = std::conditional_t<ax==axis::COL, typename T::col_vec, typename T::row_vec>;
    V dst;
    if constexpr (ax == axis::COL) row_prod(dst, src);
    else col_prod(dst, src);
    return dst;
}