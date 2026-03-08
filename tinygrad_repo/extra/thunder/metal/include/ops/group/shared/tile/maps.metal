/**  
 * @file
 * @brief Group maps on shared tiles.
 */

/**
 * @brief Performs a uniform unary operation on a tile.
 *
 * This function applies a given unary operation to each element of the source tile and stores the result in the destination tile.
 * The operation is applied independently to each element, without considering its position or the values of neighboring elements.
 *
 * @tparam op The unary operation to be applied. Must be specialized to support operation on the data type of T.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the unary operation is applied.
 */
template<typename op, typename ST> // T2, w, h can be inferred from dst as long as op is specialized
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 unary_map(threadgroup ST &dst, threadgroup const ST &src, const int threadIdx) {
    #pragma clang loop unroll(full)
    for(int i = laneid(threadIdx); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename ST::dtype>(src.data[i]);
    }
}

/**
 * @brief Performs a uniform binary operation on a tile with a scalar parameter.
 *
 * This function applies a given binary operation to each element of the source tile and a scalar parameter, then stores the result in the destination tile.
 * The operation is applied independently to each element, treating the scalar parameter as the second operand for each operation.
 *
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the scalar parameter.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] param The scalar parameter to be used as the second operand in the binary operation.
 */
template<typename op, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 bin_map(threadgroup ST &dst, threadgroup const ST &src, thread const typename ST::dtype &param, const int threadIdx) {
    #pragma clang loop unroll(full)
    for(int i = laneid(threadIdx); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename ST::dtype>(src.data[i], param);
    }
}

/**
 * @brief Performs a uniform binary operation on two tiles.
 *
 * This function applies a given binary operation to corresponding elements of two source tiles and stores the result in the destination tile.
 * The operation is applied independently to each pair of elements, without considering their positions or the values of neighboring elements.
 *
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile to which the binary operation is applied.
 * @param[in] rhs The second source tile to which the binary operation is applied.
 */
template<typename op, typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 bin_map(threadgroup ST &dst, threadgroup const ST &lhs, threadgroup const ST &rhs, const int threadIdx) {
    #pragma clang loop unroll(full)
    for(int i = laneid(threadIdx); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename ST::dtype>(lhs.data[i], rhs.data[i]);
    }
}

/**
 * @brief Performs a row-wise binary operation on a tile with a vector.
 *
 * This function applies a given binary operation to each row of the source tile and the corresponding element of the source vector,
 * then stores the result in the destination tile. The operation is applied independently to each row, using the vector element as
 * the second operand for each element in the row.
 *
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the vector elements.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @tparam V The type of the vector. Must have the same data type as T.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] vec The source vector containing the second operand for each row operation.
 */
template<typename op, typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
row_map(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &vec, const int threadIdx) {
    static_assert(metal::is_same<typename ST::dtype, typename SV::dtype>::value, "Tile and vector must have the same data type");
    static_assert(SV::length == ST::rows, "Vector length must match the number of rows in the tile");
    #pragma clang loop unroll(full)
    for(int i = laneid(threadIdx); i < dst.num_elements; i += GROUP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = op::template op<typename ST::dtype>(src[{row, col}], vec[row]);
    }
}

/**
 * @brief Performs a column-wise binary operation on a tile with a vector.
 *
 * This function applies a given binary operation to each column of the source tile and the corresponding element of the source vector,
 * then stores the result in the destination tile. The operation is applied independently to each column, using the vector element as
 * the second operand for each element in the column.
 *
 * @tparam op The binary operation to be applied. Must be specialized to support operation on the data type of T and the vector elements.
 * @tparam T The type of the tiles. Must satisfy the `ducks::st::all` concept.
 * @tparam V The type of the vector. Must have the same data type as T.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the binary operation is applied.
 * @param[in] vec The source vector containing the second operand for each column operation.
 */
template<typename op, typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 col_map(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &vec, const int threadIdx) {
    static_assert(metal::is_same<typename ST::dtype, typename SV::dtype>::value, "Tile and vector must have the same data type");
    static_assert(SV::length == ST::cols, "Vector length must match the number of columns in the tile");
    #pragma clang loop unroll(full)
    for(int i = laneid(threadIdx); i < dst.num_elements; i += GROUP_THREADS) {
        int row = i/dst.cols, col = i%dst.cols;
        dst[{row, col}] = op::template op<typename ST::dtype>(src[{row, col}], vec[col]);
    }
}


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// All of the annoying qualifiers *should* be automatically inferred during compile-time.
// So, syntax should just be mittens::add_row(tile, colvec);

// const maps
/**
 * @brief Sets all elements of the destination tile to zero.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 zero(threadgroup ST &dst, const int threadIdx) {
    unary_map<base_ops::zero, ST>(dst, dst, threadIdx);
}
/**
 * @brief Sets all elements of the destination tile to one.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 one(threadgroup ST &dst, const int threadIdx) {
    unary_map<base_ops::one, ST>(dst, dst, threadIdx);
}
/**
 * @brief Sets all elements of the destination tile to positive infinity.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 pos_infty(threadgroup ST &dst, const int threadIdx) {
    unary_map<base_ops::pos_infty, ST>(dst, dst, threadIdx);
}
/**
 * @brief Sets all elements of the destination tile to negative infinity.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 neg_infty(threadgroup ST &dst, const int threadIdx) {
    unary_map<base_ops::neg_infty, ST>(dst, dst, threadIdx);
}

// unary maps
/**
 * @brief Applies the exponential function to each element of the source tile and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the exponential function is applied.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 exp(threadgroup ST &dst, threadgroup const ST &src, const int threadIdx) {
    unary_map<base_ops::exp, ST>(dst, src, threadIdx);
}
/**
 * @brief Applies the exponential function to each element of the source tile and stores the result in the destination tile, in base 2.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the exponential function is applied.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 exp2(threadgroup ST &dst, threadgroup const ST &src, const int threadIdx) {
    unary_map<base_ops::exp2, ST>(dst, src, threadIdx);
}
/**
 * @brief Applies the natural logarithm function to each element of the source tile and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the natural logarithm function is applied.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
log(threadgroup ST &dst, threadgroup const ST &src, const int threadIdx) {
    unary_map<base_ops::log, ST>(dst, src, threadIdx);
}
/**
 * @brief Applies the absolute function to each element of the source tile and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the absolute function is applied.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
abs(threadgroup ST &dst, threadgroup const ST &src, const int threadIdx) {
    unary_map<base_ops::abs, ST>(dst, src, threadIdx);
}
/**
 * @brief Applies the rectified linear unit function to each element of the source tile and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the rectified linear unit function is applied.
 */
template<typename ST>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
relu(threadgroup ST &dst, threadgroup const ST &src, const int threadIdx) {
    unary_map<base_ops::relu, ST>(dst, src, threadIdx);
}
/**
 * @brief Copies the elements of the source tile to the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source data to be copied.
 */
template<typename ST, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 copy(threadgroup ST &dst, thread const U &src, const int threadIdx) {
    bin_map<base_ops::copy, ST>(dst, src, threadIdx);
}

// uniform binary maps
/**
 * @brief Finds the maximum of each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<typename ST, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 max(threadgroup ST &dst, threadgroup const ST &lhs, thread const U &rhs, const int threadIdx) {
    bin_map<base_ops::max, ST>(dst, lhs, rhs, threadIdx);
}
/**
 * @brief Finds the minimum of each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<typename ST, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 min(threadgroup ST &dst, threadgroup const ST &lhs, thread const U &rhs, const int threadIdx) {
    bin_map<base_ops::min, ST>(dst, lhs, rhs, threadIdx);
}
/**
 * @brief Adds each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<typename ST, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 add(threadgroup ST &dst, threadgroup const ST &lhs, thread const U &rhs, const int threadIdx) {
    bin_map<base_ops::sum, ST>(dst, lhs, rhs, threadIdx);
}
/**
 * @brief Subtracts each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<typename ST, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 sub(threadgroup ST &dst, threadgroup const ST &lhs, thread const U &rhs, const int threadIdx) {
    bin_map<base_ops::sub, ST>(dst, lhs, rhs, threadIdx);
}
/**
 * @brief Multiplies each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<typename ST, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
 mul(threadgroup ST &dst, threadgroup const ST &lhs, thread const U &rhs, const int threadIdx) {
    bin_map<base_ops::mul, ST>(dst, lhs, rhs, threadIdx);
}
/**
 * @brief Divides each pair of corresponding elements in the two source tiles and stores the result in the destination tile.
 *
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @tparam U The type of the second source data. Must be convertible to the data type of the destination tile.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] lhs The first source tile.
 * @param[in] rhs The second source data.
 */
template<typename ST, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>(), void>::type
div(threadgroup ST &dst, threadgroup const ST &lhs, thread const U &rhs, const int threadIdx) {
    bin_map<base_ops::div, ST>(dst, lhs, rhs, threadIdx);
}

// Row and col maps

/**
 * @brief Adds row values to each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param row_values[in] Column vector containing values to add to each row.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 add_row(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &row_values, const int threadIdx) {
    row_map<base_ops::sum, ST, SV>(dst, src, row_values, threadIdx);
}
/**
 * @brief Subtracts row values from each row of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param row_values[in] Column vector containing values to subtract from each row.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 sub_row(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &row_values, const int threadIdx) {
    row_map<base_ops::sub, ST, SV>(dst, src, row_values, threadIdx);
}
/**
 * @brief Multiplies each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param row_values[in] Column vector containing values to multiply each row by.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 mul_row(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &row_values, const int threadIdx) {
    row_map<base_ops::mul, ST, SV>(dst, src, row_values, threadIdx);
}
/**
 * @brief Divides each row of a tile by row values.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param row_values[in] Column vector containing values to divide each row by.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 div_row(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &row_values, const int threadIdx) {
    row_map<base_ops::div, ST, SV>(dst, src, row_values, threadIdx);
}
/**
 * @brief Broadcast a vector into into a tile's rows.
 *
 * @tparam T Tile type.
 * @tparam V Column vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Column vector containing values to broadcast into rows.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 broadcast_row(threadgroup ST &dst, threadgroup const SV &row_values, const int threadIdx) {
    row_map<base_ops::copy2, ST, SV>(dst, dst, row_values, threadIdx);
}


// col maps
/**
 * @brief Adds column values to each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the addition on.
 * @param col_values[in] Row vector containing values to add to each column.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 add_col(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &col_values, const int threadIdx) {
    col_map<base_ops::sum, ST, SV>(dst, src, col_values, threadIdx);
}
/**
 * @brief Subtracts column values from each column of a tile.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the subtraction on.
 * @param col_values[in] Row vector containing values to subtract from each column.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 sub_col(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &col_values, const int threadIdx) {
    col_map<base_ops::sub, ST, SV>(dst, src, col_values, threadIdx);
}
/**
 * @brief Multiplies each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the multiplication on.
 * @param col_values[in] Row vector containing values to multiply each column by.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 mul_col(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &col_values, const int threadIdx) {
     col_map<base_ops::mul, ST, SV>(dst, src, col_values, threadIdx);
}
/**
 * @brief Divides each column of a tile by column values.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param src[in] Source tile to apply the division on.
 * @param col_values[in] Row vector containing values to divide each column by.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 div_col(threadgroup ST &dst, threadgroup const ST &src, threadgroup const SV &col_values, const int threadIdx) {
    col_map<base_ops::div, ST, SV>(dst, src, col_values, threadIdx);
}
/**
 * @brief Broadcast a vector into into a tile's columns.
 *
 * @tparam T Tile type.
 * @tparam V Row vector type.
 * @param dst[out] Destination tile where the result is stored.
 * @param row_values[in] Row vector containing values to broadcast into cols.
 */
template<typename ST, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_tile<ST>() && ducks::is_shared_vector<SV>(), void>::type
 broadcast_col(threadgroup ST &dst, threadgroup const SV &col_values, const int threadIdx) {
    col_map<base_ops::copy2, ST, SV>(dst, dst, col_values, threadIdx);
}
