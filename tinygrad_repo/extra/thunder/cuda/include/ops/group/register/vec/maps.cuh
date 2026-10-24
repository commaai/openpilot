/**
 * @file
 * @brief Maps on vectors stored in registers.
 */

/* ----------  Vector Maps  ---------- */

/**
 * @brief Perform a unary operation on a vector.
 *
 * @tparam op The unary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector to perform the operation on.
 */
template<typename op, ducks::rv::all T>
__device__ static inline void unary_op(T &dst, const T &src) {
    #pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j]);
        }
    }
}
/**
 * @brief Perform a binary operation on two vectors.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vectors.
 * @param dst[out] The destination vector where the result is stored.
 * @param lhs[in] The left-hand side vector for the operation.
 * @param rhs[in] The right-hand side vector for the operation.
 */
template<typename op, ducks::rv::all T>
__device__ static inline void bin_op(T &dst, const T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(lhs[i][j], rhs[i][j]);
        }
    }
}
/**
 * @brief Perform a binary operation on a vector and a scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The scalar parameter for the operation.
 */
template<typename op, ducks::rv::all T>
__device__ static inline void bin_op(T &dst, const T &src, const typename T::dtype &param) {
    #pragma unroll
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma unroll
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename T::dtype>(src[i][j], param);
        }
    }
}
/**
 * @brief Perform a binary operation on a vector and an unpacked scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The unpacked scalar parameter for the operation.
 */
template<typename op, ducks::rv::tile_layout T>
__device__ static inline void bin_op(T &dst, const T &src, const typename base_types::packing<typename T::dtype>::unpacked_type &param) {
    bin_op<op, T>(dst, src, base_types::packing<typename T::dtype>::pack(param));
}


template<ducks::rv::all RV, typename Lambda>
__device__ static inline void apply(RV &dst, const RV &src, Lambda &&lambda) {
    int group_offset = 0;
    if constexpr(GROUP_WARPS > 1) {
        group_offset = warpid()*RV::length;
    }
    static_assert(sizeof(RV::T) != 1, "Cannot apply lambda to 8-bit types");
    if constexpr (ducks::rv::ortho_layout<RV>) {
        #pragma unroll
        for(int i = 0; i < dst.outer_dim; i++) {
            int base_idx = group_offset + i*16 + ::kittens::laneid()/4;
            dst[i][0].x = lambda(base_idx+0, src[i][0].x);
            dst[i][0].y = lambda(base_idx+8, src[i][0].y);
        }
    }
    else if constexpr (ducks::rv::align_layout<RV>) {
        #pragma unroll
        for(int i = 0; i < dst.outer_dim; i++) {
            int base_idx = group_offset + i*16 + 2*(::kittens::laneid()%4);
            dst[i][0].x = lambda(base_idx+0, src[i][0].x);
            dst[i][0].y = lambda(base_idx+1, src[i][0].y);
            dst[i][1].x = lambda(base_idx+8, src[i][1].x);
            dst[i][1].y = lambda(base_idx+9, src[i][1].y);
        }
    }
    else {
        #pragma unroll
        for(int i = 0; i < dst.outer_dim; i++) {
            int base_idx = group_offset + i*32 + ::kittens::laneid();
            if (i < dst.outer_dim-1 || dst.length%32 == 0 || ::kittens::laneid()<16) {
                dst[i][0] = lambda(base_idx, src[i][0]);
            }
        }
    }
}
template<ducks::rv::all RV, typename Lambda>
__device__ static inline RV apply(const RV &src, Lambda &&lambda) {
    RV dst;
    apply<RV, Lambda>(dst, src, std::forward<Lambda>(lambda));
    return dst;
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

/**
 * @brief Sets all elements of a register vector to zero.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to zero.
 */
template<ducks::rv::all T>
__device__ static inline void zero(T &dst) {
    unary_op<base_ops::zero, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to one.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to one.
 */
template<ducks::rv::all T>
__device__ static inline void one(T &dst) {
    unary_op<base_ops::one, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to positive infinity.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to positive infinity.
 */
template<ducks::rv::all T>
__device__ static inline void pos_infty(T &dst) {
    unary_op<base_ops::pos_infty, T>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to negative infinity.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to negative infinity.
 */
template<ducks::rv::all T>
__device__ static inline void neg_infty(T &dst) {
    unary_op<base_ops::neg_infty, T>(dst, dst);
}

// ---- unary ops ----

/**
 * @brief Copies the elements from one register vector to another.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the source vector.
 * @param dst[out] Destination vector where the elements will be copied to.
 * @param src[in] Source vector to copy the elements from.
 */
template<ducks::rv::all T, typename U>
__device__ static inline void copy(T &dst, const U &src) {
    bin_op<base_ops::copy2, T>(dst, dst, src); // the second arg is ignored here.
}
/**
 * @brief Applies the exponential function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::rv::all T>
__device__ static inline void exp(T &dst, const T &src) {
    unary_op<base_ops::exp, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T exp(const T &src) {
    T dst;
    exp(dst, src);
    return dst;
}
/**
 * @brief Applies the exponential function element-wise to a register vector, in base 2.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::rv::all T>
__device__ static inline void exp2(T &dst, const T &src) {
    unary_op<base_ops::exp2, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T exp2(const T &src) {
    T dst;
    exp2(dst, src);
    return dst;
}
/**
 * @brief Applies the natural logarithm function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<ducks::rv::all T>
__device__ static inline void log(T &dst, const T &src) {
    unary_op<base_ops::log, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T log(const T &src) {
    T dst;
    log(dst, src);
    return dst;
}
/**
 * @brief Applies the logarithm base 2 function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the logarithm base 2 function to.
 */
template<ducks::rv::all T>
__device__ static inline void log2(T &dst, const T &src) {
    unary_op<base_ops::log2, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T log2(const T &src) {
    T dst;
    log2(dst, src);
    return dst;
}
/**
 * @brief Applies the absolute value function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the absolute values will be stored.
 * @param src[in] Source vector to apply the absolute value function to.
 */
template<ducks::rv::all T>
__device__ static inline void abs(T &dst, const T &src) {
    unary_op<base_ops::abs, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T abs(const T &src) {
    T dst;
    abs(dst, src);
    return dst;
}
/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the ReLU values will be stored.
 * @param src[in] Source vector to apply the ReLU function to.
 */
template<ducks::rv::all T>
__device__ static inline void relu(T &dst, const T &src) {
    unary_op<base_ops::relu, T>(dst, src);
}
template<ducks::rv::all T>
__device__ static inline T relu(const T &src) {
    T dst;
    relu(dst, src);
    return dst;
}

// ---- binary ops ----

/**
 * @brief Computes the element-wise maximum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the maximum values will be stored.
 * @param lhs[in] First vector for the maximum operation.
 * @param rhs[in] Second vector for the maximum operation.
 */
template<ducks::rv::all T, typename U>
__device__ static inline void max(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::max, T>(dst, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T max(const T &lhs, const U &rhs) {
    T dst;
    max(dst, lhs, rhs);
    return dst;
}
/**
 * @brief Computes the element-wise minimum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the minimum values will be stored.
 * @param lhs[in] First vector for the minimum operation.
 * @param rhs[in] Second vector for the minimum operation.
 */
template<ducks::rv::all T, typename U>
__device__ static inline void min(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::min, T>(dst, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T min(const T &lhs, const U &rhs) {
    T dst;
    min(dst, lhs, rhs);
    return dst;
}
/**
 * @brief Computes the element-wise sum of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the sum values will be stored.
 * @param lhs[in] First vector for the sum operation.
 * @param rhs[in] Second vector for the sum operation.
 */
template<ducks::rv::all T, typename U>
__device__ static inline void add(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sum, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise difference of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the difference values will be stored.
 * @param lhs[in] First vector for the difference operation.
 * @param rhs[in] Second vector for the difference operation.
 */
template<ducks::rv::all T, typename U>
__device__ static inline void sub(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::sub, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise product of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the product values will be stored.
 * @param lhs[in] First vector for the product operation.
 * @param rhs[in] Second vector for the product operation.
 */
template<ducks::rv::all T, typename U>
__device__ static inline void mul(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::mul, T>(dst, lhs, rhs);
}
/**
 * @brief Computes the element-wise division of two register vectors.
 *
 * @tparam T Register vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the division values will be stored.
 * @param lhs[in] First vector for the division operation.
 * @param rhs[in] Second vector for the division operation.
 */
template<ducks::rv::all T, typename U>
__device__ static inline void div(T &dst, const T &lhs, const U &rhs) {
    bin_op<base_ops::div, T>(dst, lhs, rhs);
}