/**
 * @file
 * @brief Warp-scope maps on shared vectors.
 */

#pragma once

#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {

/**
 * @brief Applies a unary operation to each element of a shared memory vector.
 *
 * @tparam op Unary operation type.
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector in which to store the result.
 * @param src[in] Source vector to apply the unary operation.
 */ 
template<typename op, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
unary_op(threadgroup SV &dst, threadgroup const SV &src, const ushort laneid) {
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for(int cur = laneid; cur < SV::length; cur+=SIMD_THREADS) {
        dst[cur] = op::template op<typename SV::dtype>(src[cur]);
    }
}
/**
 * @brief Perform a binary operation on two shared vectors.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vectors.
 * @param dst[out] The destination vector where the result is stored.
 * @param lhs[in] The left-hand side vector for the operation.
 * @param rhs[in] The right-hand side vector for the operation.
 */
template<typename op, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
bin_op(threadgroup SV &dst, threadgroup const SV &lhs, threadgroup const SV &rhs, const ushort laneid) {
    #pragma clang loop unroll(full)
    for(int cur = laneid; cur < SV::length; cur+=SIMD_THREADS) {
        dst[cur] = op::template op<typename SV::dtype>(lhs[cur], rhs[cur]);
    }
}
/**
 * @brief Perform a binary operation on a shared vector and a scalar.
 *
 * @tparam op The binary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector for the operation.
 * @param param[in] The scalar parameter for the operation.
 */
template<typename op, typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
bin_op(threadgroup SV &dst, threadgroup const SV &src, thread const typename SV::T &param, const ushort laneid) {
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for(int cur = laneid; cur < SV::length; cur+=SIMD_THREADS) {
        dst[cur] = op::template op<typename SV::dtype>(src[cur], param);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

/**
 * @brief Sets all elements of a shared memory vector to zero.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to zero.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
zero(threadgroup SV &dst, const ushort laneid) {
    unary_op<base_ops::zero, SV>(dst, dst, laneid);
}
/**
 * @brief Sets all elements of a shared memory vector to one.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to one.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
one(threadgroup SV &dst, const ushort laneid) {
    unary_op<base_ops::one, SV>(dst, dst, laneid);
}
/**
 * @brief Sets all elements of a shared memory vector to positive infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to positive infinity.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
pos_infty(threadgroup SV &dst, const ushort laneid) {
    unary_op<base_ops::pos_infty, SV>(dst, dst, laneid);
}
/**
 * @brief Sets all elements of a shared memory vector to negative infinity.
 *
 * @tparam T Shared memory vector type.
 * @param dst[out] Destination vector to be set to negative infinity.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
neg_infty(threadgroup SV &dst, const ushort laneid) {
    unary_op<base_ops::neg_infty, SV>(dst, dst, laneid);
}

// ---- unary ops ----

/**
 * @brief Copies the elements from one shared vector to another.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the source vector.
 * @param dst[out] Destination vector where the elements will be copied to.
 * @param src[in] Source vector to copy the elements from.
 */
template<typename SV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
copy(threadgroup SV &dst, thread const U &src, const ushort laneid) {
    bin_op<base_ops::copy2, SV>(dst, dst, src, laneid); // the second arg is ignored here.
}
/**
 * @brief Applies the exponential function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
exp(threadgroup SV &dst, threadgroup const SV &src, const ushort laneid) {
    unary_op<base_ops::exp, SV>(dst, src, laneid);
}
/**
 * @brief Applies the exponential function element-wise to a shared vector, in base 2.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
exp2(threadgroup SV &dst, threadgroup const SV &src, const ushort laneid) {
    unary_op<base_ops::exp2, SV>(dst, src, laneid);
}
/**
 * @brief Applies the natural logarithm function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the logarithm values will be stored.
 * @param src[in] Source vector to apply the logarithm function to.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
log(threadgroup SV &dst, threadgroup const SV &src, const ushort laneid) {
    unary_op<base_ops::log, SV>(dst, src, laneid);
}
/**
 * @brief Applies the absolute value function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the absolute values will be stored.
 * @param src[in] Source vector to apply the absolute value function to.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
abs(threadgroup SV &dst, threadgroup const SV &src, const ushort laneid) {
    unary_op<base_ops::abs, SV>(dst, src, laneid);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a shared vector.
 *
 * @tparam T Shared vector type.
 * @param dst[out] Destination vector where the ReLU values will be stored.
 * @param src[in] Source vector to apply the ReLU function to.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
relu(threadgroup SV &dst, threadgroup const SV &src, const ushort laneid) {
    unary_op<base_ops::relu, SV>(dst, src, laneid);
}

// ---- binary ops ----

/**
 * @brief Computes the element-wise maximum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the maximum values will be stored.
 * @param lhs[in] First vector for the maximum operation.
 * @param rhs[in] Second vector for the maximum operation.
 */
template<typename SV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
max(threadgroup SV &dst, threadgroup const SV &lhs, thread const U &rhs, const ushort laneid) {
    bin_op<base_ops::max, SV>(dst, lhs, rhs, laneid);
}
/**
 * @brief Computes the element-wise minimum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the minimum values will be stored.
 * @param lhs[in] First vector for the minimum operation.
 * @param rhs[in] Second vector for the minimum operation.
 */
template<typename SV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
min(threadgroup SV &dst, threadgroup const SV &lhs, thread const U &rhs, const ushort laneid) {
    bin_op<base_ops::min, SV>(dst, lhs, rhs, laneid);
}
/**
 * @brief Computes the element-wise sum of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the sum values will be stored.
 * @param lhs[in] First vector for the sum operation.
 * @param rhs[in] Second vector for the sum operation.
 */
template<typename SV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
add(threadgroup SV &dst, threadgroup const SV &lhs, thread const U &rhs, const ushort laneid) {
    bin_op<base_ops::sum, SV>(dst, lhs, rhs, laneid);
}
/**
 * @brief Computes the element-wise difference of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the difference values will be stored.
 * @param lhs[in] First vector for the difference operation.
 * @param rhs[in] Second vector for the difference operation.
 */
template<typename SV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
sub(threadgroup SV &dst, threadgroup const SV &lhs, thread const U &rhs, const ushort laneid) {
    bin_op<base_ops::sub, SV>(dst, lhs, rhs, laneid);
}
/**
 * @brief Computes the element-wise product of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the product values will be stored.
 * @param lhs[in] First vector for the product operation.
 * @param rhs[in] Second vector for the product operation.
 */
template<typename SV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
mul(threadgroup SV &dst, threadgroup const SV &lhs, thread const U &rhs, const ushort laneid) {
    bin_op<base_ops::mul, SV>(dst, lhs, rhs, laneid);
}
/**
 * @brief Computes the element-wise division of two shared vectors.
 *
 * @tparam T Shared vector type.
 * @tparam U Type of the second vector.
 * @param dst[out] Destination vector where the division values will be stored.
 * @param lhs[in] First vector for the division operation.
 * @param rhs[in] Second vector for the division operation.
 */
template<typename SV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
div(threadgroup SV &dst, threadgroup const SV &lhs, thread const U &rhs, const ushort laneid) {
    bin_op<base_ops::div, SV>(dst, lhs, rhs, laneid);
}

}
