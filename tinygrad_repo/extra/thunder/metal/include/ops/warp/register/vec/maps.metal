/**
 * @file
 * @brief Maps on vectors stored in registers.
 */

#pragma once // doneington

#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {

/* ----------  Vector Maps  ---------- */

/**
 * @brief Perform a unary operation on a vector.
 *
 * @tparam op The unary operation to perform.
 * @tparam T The type of the vector.
 * @param dst[out] The destination vector where the result is stored.
 * @param src[in] The source vector to perform the operation on.
 */
template<typename op, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
unary_op(thread RV &dst, thread const RV &src) {
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename RV::dtype>(src[i][j]);
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
template<typename op, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
bin_op(thread RV &dst, thread const RV &lhs, thread const RV &rhs) {
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.outer_dim; i++) {
        #pragma clang loop unroll(full)
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename RV::dtype>(lhs[i][j], rhs[i][j]);
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
template<typename op, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
bin_op(thread RV &dst, thread const RV &src, thread const typename RV::dtype &param) {
    #pragma clang loop unroll(full)
    for(int i = 0; i < dst.outer_dim; i++) {
    #pragma clang loop unroll(full)
        for(int j = 0; j < dst.inner_dim; j++) {
            dst[i][j] = op::template op<typename RV::dtype>(src[i][j], param);
        }
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// ---- const ops ----

/**
 * @brief Sets all elements of a register vector to zero.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to zero.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
zero(thread RV &dst) {
    unary_op<base_ops::zero, RV>(dst, dst);
}
    
/**
 * @brief Sets all elements of a register vector to one.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to one.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
one(thread RV &dst) {
    unary_op<base_ops::one, RV>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to positive infinity.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to positive infinity.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
pos_infty(thread RV &dst) {
    unary_op<base_ops::pos_infty, RV>(dst, dst);
}
/**
 * @brief Sets all elements of a register vector to negative infinity.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector to be set to negative infinity.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
neg_infty(thread RV &dst) {
    unary_op<base_ops::neg_infty, RV>(dst, dst);
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
template<typename RV, typename U>
    static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>() && ducks::base_types::isT1Type<U>(), void>::type
copy(thread RV &dst, thread const U &src) {
    bin_op<base_ops::copy2, RV>(dst, dst, src); // the second arg is ignored here.
}
/**
 * @brief Applies the exponential function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
exp(thread RV &dst, thread const RV &src) {
    unary_op<base_ops::exp, RV>(dst, src);
}
/**
 * @brief Applies the exponential function element-wise to a register vector, in base 2.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
exp2(thread RV &dst, thread const RV &src) {
    unary_op<base_ops::exp2, RV>(dst, src);
}
/**
 * @brief Applies the natural logarithm function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the exponential values will be stored.
 * @param src[in] Source vector to apply the exponential function to.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
log(thread RV &dst, thread const RV &src) {
    unary_op<base_ops::log, RV>(dst, src);
}
/**
 * @brief Applies the absolute value function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the absolute values will be stored.
 * @param src[in] Source vector to apply the absolute value function to.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
abs(thread RV &dst, thread const RV &src) {
    unary_op<base_ops::abs, RV>(dst, src);
}
/**
 * @brief Applies the rectified linear unit (ReLU) function element-wise to a register vector.
 *
 * @tparam T Register vector type.
 * @param dst[out] Destination vector where the ReLU values will be stored.
 * @param src[in] Source vector to apply the ReLU function to.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
relu(thread RV &dst, thread const RV &src) {
    unary_op<base_ops::relu, RV>(dst, src);
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
template<typename RV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
max(thread RV &dst, thread const RV &lhs, thread const U &rhs) {
    bin_op<base_ops::max, RV>(dst, lhs, rhs);
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
template<typename RV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
min(thread RV &dst, thread const RV &lhs, thread const U &rhs) {
    bin_op<base_ops::min, RV>(dst, lhs, rhs);
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
template<typename RV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
add(thread RV &dst, thread const RV &lhs, thread const U &rhs) {
    bin_op<base_ops::sum, RV>(dst, lhs, rhs);
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
template<typename RV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
sub(thread RV &dst, thread const RV &lhs, thread const U &rhs) {
    bin_op<base_ops::sub, RV>(dst, lhs, rhs);
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
template<typename RV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
mul(thread RV &dst, thread const RV &lhs, thread const U &rhs) {
    bin_op<base_ops::mul, RV>(dst, lhs, rhs);
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
template<typename RV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
div(thread RV &dst, thread const RV &lhs, thread const U &rhs) {
    bin_op<base_ops::div, RV>(dst, lhs, rhs);
}
}

