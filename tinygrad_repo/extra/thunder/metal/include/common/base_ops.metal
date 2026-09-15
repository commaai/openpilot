/**
 * @file
 * @brief Basic operations on generic types.
 */
#pragma once
#include "base_types.metal"
#include <metal_math>

namespace mittens {
/**
 * @namespace base_ops
 *
 * @brief A namespace for operations on basic data types.
 */
namespace base_ops {
#define TEMPLATE_OPS_SINGLE(func_contents) \
    template<typename T> static METAL_FUNC T op(device const T &x)      { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &x) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread const T &x)      { func_contents }

#define TEMPLATE_OPS_OVERRIDE_SINGLE(T, op_name, func_contents) \
    template<> METAL_FUNC T op_name::op<T>(device const T &x)      { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &x) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread const T &x)      { func_contents }

#define TEMPLATE_OPS_DOUBLE(func_contents) \
    template<typename T> static METAL_FUNC T op(device      const T &a, device      const T &b) { func_contents } \
    template<typename T> static METAL_FUNC T op(device      const T &a, threadgroup const T &b) { func_contents } \
    template<typename T> static METAL_FUNC T op(device      const T &a, thread      const T &b) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, device      const T &b) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, threadgroup const T &b) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, thread      const T &b) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, device      const T &b) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, threadgroup const T &b) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, thread      const T &b) { func_contents }

#define TEMPLATE_OPS_OVERRIDE_DOUBLE(T, op_name, func_contents) \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, device      const T &b) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, threadgroup const T &b) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, thread      const T &b) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, device      const T &b) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, threadgroup const T &b) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, thread      const T &b) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, device      const T &b) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, threadgroup const T &b) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, thread      const T &b) { func_contents }
    
#define TEMPLATE_OPS_TRIPLE(func_contents) \
    template<typename T> static METAL_FUNC T op(device      const T &a, device      const T &b, device      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(device      const T &a, device      const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(device 	    const T &a, device      const T &b, thread      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(device      const T &a, threadgroup const T &b, device      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(device 	    const T &a, threadgroup const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(device      const T &a, threadgroup const T &b, thread      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(device      const T &a, thread      const T &b, device      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(device      const T &a, thread      const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(device      const T &a, thread      const T &b, thread      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, device      const T &b, device      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, device      const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, device      const T &b, thread      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, threadgroup const T &b, device      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, threadgroup const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, threadgroup const T &b, thread      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, thread      const T &b, device      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, thread      const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &a, thread      const T &b, thread      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, device      const T &b, device 	    const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread 	    const T &a, device      const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, device      const T &b, thread      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, threadgroup const T &b, device      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, threadgroup const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, threadgroup const T &b, thread      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, thread      const T &b, device      const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, thread      const T &b, threadgroup const T &c) { func_contents } \
    template<typename T> static METAL_FUNC T op(thread      const T &a, thread      const T &b, thread      const T &c) { func_contents }

#define TEMPLATE_OPS_OVERRIDE_TRIPLE(T, op_name, func_contents) \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, device      const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, device      const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, device      const T &b, thread      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, threadgroup const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device 	   const T &a, threadgroup const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, threadgroup const T &b, thread      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, thread      const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, thread      const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(device      const T &a, thread      const T &b, thread      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, device      const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, device      const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, device      const T &b, thread      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, threadgroup const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, threadgroup const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, threadgroup const T &b, thread      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, thread      const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, thread      const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(threadgroup const T &a, thread      const T &b, thread      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, device      const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, device      const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, device      const T &b, thread      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, threadgroup const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, threadgroup const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, threadgroup const T &b, thread      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, thread      const T &b, device      const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, thread      const T &b, threadgroup const T &c) { func_contents } \
    template<> METAL_FUNC T op_name::op<T>(thread      const T &a, thread      const T &b, thread      const T &c) { func_contents }



/* ----------  CONST OPS  ---------- */

/**
 * @brief Represents the zero constant operation.
 *
 * This operation returns the zero value of the specified type.
 *
 * @tparam T The data type for which to return the zero value.
 * @return The zero value of type T.
 */
struct zero {
    template<typename T, typename... args> static METAL_FUNC constexpr T op(args... _) { return base_types::constants<T>::zero();      }
};
/**
 * @brief Represents the one constant operation.
 *
 * This operation returns the one value of the specified type.
 *
 * @tparam T The data type for which to return the one value.
 * @return The one value of type T.
 */
struct one {
    template<typename T, typename... args> static METAL_FUNC constexpr T op(args... _) { return base_types::constants<T>::one();       }
};
    
/**
 * @brief Represents the positive infinity constant operation.
 *
 * This operation returns the positive infinity value of the specified type.
 *
 * @tparam T The data type for which to return the positive infinity value.
 * @return The positive infinity value of type T.
 */
struct pos_infty {
    template<typename T, typename... args> static METAL_FUNC constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
};
/**
 * @brief Represents the negative infinity constant operation.
 *
 * This operation returns the negative infinity value of the specified type.
 *
 * @tparam T The data type for which to return the negative infinity value.
 * @return The negative infinity value of type T.
 */
struct neg_infty {
    template<typename T, typename... args> static METAL_FUNC constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
};
    

/* ----------  UNARY OPS  ---------- */
/**
 * @brief Exponential function operation.
 *
 * This operation calculates the exponential of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The exponential of the input value.
 */
struct exp {
    TEMPLATE_OPS_SINGLE(return metal::exp(x);)
};

TEMPLATE_OPS_OVERRIDE_SINGLE(bf16, exp, return bf16(metal::exp((float)x));)
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16_2, exp, return bf16_2(metal::exp(float2(x)));)
    
    /**
 * @brief Exponential function operation, in base 2
 *
 * This operation calculates the exponential of the input value, in base 2.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The exponential of the input value.
 */
struct exp2 {
    template<typename T> static METAL_FUNC T op(device const T &x)      { return metal::exp2(x); } \
    template<typename T> static METAL_FUNC T op(threadgroup const T &x) { return metal::exp2(x); } \
    template<typename T> static METAL_FUNC T op(thread const T &x)      { return metal::exp2(x); }
};
    
//template<> METAL_FUNC bf16 exp2::op<bf16>(device const bf16 &x)      { return bf16(metal::exp2(x)); } \
//template<> METAL_FUNC bf16 exp2::op<bf16>(threadgroup const bf16 &x) { return bf16(metal::exp2(x)); } \
//template<> METAL_FUNC bf16 exp2::op<bf16>(thread const bf16 &x)      { return bf16(metal::exp2(x)); }
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16, exp2, return bf16(metal::exp2(x));)
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16_2, exp2, return bf16_2(metal::exp2((float2)x));)

/**
 * @brief Natural log function operation.
 *
 * This operation calculates the natural logarithm of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The natural logarithm of the input value.
 */
struct log {
    TEMPLATE_OPS_SINGLE(return metal::log(x);)
};
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16, log, return bf16(metal::log(x));)
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16_2, log, return bf16_2(metal::log((float2)x));)

/**
 * @brief Absolute value operation.
 *
 * This operation calculates the absolute value of the input.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The absolute value of the input.
 */
struct abs {
    TEMPLATE_OPS_SINGLE(return metal::abs(x);)
};
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16  , abs, return bf16(metal::abs((float)x));)
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16_2, abs, return bf16_2(metal::abs((float2)x));)
/**
 * @brief Rectified Linear Unit (ReLU) operation.
 *
 * This operation applies the ReLU function to the input, which is the
 * maximum of zero and the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The result of ReLU function applied to the input.
 */
struct relu {
    TEMPLATE_OPS_SINGLE(return max(x, base_types::constants<T>::zero());)
};
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16  , relu, return bf16(metal::max((float)x, base_types::constants<float>::zero()));)
TEMPLATE_OPS_OVERRIDE_SINGLE(bf16_2, relu, return bf16_2(metal::max((float2)x, base_types::constants<float2>::zero()));)
/**
 * @brief Copy operation.
 *
 * This operation returns the input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The input value.
 * @return The same value as the input.
 */
struct copy { // for non-compile-time setters.
    TEMPLATE_OPS_SINGLE(return x;)
};

/* ----------  BINARY OPS  ---------- */
  
    
/**
 * @brief Copy2 operation.
 *
 * This operation returns the second input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value (ignored).
 * @param b[in] The second input value.
 * @return The same value as the second input.
 */
struct copy2 { // this turns out to be a slightly hacky op that makes some code cleaner :/
    TEMPLATE_OPS_DOUBLE(return b;)
};
/**
 * @brief Sum operation.
 *
 * This operation calculates the sum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The sum of the input values.
 */
struct sum {
    TEMPLATE_OPS_DOUBLE(return a+b;)
};
    
/**
 * @brief Subtraction operation.
 *
 * This operation calculates the difference between two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The difference between the input values.
 */
struct sub {
    TEMPLATE_OPS_DOUBLE(return a-b;)
};
/**
 * @brief Multiplication operation.
 *
 * This operation calculates the product of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The product of the input values.
 */
struct mul {
    TEMPLATE_OPS_DOUBLE(return a*b;)
};
/**
 * @brief Division operation.
 *
 * This operation calculates the quotient of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The quotient of the input values.
 */
struct div {
    TEMPLATE_OPS_DOUBLE(return a/b;)
};
/**
 * @brief Maximum operation.
 *
 * This operation calculates the maximum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The maximum of the input values.
 */
struct max {
    TEMPLATE_OPS_DOUBLE(return metal::max(a,b);)
};
TEMPLATE_OPS_OVERRIDE_DOUBLE(bf16  , max, return (bf16)metal::max((float)a, (float)b);)
TEMPLATE_OPS_OVERRIDE_DOUBLE(bf16_2, max, return (bf16_2)metal::max((float2)a, (float2)b);)
/**
 * @brief Minimum operation.
 *
 * This operation calculates the minimum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The minimum of the input values.
 */
struct min {
    TEMPLATE_OPS_DOUBLE(return metal::min(a,b);)
};
TEMPLATE_OPS_OVERRIDE_DOUBLE(bf16  , min, return (bf16)metal::min((float)a, (float)b);)
TEMPLATE_OPS_OVERRIDE_DOUBLE(bf16_2, min, return (bf16_2)metal::min((float2)a, (float2)b);)


/* ----------  TERNARY OPS  ---------- */
/**
 * @brief Fused multiply-add operation A * B + C.
 *
 * This operation performs a fused multiply-add, computing (A * B) + C with only one rounding.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @param c[in] The third input value to be added.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxBtC {
    TEMPLATE_OPS_TRIPLE(return sum::op<T>(mul::op<T>(a, b), c);)
};
    
/**
 * @brief Fused multiply-add operation A * C + B.
 *
 * This operation performs a fused multiply-add, computing (A * C) + B with only one rounding.
 * This is particularly useful for attention mechanisms in neural networks.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value. 
 * @param b[in] The third input value to be added.
 * @param c[in] The second input value.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxCtB { // this is the one needed for attention
    TEMPLATE_OPS_TRIPLE(return sum::op<T>(mul::op<T>(a, c), b);)
};

#undef TEMPLATE_OPS_SINGLE
#undef TEMPLATE_OPS_OVERRIDE_SINGLE
#undef TEMPLATE_OPS_DOUBLE
#undef TEMPLATE_OPS_OVERRIDE_DOUBLE
#undef TEMPLATE_OPS_TRIPLE
#undef TEMPLATE_OPS_OVERRIDE_TRIPLE
} // base_ops
} // mittens
