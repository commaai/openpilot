/**
 * @file
 * @brief Maps on vectors stored in registers.
 */

 #pragma once

 #include "../../../../../common/common.cuh"
 #include "../../../../../types/types.cuh"
 
 namespace kittens {
 
 /* ----------  Vector Maps  ---------- */
 
 /**
  * @brief Computes the element-wise product of two register vectors.
  *
  * @tparam T Register vector type.
  * @tparam U Type of the second vector.
  * @param dst[out] Destination vector where the product values will be stored.
  * @param lhs[in] First vector for the product operation.
  * @param rhs[in] Second vector for the product operation.
  */
 template<int GPR0, int GPR1, typename U>
 __device__ static inline void mul(const U &rhs) {
    macros::mul::template op<GPR0, GPR1>(rhs);
 }

 template<int GPR0>
 __device__ static inline void zero() {
   macros::zero::template op<GPR0, GPR0>();
 }

 
 }