/**
 * @file
 * @brief Reductions on vectors stored in registers.
 */

#pragma once // done

#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {
/* ----------  Vector Reductions  ---------- */

/**
 * @brief Performs a reduction operation on elements of a register vector within a warp.
 *
 * This function applies a specified operation to reduce the elements of a register vector `src` to a single value.
 * The result is stored in `accum`. If the `reset` parameter is true, the reduction includes an initial value `src_accum`.
 * The reduction operation is performed in a warp-wide context, ensuring synchronization between threads in the warp.
 *
 * @tparam op The operation to perform on the elements. Must provide a static `op` method.
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @tparam reset A boolean flag indicating whether to include an initial value in the reduction.
 * @param[out] accum The result of the reduction operation.
 * @param[in] src The register vector to reduce.
 * @param[in] src_accum The initial value to include in the reduction if `reset` is false.
 */
template<typename op, typename RV, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
reduce(
       thread typename RV::T &dst_accum,
       thread const RV &src,
       thread const typename RV::T &src_accum,
       const ushort laneid) {
    using T = typename RV::T;
   if (ducks::is_ortho_register_vector<RV>()) { // col vector
        T accum = src[0][0];
        #pragma clang loop unroll(full)
        for(int i = 1; i < src.outer_dim; i++) {
            accum = op::template op<T>(accum, src[i][0]);
        }
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 2));
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 4));
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 16));
        if (!reset) accum = op::template op<T>(accum, src_accum);
        dst_accum = shfl_sync(accum, 0);
    }
    else if (ducks::is_align_register_vector<RV>()) { // row vector
        T accum = op::template op<T>(src[0][0], src[0][1]);
        #pragma clang loop unroll(full)
        for(int i = 1; i < src.outer_dim; i++) {
            accum = op::template op<T>(accum, src[i][0]);
            accum = op::template op<T>(accum, src[i][1]);
        }
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 1));
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 8));
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
        
        accum = shfl_sync<T>(accum, 0);
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
        if (!reset) accum = op::template op<T>(accum, src_accum);
        dst_accum = accum;
    }
    else if (ducks::is_naive_register_vector<RV>()) {
//        T accum = src[0][0];
        T accum;
        if (laneid < src.length) accum = src[0][0];
        #pragma clang loop unroll(full)
        for(int i = 1; i < src.outer_dim; i++) {
            if (i*SIMD_THREADS + laneid < src.length) {
                accum = op::template op<T>(accum, src[i][0]);
            }
        }
        if (src.length == 8) {
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 1));
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 2));
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 4));
        } else if (src.length == 16) {
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 1));
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 2));
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 4));
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 8));
        } else if (src.length == 24) {
            if (laneid < 24) {
                accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 1));
                accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 2));
                accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 4));
                
                T shfle_val = shfl_down_sync<T>(accum, 8);
                if (laneid < 16) {
                    accum = op::template op<T>(accum, shfle_val);
                }
                metal::simdgroup_barrier(metal::mem_flags::mem_none);
                accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 16));
            }
            
        } else {
            metal::simdgroup_barrier(metal::mem_flags::mem_none);
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 1));
            metal::simdgroup_barrier(metal::mem_flags::mem_none);
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 2));
            metal::simdgroup_barrier(metal::mem_flags::mem_none);
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 4));
            metal::simdgroup_barrier(metal::mem_flags::mem_none);
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 8));
            metal::simdgroup_barrier(metal::mem_flags::mem_none);
            accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 16));
            metal::simdgroup_barrier(metal::mem_flags::mem_none);
        }
        
        if (!reset) accum = op::template op<T>(accum, src_accum);
        dst_accum = shfl_sync(accum, 0);
    }
}
    
/**
 * @brief Finds the maximum element in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] max_val The maximum value found in the vector.
 * @param[in] src The register vector to find the maximum in.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
max(thread typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, thread const RV &src, const ushort laneid) {
    reduce<base_ops::max, RV, true>(max_val, src, max_val, laneid);
}

/**
 * @brief Finds the minimum element in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] min_val The minimum value found in the vector.
 * @param[in] src The register vector to find the minimum in.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
min(thread typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, thread const RV &src, const ushort laneid) {
    reduce<base_ops::min, RV, true>(min_val, src, min_val, laneid);
}

/**
 * @brief Calculates the sum of elements in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] sum_val The sum of the values in the vector.
 * @param[in] src The register vector to sum.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
sum(thread typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, thread const RV &src, const ushort laneid) {
    reduce<base_ops::sum, RV, true>(sum_val, src, sum_val, laneid);
}

/**
 * @brief Calculates the product of elements in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] prod_val The product of the values in the vector.
 * @param[in] src The register vector to multiply.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
prod(thread typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, thread const RV &src, const ushort laneid) {
    reduce<base_ops::mul, RV, true>(prod_val, src, prod_val, laneid);
}

// Three operand versions.

/**
 * @brief Finds the maximum element in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] max_val The maximum value found in the vector, accumulated with src_accum.
 * @param[in] src The register vector to find the maximum in.
 * @param[in] src_accum The initial value to accumulate with the maximum value found.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
max(thread typename base_types::packing<typename RV::dtype>::unpacked_type &max_val,
    thread const RV &src,
    thread const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum, const ushort laneid) {
    reduce<base_ops::max, RV, false>(max_val, src, src_accum, laneid);
}

/**
 * @brief Finds the minimum element in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] min_val The minimum value found in the vector, accumulated with src_accum.
 * @param[in] src The register vector to find the minimum in.
 * @param[in] src_accum The initial value to accumulate with the minimum value found.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
min(thread typename base_types::packing<typename RV::dtype>::unpacked_type &min_val,
    thread const RV &src,
    thread const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum, const ushort laneid) {
    reduce<base_ops::min, RV, false>(min_val, src, src_accum, laneid);
}

/**
 * @brief Calculates the sum of elements in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] sum_val The sum of the values in the vector, accumulated with src_accum.
 * @param[in] src The register vector to sum.
 * @param[in] src_accum The initial value to accumulate with the sum of the vector.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
sum(thread typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val,
    thread const RV &src,
    thread const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum, const ushort laneid) {
    reduce<base_ops::sum, RV, false>(sum_val, src, src_accum, laneid);
}

/**
 * @brief Calculates the product of elements in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] prod_val The product of the values in the vector, accumulated with src_accum.
 * @param[in] src The register vector to multiply.
 * @param[in] src_accum The initial value to accumulate with the product of the vector.
 */
template<typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
prod(thread typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val,
     thread const RV &src,
     thread const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum, const ushort laneid) {
    reduce<base_ops::mul, RV, false>(prod_val, src, src_accum, laneid);
}

}
