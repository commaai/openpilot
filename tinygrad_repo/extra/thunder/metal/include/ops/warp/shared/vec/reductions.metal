/**
 * @file
 * @brief Warp-scope maps on shared vectors.
 */

#pragma once

#include "../../../../common/common.metal"
#include "../../../../types/types.metal"

namespace mittens {

/**
 * @brief Performs a reduction operation on elements of a shared memory vector within a warp.
 *
 * This function applies a specified operation to reduce the elements of a shared memory vector `src` to a single value.
 * The result is stored in `accum`. If the `reset` parameter is true, the reduction includes an initial value `src_accum`.
 * The reduction operation is performed in a warp-wide context, ensuring synchronization between threads in the warp.
 *
 * @tparam op The operation to perform on the elements. Must provide a static `op` method.
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @tparam reset A boolean flag indicating whether to include an initial value in the reduction.
 * @param[out] accum The result of the reduction operation.
 * @param[in] src The shared memory vector to reduce.
 * @param[in] src_accum The initial value to include in the reduction if `reset` is false.
 */
template<typename op, typename SV, bool reset>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
reduce(thread typename SV::dtype &dst_accum, threadgroup const SV &src, thread const typename SV::dtype &src_accum, const ushort laneid) {
    using T = typename SV::dtype;
    
    {
        T accum = src[0];
        for (int i = 1; i < SV::length; i++) {
            accum = op::template op<T>(accum, src[i]);
        }
        dst_accum = shfl_sync(accum, 0);
        return;
    }

//    
    T accum;
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = op::template op<T>(accum, src[i]);
    }
    if (src.length >= 32) {
//        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 1));
        accum = op::template op<T>(accum, (T)metal::simd_shuffle_rotate_down((float)accum, 1));
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
//        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 2));
        accum = op::template op<T>(accum, (T)metal::simd_shuffle_rotate_down((float)accum, 2));
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
//        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 4));
        accum = op::template op<T>(accum, (T)metal::simd_shuffle_rotate_down((float)accum, 4));
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
//        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 8));
        accum = op::template op<T>(accum, (T)metal::simd_shuffle_rotate_down((float)accum, 8));
        metal::simdgroup_barrier(metal::mem_flags::mem_none);
//        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 16));
        accum = op::template op<T>(accum, (T)metal::simd_shuffle_rotate_down((float)accum, 16));
        
    } else if (src.length == 24) {
        T shfl_val = shfl_down_sync<T>(accum, 1);
        accum = op::template op<T>(accum, shfl_val);
        
        shfl_val = shfl_down_sync<T>(accum, 2);
        accum = op::template op<T>(accum, shfl_val);
        
        shfl_val = shfl_down_sync<T>(accum, 4);
        accum = op::template op<T>(accum, shfl_val);
        
        shfl_val = shfl_down_sync<T>(accum, 8);
        if (laneid < 16) {
            accum = op::template op<T>(accum, shfl_val);
        }
        shfl_val = shfl_down_sync<T>(accum, 16);
        accum = op::template op<T>(accum, shfl_val);
    } else if (src.length == 16) {
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 1));
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 2));
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 4));
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 8));
    } else if (src.length == 8) {
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 1));
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 2));
        accum = op::template op<T>(accum, shfl_down_sync<T>(accum, 4));
    }
    if (!reset) accum = op::template op<T>(accum, src_accum);
    dst_accum = shfl_sync(accum, 0);
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Finds the maximum element in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] max_val The maximum value found in the vector.
 * @param[in] src The shared memory vector to find the maximum in.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
max(thread typename SV::dtype &max_val, threadgroup const SV &src, const ushort laneid) {
//    reduce<base_ops::max, SV, true>(max_val, src, max_val, laneid);
    using T = typename SV::dtype;
    T accum = base_types::constants<T>::neg_infty();
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = base_ops::max::template op<T>(accum, src[i]);
    }
    max_val = (T)metal::simd_max((float)accum);
}

/**
 * @brief Finds the minimum element in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] min_val The minimum value found in the vector.
 * @param[in] src The shared memory vector to find the minimum in.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
min(thread typename SV::dtype &min_val, threadgroup const SV &src, const ushort laneid) {
//    reduce<base_ops::min, SV, true>(min_val, src, min_val);
    
    using T = typename SV::dtype;
    T accum = base_types::constants<T>::pos_infty();
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = base_ops::min::template op<T>(accum, src[i]);
    }
    min_val = (T)metal::simd_min((float)accum);
}

/**
 * @brief Calculates the sum of elements in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] sum_val The sum of the values in the vector.
 * @param[in] src The shared memory vector to sum.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
sum(thread typename SV::dtype &sum_val, threadgroup const SV &src, const ushort laneid) {
//    reduce<base_ops::sum, SV, true>(sum_val, src, sum_val, laneid);
    using T = typename SV::dtype;
    T accum = base_types::constants<T>::zero();
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = base_ops::min::template op<T>(accum, src[i]);
    }
    sum_val = (T)metal::simd_sum((float)accum);
}

/**
 * @brief Calculates the product of elements in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] prod_val The product of the values in the vector.
 * @param[in] src The shared memory vector to multiply.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
prod(thread typename SV::dtype &prod_val, threadgroup const SV &src, const ushort laneid) {
//    reduce<base_ops::mul, SV, true>(prod_val, src, prod_val, laneid);
    using T = typename SV::dtype;
    T accum = base_types::constants<T>::one();
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = base_ops::min::template op<T>(accum, src[i]);
    }
    prod_val = (T)metal::simd_product((float)accum);
}

// Three operand versions.

/**
 * @brief Finds the maximum element in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] max_val The maximum value found in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to find the maximum in.
 * @param[in] src_accum The initial value to accumulate with the maximum value found.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
max(thread typename SV::dtype &max_val, threadgroup const SV &src, thread const typename SV::dtype &src_accum, const ushort laneid) {
//    reduce<base_ops::max, SV, false>(max_val, src, src_accum, laneid);
    using T = typename SV::dtype;
    T accum = base_types::constants<T>::neg_infty();
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = base_ops::max::template op<T>(accum, src[i]);
    }
    max_val = (T)metal::simd_max((float)accum);
    max_val = base_ops::max::template op<T>(max_val, src_accum);
}

/**
 * @brief Finds the minimum element in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] min_val The minimum value found in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to find the minimum in.
 * @param[in] src_accum The initial value to accumulate with the minimum value found.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
min(thread typename SV::dtype &min_val, threadgroup const SV &src, thread const typename SV::dtype &src_accum, const ushort laneid) {
//    reduce<base_ops::min, SV, false>(min_val, src, src_accum, laneid);
    using T = typename SV::dtype;
    T accum = base_types::constants<T>::pos_infty();
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = base_ops::max::template op<T>(accum, src[i]);
    }
    min_val = (T)metal::simd_min((float)accum);
    min_val = base_ops::max::template op<T>(min_val, src_accum);
}

/**
 * @brief Calculates the sum of elements in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] sum_val The sum of the values in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to sum.
 * @param[in] src_accum The initial value to accumulate with the sum of the vector.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
sum(thread typename SV::dtype &sum_val, threadgroup const SV &src, threadgroup const typename SV::dtype &src_accum, const ushort laneid) {
//    reduce<base_ops::sum, SV, false>(sum_val, src, src_accum, laneid);
    using T = typename SV::dtype;
    T accum = base_types::constants<T>::zero();
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = base_ops::max::template op<T>(accum, src[i]);
    }
    sum_val = (T)metal::simd_sum((float)accum);
    sum_val = base_ops::max::template op<T>(sum_val, src_accum);
}

/**
 * @brief Calculates the product of elements in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] prod_val The product of the values in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to multiply.
 * @param[in] src_accum The initial value to accumulate with the product of the vector.
 */
template<typename SV>
static METAL_FUNC typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
prod(thread typename SV::dtype &prod_val, threadgroup const SV &src, thread const typename SV::dtype &src_accum, const ushort laneid) {
//    reduce<base_ops::mul, SV, false>(prod_val, src, src_accum, laneid);
    using T = typename SV::dtype;
    T accum = base_types::constants<T>::one();
    if(laneid < SV::length) accum = src[laneid]; // initialize a register accumulator
    for(int i = laneid + 32; i < SV::length; i+=32) {
        accum = base_ops::max::template op<T>(accum, src[i]);
    }
    prod_val = (T)metal::simd_product((float)accum);
    prod_val = base_ops::max::template op<T>(prod_val, src_accum);
}
}



