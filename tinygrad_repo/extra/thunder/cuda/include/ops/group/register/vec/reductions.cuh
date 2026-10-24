/**
 * @file
 * @brief Reductions on vectors stored in registers.
 */

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
template<typename op, ducks::rv::all RV, bool reset>
__device__ static inline void reduce(
        typename base_types::packing<typename RV::dtype>::unpacked_type &dst_accum,
        const RV &src,
        const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    KITTENS_CHECK_WARP
    using T = base_types::packing<typename RV::dtype>::unpacked_type;
    int laneid = kittens::laneid();
    if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        T accum = op::template op<T>(src[0][0].x, src[0][0].y);
        #pragma unroll
        for(int i = 1; i < src.outer_dim; i++) {
            accum = op::template op<T>(accum, src[i][0].x);
            accum = op::template op<T>(accum, src[i][0].y);
        }
        // we've now reduced everything into 8 distinct values, replicated across lanes x, x+1, x+2, x+3 for x≡0(mod4)
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));
        // we've now reduced everything into 1 distinct value, replicated across lanes 0, 1, 2, 3
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);
        // final result has now been achieved (incorporating src_accum if necessary), finally broadcast back to all threads.
        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);
    }
    else if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        T accum = op::template op<T>(src[0][0].x, src[0][0].y);
        accum = op::template op<T>(accum,       src[0][1].x);
        accum = op::template op<T>(accum,       src[0][1].y);
        #pragma unroll
        for(int i = 1; i < src.outer_dim; i++) {
            // it is possible that shfl_sync's would be faster but I doubt it, replication is likely better. Certainly simpler.
            accum = op::template op<T>(accum, src[i][0].x);
            accum = op::template op<T>(accum, src[i][0].y);
            accum = op::template op<T>(accum, src[i][1].x);
            accum = op::template op<T>(accum, src[i][1].y);
        }
        // we've now reduced everything into 4 distinct values, replicated across lanes x, x+4, x+8, ..., x+28 for x<4
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));
        // we've now reduced everything into 1 distinct value, replicated across lanes 0, 4, 8, 12, ..., 28
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);
        // final result has now been achieved (incorporating src_accum if necessary), finally broadcast back to all threads from lane 0
        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        T accum = src[0][0];
        #pragma unroll
        for(int i = 1; i < src.outer_dim; i++) {
            if (i < src.outer_dim-1 || i*kittens::TILE_ROW_DIM<T>*2 + laneid < src.length) {
                accum = op::template op<T>(accum, src[i][0]);
            }
        }
        if(src.length > 16) accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);
        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0);
    }
}


/**
 * @brief Finds the maximum element in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] max_val The maximum value found in the vector.
 * @param[in] src The register vector to find the maximum in.
 */
template<ducks::rv::all RV>
__device__ static inline void max(typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, const RV &src) {
    reduce<base_ops::max, RV, true>(max_val, src, max_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type max(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type max_val;
    reduce<base_ops::max, RV, true>(max_val, src, max_val);
    return max_val;
}

/**
 * @brief Finds the minimum element in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] min_val The minimum value found in the vector.
 * @param[in] src The register vector to find the minimum in.
 */
template<ducks::rv::all RV>
__device__ static inline void min(typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, const RV &src) {
    reduce<base_ops::min, RV, true>(min_val, src, min_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type min(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type min_val;
    reduce<base_ops::min, RV, true>(min_val, src, min_val);
    return min_val;
}

/**
 * @brief Calculates the sum of elements in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] sum_val The sum of the values in the vector.
 * @param[in] src The register vector to sum.
 */
template<ducks::rv::all RV>
__device__ static inline void sum(typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, const RV &src) {
    reduce<base_ops::sum, RV, true>(sum_val, src, sum_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type sum(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type sum_val;
    reduce<base_ops::sum, RV, true>(sum_val, src, sum_val);
    return sum_val;
}

/**
 * @brief Calculates the product of elements in a register vector.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] prod_val The product of the values in the vector.
 * @param[in] src The register vector to multiply.
 */
template<ducks::rv::all RV>
__device__ static inline void prod(typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, const RV &src) {
    reduce<base_ops::mul, RV, true>(prod_val, src, prod_val);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type prod(const RV &src) {
    typename base_types::packing<typename RV::dtype>::unpacked_type prod_val;
    reduce<base_ops::mul, RV, true>(prod_val, src, prod_val);
    return prod_val;
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
template<ducks::rv::all RV>
__device__ static inline void max(typename base_types::packing<typename RV::dtype>::unpacked_type &max_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::max, RV, false>(max_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type max(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type max_val;
    reduce<base_ops::max, RV, false>(max_val, src, src_accum);
    return max_val;
}

/**
 * @brief Finds the minimum element in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] min_val The minimum value found in the vector, accumulated with src_accum.
 * @param[in] src The register vector to find the minimum in.
 * @param[in] src_accum The initial value to accumulate with the minimum value found.
 */
template<ducks::rv::all RV>
__device__ static inline void min(typename base_types::packing<typename RV::dtype>::unpacked_type &min_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::min, RV, false>(min_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type min(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type min_val;
    reduce<base_ops::min, RV, false>(min_val, src, src_accum);
    return min_val;
}

/**
 * @brief Calculates the sum of elements in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] sum_val The sum of the values in the vector, accumulated with src_accum.
 * @param[in] src The register vector to sum.
 * @param[in] src_accum The initial value to accumulate with the sum of the vector.
 */
template<ducks::rv::all RV>
__device__ static inline void sum(typename base_types::packing<typename RV::dtype>::unpacked_type &sum_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::sum, RV, false>(sum_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type sum(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type sum_val;
    reduce<base_ops::sum, RV, false>(sum_val, src, src_accum);
    return sum_val;
}

/**
 * @brief Calculates the product of elements in a register vector and accumulates it with src_accum.
 *
 * @tparam RV The type of the register vector. Must satisfy the `ducks::rv::all` concept.
 * @param[out] prod_val The product of the values in the vector, accumulated with src_accum.
 * @param[in] src The register vector to multiply.
 * @param[in] src_accum The initial value to accumulate with the product of the vector.
 */
template<ducks::rv::all RV>
__device__ static inline void prod(typename base_types::packing<typename RV::dtype>::unpacked_type &prod_val, const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    reduce<base_ops::mul, RV, false>(prod_val, src, src_accum);
}
template<ducks::rv::all RV>
__device__ static inline typename base_types::packing<typename RV::dtype>::unpacked_type prod(const RV &src, const typename base_types::packing<typename RV::dtype>::unpacked_type &src_accum) {
    typename base_types::packing<typename RV::dtype>::unpacked_type prod_val;
    reduce<base_ops::mul, RV, false>(prod_val, src, src_accum);
    return prod_val;
}