/**
 * @file
 * @brief Group reductions on shared vectors.
 */

// The fastest way to do this, under most circumstances, is actually to just have each warp replicate it.
// This is not true for enormous shared vectors, but doing that efficiently actually requires some extra scratch shared memory.
// So, this is sufficient for the time being.
template<typename op, ducks::sv::all SV, bool reset>
__device__ static inline void reduce(typename SV::dtype &dst_accum, const SV &src, const typename SV::dtype &src_accum) {
    if constexpr (GROUP_WARPS == 1) {
        using T = SV::dtype;
        int lane = laneid();
        T accum;
        if(lane < src.length) accum = src[lane]; // initialize a register accumulator
        __syncwarp();
        for(int i = lane+kittens::WARP_THREADS; i < src.length; i+=kittens::WARP_THREADS) {
            accum = op::template op<T>(accum, src[i]);
        }
        __syncwarp();
        // We can now reduce within the warp.
        if constexpr (src.length > 16) {
            accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 16));
            __syncwarp();
        }
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 8));
        __syncwarp();
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 4));
        __syncwarp();
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 2));
        __syncwarp();
        accum = op::template op<T>(accum, packed_shfl_down_sync(kittens::MASK_ALL, accum, 1));
        __syncwarp();
        if constexpr (!reset) accum = op::template op<T>(accum, src_accum);
        // broadcast to all threads in the warp.
        dst_accum = packed_shfl_sync(kittens::MASK_ALL, accum, 0); // everyone takes from warp leader
    }
    else {
        ::kittens::group<1>::reduce<op, SV, reset>(dst_accum, src, src_accum);
    }
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

/**
 * @brief Finds the maximum element in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] max_val The maximum value found in the vector.
 * @param[in] src The shared memory vector to find the maximum in.
 */
template<ducks::sv::all SV>
__device__ static inline void max(typename SV::dtype &max_val, const SV &src) {
    reduce<base_ops::max, SV, true>(max_val, src, max_val);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype max(const SV &src) {
    typename SV::dtype max_val;
    reduce<base_ops::max, SV, true>(max_val, src, max_val);
    return max_val;
}

/**
 * @brief Finds the minimum element in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] min_val The minimum value found in the vector.
 * @param[in] src The shared memory vector to find the minimum in.
 */
template<ducks::sv::all SV>
__device__ static inline void min(typename SV::dtype &min_val, const SV &src) {
    reduce<base_ops::min, SV, true>(min_val, src, min_val);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype min(const SV &src) {
    typename SV::dtype min_val;
    reduce<base_ops::min, SV, true>(min_val, src, min_val);
    return min_val;
}

/**
 * @brief Calculates the sum of elements in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] sum_val The sum of the values in the vector.
 * @param[in] src The shared memory vector to sum.
 */
template<ducks::sv::all SV>
__device__ static inline void sum(typename SV::dtype &sum_val, const SV &src) {
    reduce<base_ops::sum, SV, true>(sum_val, src, sum_val);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype sum(const SV &src) {
    typename SV::dtype sum_val;
    reduce<base_ops::sum, SV, true>(sum_val, src, sum_val);
    return sum_val;
}

/**
 * @brief Calculates the product of elements in a shared memory vector.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] prod_val The product of the values in the vector.
 * @param[in] src The shared memory vector to multiply.
 */
template<ducks::sv::all SV>
__device__ static inline void prod(typename SV::dtype &prod_val, const SV &src) {
    reduce<base_ops::mul, SV, true>(prod_val, src, prod_val);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype prod(const SV &src) {
    typename SV::dtype prod_val;
    reduce<base_ops::mul, SV, true>(prod_val, src, prod_val);
    return prod_val;
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
template<ducks::sv::all SV>
__device__ static inline void max(typename SV::dtype &max_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::max, SV, false>(max_val, src, src_accum);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype max(const SV &src, const typename SV::dtype &src_accum) {
    typename SV::dtype max_val;
    reduce<base_ops::max, SV, false>(max_val, src, src_accum);
    return max_val;
}

/**
 * @brief Finds the minimum element in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] min_val The minimum value found in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to find the minimum in.
 * @param[in] src_accum The initial value to accumulate with the minimum value found.
 */
template<ducks::sv::all SV>
__device__ static inline void min(typename SV::dtype &min_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::min, SV, false>(min_val, src, src_accum);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype min(const SV &src, const typename SV::dtype &src_accum) {
    typename SV::dtype min_val;
    reduce<base_ops::min, SV, false>(min_val, src, src_accum);
    return min_val;
}

/**
 * @brief Calculates the sum of elements in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] sum_val The sum of the values in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to sum.
 * @param[in] src_accum The initial value to accumulate with the sum of the vector.
 */
template<ducks::sv::all SV>
__device__ static inline void sum(typename SV::dtype &sum_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::sum, SV, false>(sum_val, src, src_accum);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype sum(const SV &src, const typename SV::dtype &src_accum) {
    typename SV::dtype sum_val;
    reduce<base_ops::sum, SV, false>(sum_val, src, src_accum);
    return sum_val;
}

/**
 * @brief Calculates the product of elements in a shared memory vector and accumulates it with src_accum.
 *
 * @tparam SV The type of the shared memory vector. Must satisfy the `ducks::sv::all` concept.
 * @param[out] prod_val The product of the values in the vector, accumulated with src_accum.
 * @param[in] src The shared memory vector to multiply.
 * @param[in] src_accum The initial value to accumulate with the product of the vector.
 */
template<ducks::sv::all SV>
__device__ static inline void prod(typename SV::dtype &prod_val, const SV &src, const typename SV::dtype &src_accum) {
    reduce<base_ops::mul, SV, false>(prod_val, src, src_accum);
}
template<ducks::sv::all SV>
__device__ static inline typename SV::dtype prod(const SV &src, const typename SV::dtype &src_accum) {
    typename SV::dtype prod_val;
    reduce<base_ops::mul, SV, false>(prod_val, src, src_accum);
    return prod_val;
}