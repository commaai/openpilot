
/**
 * @file
 * @brief Functions for a warpgroup to collaboratively transfer  data directly between global memory and registers and back.
 */

/**
 * @brief Collaboratively loads data into register vectors from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<typename RV, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
load(thread RV &dst, thread const GL &_src, thread coord idx, const int threadIdx) {
    using T  = typename RV::dtype;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    using T2 = typename base_types::packing<T>::packed_type;
    
    idx.c += warpid(threadIdx);
    // Call warp level store
    ::mittens::load(dst, _src, idx, simd_laneid(threadIdx));
}

/**
 * @brief Collaboratively stores data from register vectors to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<typename RV, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
store(thread GL &_dst, thread const RV &src, thread coord idx, const int threadIdx) {
    using T  = typename RV::dtype;
//    using U2 = typename base_types::packing<U>::packed_type;
    using T2 = typename base_types::packing<T>::packed_type;
    
    idx.c += warpid(threadIdx);

    // Call warp level store
    ::mittens::store(_dst, src, idx, simd_laneid(threadIdx));
}
