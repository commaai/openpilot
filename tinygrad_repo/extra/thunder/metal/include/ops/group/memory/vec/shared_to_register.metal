/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly between shared memory and registers and back.
 */

/**
 * @brief Collaboratively load data from a shared vector into register vectors split across a warpgroup.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<typename RV, typename SV>
METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>() && ducks::is_shared_vector<SV>(), void>::type
load(thread RV &dst, threadgroup const SV &_src, const int threadIdx) {
    using T  = typename RV::dtype;
    using U  = typename SV::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    using T2 = typename base_types::packing<T>::packed_type;

    static_assert(SV::length == RV::length*N_WARPS, "rv and sv dimensions do not match");// confirm size correct
//    threadgroup typename SV::template subvec<typename SV::dtype, RV::outer_dim> &src = subvec_inplace<RV::outer_dim, SV>(_src, warpid(threadIdx));
    //    threadgroup subvec &src = subvec_inplace<RV::outer_dim, SV>(_src, warpid(threadIdx));
    unsigned warpId = warpid(threadIdx);
    using subvec = typename SV::template subvec<RV::length>;

    threadgroup subvec& src = *(threadgroup subvec*)(&_src[warpId *RV::length]);
    
    ::mittens::load<RV, subvec>(dst, src, simd_laneid(threadIdx)); // warp-level
}

/**
 * @brief Collaboratively store data into a shared vector from register vectors split across a warpgroup.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<typename SV, typename RV>
METAL_FUNC static typename metal::enable_if<ducks::is_register_vector<RV>() && ducks::is_shared_vector<SV>(), void>::type
store(threadgroup SV &_dst, thread const RV &src, const int threadIdx) {
    using T  = typename RV::dtype;
    using U  = typename SV::dtype;
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;
    

    static_assert(SV::length == RV::length*N_WARPS, "rv and sv dimensions do not match");// confirm size correct
    
//    threadgroup typename SV::template subvec<typename SV::dtype, RV::outer_dim> &dst = subvec_inplace<RV::outer_dim, SV>(_dst, warpid(threadIdx));
//    ::mittens::store<threadgroup typename SV::template subvec<typename SV::dtype, RV::outer_dim>, RV>(dst, src, simd_laneid(threadIdx)); // warp-level
    
    unsigned warpId = warpid(threadIdx);
    using subvec = typename SV::template subvec<RV::length>;
    threadgroup subvec& dst = *(threadgroup subvec*)(&_dst[warpId * RV::length]);
    
    ::mittens::store(dst, src, simd_laneid(threadIdx)); // warp-level
}
