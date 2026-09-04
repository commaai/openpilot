/**
 * @file
 * @brief Functions for a group scope to call vec TMA cluster functions.
 */

/**
 * @brief Asynchronously loads data from global memory into a shared memory vector, broadcast across a cluster
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam SV A shared vector type with a TMA-compatible layout
 * @param[out] dst The destination shared memory vector.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 * @param[in] vec_idx The coord of the requested vector.
 * @param[in] cluster_mask The mask of the clusters to broadcast to.
 */
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = ::kittens::laneid(); i < ::kittens::detail::tma::sv_tma_dim2<SV>; i += WARP_THREADS) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::cluster::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord, cluster_mask, dst_mbar_cta);
    }
}
__KITTENS_TMA_DEFINE_CLUSTER_SEMAPHORE_CACHE_VEC__(load_async)
