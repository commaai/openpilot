/**
 * @file
 * @brief Functions for a group scope to call tile TMA cluster functions.
 */


#ifdef KITTENS_BLACKWELL
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::load_async<axis, policy, ST, GL, COORD>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
    }
}
#else
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::load_async<axis, policy, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask) {
    if(laneid() == 0) {
        ::kittens::tma::cluster::load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
    }
}
#endif