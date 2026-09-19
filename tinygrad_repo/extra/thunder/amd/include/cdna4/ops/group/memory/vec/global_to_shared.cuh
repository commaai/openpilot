/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared vectors from and storing to global memory. 
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> // default case
__device__ static inline void load(SV &dst, const GL &src, const COORD &idx) {
    kittens::load<SV, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> // default case
__device__ static inline void store(const GL &dst, const SV &src, const COORD &idx) {
    kittens::store<SV, GL, COORD, GROUP_THREADS>(dst, src, idx);
}

