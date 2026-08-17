/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void prefill_swizzled_offsets(ST &dst, const GL &src, uint32_t *swizzled_offsets) {
    kittens::prefill_swizzled_offsets<axis, assume_aligned, ST, GL, GROUP_THREADS>(dst, src, swizzled_offsets);
}
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void prefill_swizzled_offsets(ST &dst, const GL &src, uint32_t *swizzled_offsets) {
    kittens::prefill_swizzled_offsets<2, false, ST, GL, GROUP_THREADS>(dst, src, swizzled_offsets);
}
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t *swizzled_offsets) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx, swizzled_offsets);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t *swizzled_offsets) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx, swizzled_offsets);
}
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t *__restrict__ swizzled_offsets, i32x4 srd, const void* base_ptr, uint32_t lds_base) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx, swizzled_offsets, srd, base_ptr, lds_base);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t *__restrict__ swizzled_offsets, i32x4 srd, const void* base_ptr, uint32_t lds_base) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx, swizzled_offsets, srd, base_ptr, lds_base);
}