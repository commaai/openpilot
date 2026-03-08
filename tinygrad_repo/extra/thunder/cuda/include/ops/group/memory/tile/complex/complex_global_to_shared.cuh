/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */

template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load(CST &dst, const CGL &src, const COORD &idx) {
    load<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}

template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void store(CGL &dst, const CST &src, const COORD &idx) {
    store<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    store<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void store(CGL &dst, const CST &src, const COORD &idx) {
    store<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    store<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}

template<int axis, bool assume_aligned, ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    load_async<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load_async<axis, assume_aligned, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}
template<ducks::cst::all CST, ducks::cgl::all CGL, ducks::coord::tile COORD=coord<CST>>
__device__ static inline void load_async(CST &dst, const CGL &src, const COORD &idx) {
    load_async<2, false, typename CST::component, typename CGL::component, COORD>(dst.real, src.real, idx);
    load_async<2, false, typename CST::component, typename CGL::component, COORD>(dst.imag, src.imag, idx);
}