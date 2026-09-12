/**
 * @file
 * @brief Functions for a group scope to call tile TMA functions.
 */

template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::prefetch<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::prefetch<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<axis, policy, ST, PGL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<dim::ROW, cache_policy::NORMAL, ST, PGL, COORD>(dst, src, idx);
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<axis, policy, ST, PGL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_add_async<dim::ROW, cache_policy::NORMAL, ST, PGL, COORD>(dst, src, idx);
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_min_async<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_min_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_min_async<axis, policy, ST, PGL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_min_async<dim::ROW, cache_policy::NORMAL, ST, PGL, COORD>(dst, src, idx);
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_max_async<axis, policy, ST, GL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_max_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_max_async<axis, policy, ST, PGL, COORD>(dst, src, idx); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const PGL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_max_async<dim::ROW, cache_policy::NORMAL, ST, PGL, COORD>(dst, src, idx);
    }
}

template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    if(laneid() == 0) {
        ::kittens::tma::load_async<axis, policy, ST, GL, COORD>(dst, src, idx, bar); // Don't do the mask
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    if(laneid() == 0) {
        ::kittens::tma::load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar);
    }
}