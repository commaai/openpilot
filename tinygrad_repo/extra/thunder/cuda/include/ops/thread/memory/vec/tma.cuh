#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

#include <cuda.h>
#include <iostream>

// This is a macro that helps us define default cache policy versions of each function.
#define __KITTENS_TMA_DEFINE_DEFAULT_LOAD_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const GL &src, const COORD &idx) { \
    function_name<cache_policy::NORMAL>(dst, src, idx); \
}
#define __KITTENS_TMA_DEFINE_PGL_DEFAULT_LOAD_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const PGL &src, const COORD &idx) { \
    function_name<cache_policy::NORMAL>(dst, src, idx); \
}
#define __KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(const GL &dst, const SV &src, const COORD &idx) { \
    function_name<cache_policy::NORMAL>(dst, src, idx); \
}
#define __KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(const PGL &dst, const SV &src, const COORD &idx) { \
    function_name<cache_policy::NORMAL>(dst, src, idx); \
}
#define __KITTENS_TMA_DEFINE_SEMAPHORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const GL &src, const COORD &idx, semaphore& bar) { \
    function_name<cache_policy::NORMAL>(dst, src, idx, bar); \
}
#define __KITTENS_TMA_DEFINE_PGL_SEMAPHORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const PGL &src, const COORD &idx, semaphore& bar) { \
    function_name<cache_policy::NORMAL>(dst, src, idx, bar); \
}
#define __KITTENS_TMA_DEFINE_CLUSTER_SEMAPHORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) { \
    function_name<cache_policy::NORMAL>(dst, src, idx, bar, cluster_mask, dst_mbar_cta); \
}
#define __KITTENS_TMA_DEFINE_PGL_CLUSTER_SEMAPHORE_CACHE_VEC__(function_name) \
template<ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>> \
__device__ static inline void function_name(SV &dst, const PGL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) { \
    function_name<cache_policy::NORMAL>(dst, src, idx, bar, cluster_mask, dst_mbar_cta); \
}


namespace kittens {

namespace detail {
namespace tma {

template<cache_policy policy> __device__ static inline void vec_prefetch_tma_internal(uint64_t tma_ptr, coord<> tma_coord) {
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.prefetch.tensor.4d.L2.global.tile"
            " [%0, {%1, %2, %3, %4}];"
            :
            : "l"(tma_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint"
            " [%0, {%1, %2, %3, %4}], %5;"
            :
            : "l"(tma_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

template<cache_policy policy> __device__ static inline void vec_store_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

template<cache_policy policy> __device__ static inline void vec_store_add_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

template<cache_policy policy> __device__ static inline void vec_store_min_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

template<cache_policy policy> __device__ static inline void vec_store_max_async_tma_internal(uint64_t tma_ptr, uint32_t src_i_ptr, coord<> tma_coord) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5}], [%1], %6;"
            :
            : "l"(tma_ptr), "r"(src_i_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

template<cache_policy policy> __device__ static inline void vec_load_async_tma_internal(uint64_t tma_ptr, uint32_t dst_i_ptr, uint32_t mbar_ptr, coord<> tma_coord) {
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6}], [%2];"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr), "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}

namespace cluster {
template<cache_policy policy> __device__ static inline void vec_load_async_tma_internal(uint64_t tma_ptr, uint32_t dst_i_ptr, uint32_t mbar_ptr, coord<> tma_coord, uint16_t cluster_mask, int dst_mbar_cta=-1) {
#ifdef KITTENS_BLACKWELL
    if(dst_mbar_cta != -1) {
        uint32_t neighbor_mbar_ptr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_mbar_ptr)
            : "r"(mbar_ptr), "r"(dst_mbar_cta)
        );
        if constexpr (policy == cache_policy::NORMAL) {
            asm volatile (
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
                " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
                :
                : "r"(dst_i_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "h"(cluster_mask)
                : "memory"
            );
        }
        else {
            asm volatile (
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6}], [%2], %7, %8;"
                :
                : "r"(dst_i_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else 
#endif
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "h"(cluster_mask)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5, %6}], [%2], %7, %8;"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b), "h"(cluster_mask), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}
} // namespace cluster

} // namespace tma
} // namespace detail

namespace tma {

template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void prefetch(SV &dst, const GL &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        ::kittens::detail::tma::vec_prefetch_tma_internal<policy>(tma_ptr, tma_coord);
    }
}
__KITTENS_TMA_DEFINE_DEFAULT_LOAD_CACHE_VEC__(prefetch)

template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_async)

template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_async(const PGL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_async)

template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const GL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_add_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_add_async)

template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_add_async(const PGL &dst, const SV &src, const COORD &idx) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_add_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_add_async)

template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const GL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_min_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_min_async)

template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_min_async(const PGL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_min_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_min_async)

template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const GL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_max_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_DEFAULT_STORE_CACHE_VEC__(store_max_async)

template<cache_policy policy, ducks::sv::all SV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store_max_async(const PGL &dst, const SV &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename SV::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst.template get_tma<SV, -1>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t src_i_ptr = src_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_store_max_async_tma_internal<policy>(tma_ptr, src_i_ptr, tma_coord);
    }
    ::kittens::tma::store_commit_group();
}
__KITTENS_TMA_DEFINE_PGL_DEFAULT_STORE_CACHE_VEC__(store_max_async)

template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord);
    }
}
__KITTENS_TMA_DEFINE_SEMAPHORE_CACHE_VEC__(load_async)

namespace cluster {
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for(int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i*::kittens::detail::tma::sv_tma_dim1<SV>*sizeof(typename SV::dtype);
        ::kittens::detail::tma::cluster::vec_load_async_tma_internal<policy>(tma_ptr, dst_i_ptr, mbar_ptr, tma_coord, cluster_mask, dst_mbar_cta);
    }
}
__KITTENS_TMA_DEFINE_CLUSTER_SEMAPHORE_CACHE_VEC__(load_async)
} // namespace cluster
} // namespace tma
} // namespace kittens