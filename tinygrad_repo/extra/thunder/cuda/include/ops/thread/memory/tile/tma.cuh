#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {
namespace tma {

namespace detail {
template<kittens::ducks::st::all ST, int axis> __device__ inline int4 tma_coords(const coord<ducks::default_type> &unit_coord) {
    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(typename ST::dtype);
    if constexpr      (axis == 2) return {unit_coord.r, unit_coord.c / swizzle_elements, unit_coord.d, unit_coord.b};
    else if constexpr (axis == 1) return {unit_coord.d, unit_coord.c / swizzle_elements, unit_coord.r, unit_coord.b};
    else if constexpr (axis == 0) return {unit_coord.b, unit_coord.c / swizzle_elements, unit_coord.r, unit_coord.d};
}
}

/* ----------   Prefetch Tensor Map  ---------- */

/**
 * @brief Prefetches data from global memory into a shared memory tile, along with the tensormap.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] tile_row_idx The row coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the requested tile. This is in units of complete tiles.
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.prefetch.tensor.5d.L2.global.tile"
            " [%0, {%1, %2, %3, %4, %5}];"
            :
            : "l"(tma_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint"
            " [%0, {%1, %2, %3, %4, %5}], %6;"
            :
            : "l"(tma_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void prefetch(ST &dst, const GL &src, const COORD &idx) {
    prefetch<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}

/* ----------   Async load and store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously stores data into global memory from a shared memory tile.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row coord of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the tile destination. This is in units of complete tiles.
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
    store_commit_group();
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    store_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const PGL &dst, const ST &src, const COORD &idx) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
    store_commit_group();
}
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_async(const PGL &dst, const ST &src, const COORD &idx) {
    store_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
}

/* ----------   Async reduction + store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously performs an add reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous add reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row coord of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the tile destination. This is in units of complete tiles.
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {

    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");
                    
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
    store_commit_group();
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const COORD &idx) {
    store_add_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const PGL &dst, const ST &src, const COORD &idx) {

    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");

    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
    store_commit_group();
}
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_add_async(const PGL &dst, const ST &src, const COORD &idx) {
    store_add_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
}

/**
 * @brief Asynchronously performs an min reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous min reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row coord of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the tile destination. This is in units of complete tiles.
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");

    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");

    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
    store_commit_group();
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const COORD &idx) {
    store_min_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const PGL &dst, const ST &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");

    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");

    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
    store_commit_group();
}
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_min_async(const PGL &dst, const ST &src, const COORD &idx) {
    store_min_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
}

/**
 * @brief Asynchronously performs an max reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous max reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row coord of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the tile destination. This is in units of complete tiles.
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");

    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");

    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
    store_commit_group();
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const COORD &idx) {
    store_max_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx);
}
template<int axis, cache_policy policy, ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const PGL &dst, const ST &src, const COORD &idx) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");

    static_assert(!(std::is_same_v<typename ST::dtype, fp8e4m3> ||
                    std::is_same_v<typename ST::dtype, fp8e5m2>), 
                    "TMA does not support async add reductions for fp8 types.");

    uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST, axis>());
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group.L2::cache_hint"
            " [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
    store_commit_group();
}
template<ducks::st::all ST, ducks::pgl::all PGL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store_max_async(const PGL &dst, const ST &src, const COORD &idx) {
    store_max_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 * @param[in] tile_row_idx The row coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the requested tile. This is in units of complete tiles.
 */
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile(
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w)
            : "memory"
        );
    }
    else {
        asm volatile(
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar);
}

namespace cluster {

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile, across a threadblock cluster
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 * @param[in] tile_row_idx The row coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the requested tile. This is in units of complete tiles.
 * @param[in] cluster_mask The mask of the clusters to broadcast to.
 */
#ifdef KITTENS_BLACKWELL
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1)
#else
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask)
#endif
{
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST, axis>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    coord<ducks::default_type> unit_coord = idx.template unit_coord<axis, 3>(); // convert to unit coordinates
    int4 tma_coords = detail::tma_coords<ST, axis>(unit_coord);

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
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "h"(cluster_mask)
                : "memory"
            );
        }
        else {
            asm volatile (
                "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2.multicast::cluster.L2::cache_hint"
                " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8, %9;"
                :
                : "r"(dst_ptr), "l"(tma_ptr), "r"(neighbor_mbar_ptr),
                "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "h"(cluster_mask), "l"(make_cache_policy<policy>())
                : "memory"
            );
        }
    } else
#endif
    if constexpr (policy == cache_policy::NORMAL) {
        asm volatile (
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "h"(cluster_mask)
            : "memory"
        );
    }
    else {
        asm volatile (
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8, %9;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "n"(0), "r"(tma_coords.x), "r"(tma_coords.y), "r"(tma_coords.z), "r"(tma_coords.w), "h"(cluster_mask), "l"(make_cache_policy<policy>())
            : "memory"
        );
    }
}
#ifdef KITTENS_BLACKWELL
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask, int dst_mbar_cta=-1) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask, dst_mbar_cta);
}
#else
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore& bar, uint16_t cluster_mask) {
    load_async<dim::ROW, cache_policy::NORMAL, ST, GL, COORD>(dst, src, idx, bar, cluster_mask);
}
#endif

} // namespace cluster
} // namespace tma

} // namespace kittens
