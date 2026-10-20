/**
 * @file
 * @brief Synchronization primitives for gfx1250.
 *
 * gfx1250 replaces the unified `s_waitcnt` with per-counter waits and exposes
 * a split barrier that lets a warp signal "data ready" before waiting on
 * the corresponding consumer barrier. This header wraps every primitive
 * behind a clean `kittens::sync::*` API; the actual instructions are emitted
 * via clang `__builtin_amdgcn_*` builtins where available, and inline
 * assembly otherwise (per-counter waits other than `asynccnt`/`tensorcnt`
 * are not yet exposed as clang builtins as of LLVM 23).
 */

#pragma once


#include "../../../common/common.cuh"

namespace kittens {
namespace sync {

/* ----------  SPLIT BARRIER (BLOCK-WIDE)  ---------- */

/**
 * @brief Signal a block-wide split barrier.
 *
 * Lowers to `s_barrier_signal -1`. May be issued from any warp and returns
 * immediately; only `wait()` blocks until every warp in the block has signalled.
 */
__device__ __forceinline__ void arrive() {
    __builtin_amdgcn_s_barrier_signal(-1);
}

/**
 * @brief Wait on a block-wide split barrier.
 *
 * Lowers to `s_barrier_wait -1`. Blocks until every warp in the block has
 * called `arrive()` since the last completion of this barrier.
 */
__device__ __forceinline__ void wait() {
    __builtin_amdgcn_s_barrier_wait(-1);
}

/**
 * @brief Block-wide barrier (signal + wait).
 *
 * Semantically equivalent to `__syncthreads()`. Prefer the split form
 * (`arrive()` followed by independent work followed by `wait()`) when the
 * window between signalling and waiting can be filled with non-dependent
 * instructions.
 */
__device__ __forceinline__ void sync() {
    arrive();
    wait();
}

/* ----------  GFX12+ PER-COUNTER WAITS  ---------- */
//
// Each gfx1250 wait counter is 6 bits; the `N` template parameter is the
// maximum number of in-flight ops that may remain after the wait. The default
// `N=0` drains the counter completely (semantically a full sync of that class).
// Use a non-zero `N` to keep a `K`-deep pipeline running, draining one slot
// at a time as new ops are issued.

/**
 * @brief Wait for outstanding global (and texture) loads, leaving up to N in flight.
 *
 * Lowers to `s_wait_loadcnt N`. Required after a `global_load_async_to_lds`
 * or any `global_load_*` whose results are about to be read.
 *
 * @note Clang 23 does not yet expose `__builtin_amdgcn_s_wait_loadcnt`;
 *       we emit the instruction directly with `N` as an immediate operand.
 */
template<int N = 0>
__device__ __forceinline__ void wait_load() {
    static_assert(N >= 0 && N < 64, "loadcnt is 6-bit; max 63");
    asm volatile("s_wait_loadcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding global stores, leaving up to N in flight.
 *
 * Lowers to `s_wait_storecnt N`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_store() {
    static_assert(N >= 0 && N < 64, "storecnt is 6-bit; max 63");
    asm volatile("s_wait_storecnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding LDS (DS_*) operations, leaving up to N in flight.
 *
 * Lowers to `s_wait_dscnt N`. Required between LDS writes (or `ds_load_b*`
 * issues) and a dependent VALU/WMMA consumer.
 */
template<int N = 0>
__device__ __forceinline__ void wait_ds() {
    static_assert(N >= 0 && N < 64, "dscnt is 6-bit; max 63");
    asm volatile("s_wait_dscnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding kernel-message ops, leaving up to N in flight.
 *
 * Lowers to `s_wait_kmcnt N`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_km() {
    static_assert(N >= 0 && N < 64, "kmcnt is 6-bit; max 63");
    asm volatile("s_wait_kmcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding async global->LDS transfers, leaving up to N in flight.
 *
 * Lowers to `s_wait_asynccnt N`. Drains anything started by
 * `__builtin_amdgcn_(global|cluster)_load_async_to_lds_*`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_async() {
    static_assert(N >= 0 && N < 64, "asynccnt is 6-bit; max 63");
    __builtin_amdgcn_s_wait_asynccnt(N);
}

/**
 * @brief Wait for outstanding TDM transfers, leaving up to N in flight.
 *
 * Lowers to `s_wait_tensorcnt N`. Drains anything started by
 * `__builtin_amdgcn_tensor_load_to_lds` or `tensor_store_from_lds`.
 *
 * @code
 *   load_tdm(buf[0], ...);
 *   load_tdm(buf[1], ...);
 *   load_tdm(buf[2], ...);
 *   for (int k = 0; k + 3 < K; ++k) {
 *       sync::wait_tdm<2>();              // drain one slot, two stay in flight
 *       consume(buf[k % 3]);
 *       load_tdm(buf[k % 3], ...);
 *   }
 *   sync::wait_tdm<0>();                  // drain the tail
 * @endcode
 */
template<int N = 0>
__device__ __forceinline__ void wait_tdm() {
    static_assert(N >= 0 && N < 64, "tensorcnt is 6-bit; max 63");
    __builtin_amdgcn_s_wait_tensorcnt(N);
}

/**
 * @brief Memory fence covering both global loads and LDS ops.
 *
 * Convenience for the common "producer side" pattern: ensure all in-flight
 * loads have settled into LDS before signalling consumers.
 */
__device__ __forceinline__ void fence() {
    wait_load<0>();
    wait_ds<0>();
}

/* ----------  LDS BARRIER CELLS (FOR TDM / ASYNC ARRIVE)  ---------- */
//
// 64-bit LDS-resident barrier cell, per SP3 section 9.8.13
// (DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64). The cell is packed as:
//
//   bits 63..48 : reserved (zero)
//   bits 47..32 : init_count        (reload value at phase flip)
//   bits 31..0  : bar_state, itself packed as [phase | pending_count]
//                 with the WIDTH boundary at bit 16:
//                   bits 31..16 : phase            (counter, parity alternates)
//                   bits 15..0  : pending_count
//
// Each arrive subtracts 1 from bar_state. When pending rolls under (its
// MSB becomes 1), the hardware reloads bar_state to
//   (new_phase << 16) | init_count
// and wakes any wave sleeping on the cell.
//
// To expect N arrivals per phase: pending starts at N-1 and init_count is
// N-1. Phase decrements per flip, so its LSB alternates 0,1,0,1... -- the
// classic parity-flip pattern.

/**
 * @brief 64-bit LDS barrier cell.
 *
 * Allocate as `__shared__ kittens::sync::barrier_lds bar;` and prime once
 * with `init_barrier(&bar.state, count)` before any arrive. `count` is the
 * number of arrivals required to flip the phase.
 */
struct alignas(8) barrier_lds { uint64_t state; };

/// @brief Initialize an LDS barrier cell to expect `count` arrivals per phase.
__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    // pending = count - 1, phase = 0, init_count = count - 1.
    const uint32_t pending  = count - 1;
    const uint32_t init_cnt = count - 1;
    *bar =  uint64_t(pending  & 0xFFFFu)            // bits 15..0  pending
         | (uint64_t(0)                  << 16)    // bits 31..16 phase = 0
         | (uint64_t(init_cnt & 0xFFFFu) << 32);   // bits 47..32 init_count
}

/**
 * @brief Block on `bar` until its phase LSB matches `expected_phase`.
 *
 * Phase decrements once per flip, so its low bit alternates 0,1,0,1...
 * Callers maintain a parity bit per barrier and pass it inverted before
 * each wait (`expected = (phase ^= 1)`). The hardware wakes sleeping
 * waves on phase flip; `s_sleep 1` yields the SIMD between polls.
 */
__device__ __forceinline__ void wait_barrier(uint64_t* bar, int expected_phase) {
    const uint32_t lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(bar));
    while (true) {
        uint64_t v;
        asm volatile("ds_load_b64 %0, %1 offset:0"
            : "=v"(v) : "v"(lds_addr) : "memory");
        // Phase lives in the high 16 bits of the low 32-bit bar_state.
        const uint32_t bar_state = static_cast<uint32_t>(v);
        const int phase_lsb = int((bar_state >> 16) & 1);
        if (phase_lsb == expected_phase) break;
        __builtin_amdgcn_s_sleep(1);
    }
}

/**
 * @brief Arrive at an LDS barrier cell from an async-ordered path.
 *
 * Lowers to `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`. Use this to manually
 * arrive at a cell (the auto-arrive form is encoded in the TDM descriptor
 * via `load_tdm_arrive`).
 */
__device__ __forceinline__ void async_barrier_arrive(uint64_t* lds_counter) {
    uintptr_t lds_uint = reinterpret_cast<uintptr_t>(lds_counter);
    __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64(
        reinterpret_cast<long __attribute__((address_space(3)))*>(lds_uint));
}

} // namespace sync
} // namespace kittens

