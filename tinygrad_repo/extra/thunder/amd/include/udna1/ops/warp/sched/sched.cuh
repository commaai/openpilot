/**
 * @file
 * @brief Scheduling primitives for gfx1250.
 *
 * Wraps three families of low-level scheduling controls behind a
 * `kittens::sched::*` API:
 *
 *   1. **Expert scheduling** -- by default a wave's memory instructions
 *      (loads and stores) are held until that wave's earlier math
 *      instructions have finished, so the hardware never has to track a
 *      dependency between the two for you. Expert mode lifts that automatic
 *      hold, so memory and math from the same wave overlap freely. That
 *      overlap is the key enabler for tight producer/consumer GEMM loops.
 *      In exchange, the kernel author becomes responsible for inserting an
 *      explicit wait (`wait_alu`) wherever a later instruction really does
 *      depend on an earlier one. See `set_expert`, `expert_scope`,
 *      `lock_simd`.
 *   2. **Wave priority** -- a hint that breaks ties when several waves on
 *      the same SIMD want to issue in the same cycle; the higher-priority
 *      wave goes first. See `set_priority` (this wave alone) and
 *      `boost_priority` (the rest of the workgroup on this SIMD).
 *   3. **Compiler fence** -- a compile-time-only barrier that stops the
 *      compiler from reordering instructions across a point. Emits no
 *      hardware instruction and costs nothing at runtime. See
 *      `compiler_fence`.
 */

#pragma once


#include "../../../common/common.cuh"

namespace kittens {
namespace sched {

namespace detail {
// Opaque, gfx1250-specific operands for the s_setreg instructions that
// toggle this wave's expert-scheduling controls.
constexpr int SCHED_MODE_EXPERT_SIMM16    = 26 | (0 << 6) | (1 << 11);  // expert mode on/off
constexpr int SCHED_MODE_CLAIM_SIMD_SIMM16 = 26 | (4 << 6) | (0 << 11);  // back-to-back WMMAs
} // namespace detail

/**
 * @brief Enable (`on=true`) or disable expert scheduling for this wave.
 *
 * Enabling removes the default hold that keeps a wave's memory
 * instructions (loads and stores) from issuing until that wave's earlier
 * math instructions have finished. With it on, memory and math from the
 * same wave overlap freely -- the key enabler for tight producer/consumer
 * GEMM inner loops, where an async global-to-LDS load is launched ahead of
 * a sequence of matrix multiplies with no stall between them.
 *
 * Caller responsibility once expert mode is on: any time a later
 * instruction genuinely needs the result of a math instruction (anything
 * the wave computes on its vector lanes -- ordinary arithmetic and matrix
 * multiplies / WMMA alike), insert an explicit `kittens::sched::wait_alu()`
 * before that consumer. With expert mode off the hardware inserts this wait
 * for you; with it on, you do.
 *
 * Whoever turns it on must also turn it off. Prefer `expert_scope`, which
 * does both at the start and end of a scope.
 */
__device__ __forceinline__ void set_expert(bool on) {
    // Value 2 selects the scheduling mode that lifts *only* the memory<->math
    // overlap interlock these loops rely on, so a single `wait_alu()` at each
    // genuine dependency is the complete fix. (The more aggressive value 1
    // also lifts the scalar-side interlocks, which clang-generated scalar code
    // in these kernels cannot safely take responsibility for.)
    __builtin_amdgcn_s_setreg(detail::SCHED_MODE_EXPERT_SIMM16,
                              on ? 2u : 0u);
}

/**
 * @brief RAII guard: enable expert mode for a scope; restore default on exit.
 *
 * @code
 *   {
 *     kittens::sched::expert_scope _sched;     // expert mode on here
 *     for (int k = 0; k < K; ++k) {
 *       // memory loads and matrix multiplies overlap within this wave
 *     }
 *   }                                          // destructor turns it back off
 * @endcode
 *
 * On scope exit the destructor calls `set_expert(false)`, restoring default
 * scheduling -- so the [enable -> K-loop -> disable] pattern stays correct
 * even on an early return or a thrown exception. Non-copyable and
 * non-movable by design: a wave-state guard's lifetime should track a stack
 * scope exactly.
 */
struct expert_scope {
    __device__ __forceinline__ expert_scope()  { set_expert(true);  }
    __device__ __forceinline__ ~expert_scope() { set_expert(false); }

    expert_scope(const expert_scope&)            = delete;
    expert_scope& operator=(const expert_scope&) = delete;
};

/**
 * @brief Lock the SIMD for this wave so it can issue matrix multiplies
 *        back-to-back, instead of yielding to co-resident waves between them.
 *
 * Normally, after a wave issues a matrix multiply (WMMA) the hardware makes
 * it pause briefly before issuing anything else, so other waves sharing the
 * same SIMD get a turn to run their own work in the gap. This call removes
 * that pause for the issuing wave, letting it stream WMMAs one after another
 * with no forced break.
 *
 * **Use it** when a single wave owns the SIMD (one-wave-per-SIMD kernels):
 * there is no one else to hand the gap to, so the pause is pure overhead.
 *
 * **Avoid it** when two or more waves share a SIMD and you rely on them to
 * fill each other's gaps -- removing the pause starves the others.
 *
 * Set this once near the top of the kernel. Unlike `expert_scope` it is not
 * scoped and there is no guard for it: it stays in effect for the wave's
 * lifetime.
 */
__device__ __forceinline__ void lock_simd() {
    __builtin_amdgcn_s_setreg(detail::SCHED_MODE_CLAIM_SIMD_SIMM16, 1u);
}

/**
 * @brief Resolve this wave's outstanding ALU dependencies by hand
 *        (needed only under expert mode).
 *
 * Drains both dependency counters that expert scheduling stops the hardware
 * from checking for you:
 *   - outstanding vector/matrix (WMMA) result writebacks, so a later
 *     instruction may read those results or reuse the registers they consumed;
 *   - outstanding load/store source-register reads, so those registers may be
 *     safely overwritten.
 * Once it returns, both are complete. The common case in a GEMM K-loop is
 * reusing a register tile a matrix multiply read: call this before loading the
 * next K-tile into the same registers the previous multiply consumed.
 *
 * With expert scheduling off the hardware inserts these waits for you, so this
 * is never needed. Note it does *not* wait for loaded data to *arrive* -- that
 * is a memory-count wait (`sync::wait_load`, `wait_ds`, `wait_async`, ...),
 * which expert mode does not affect.
 *
 * @note Clang 23 does not expose `__builtin_amdgcn_s_wait_alu`; we emit the
 *       instructions directly. Lowers to `s_wait_alu depctr_va_vdst(0)` then
 *       `s_wait_alu depctr_vm_vsrc(0)`.
 */
__device__ __forceinline__ void wait_alu() {
    asm volatile("s_wait_alu depctr_va_vdst(0)" ::: "memory");
    asm volatile("s_wait_alu depctr_vm_vsrc(0)" ::: "memory");
}

/**
 * @brief Set this wave's issue priority to `PRIO`.
 *
 * Lowers to `s_setprio N`, with `N` in `[0, 3]`. When several waves on the
 * same SIMD want to issue in the same cycle, the hardware lets the
 * higher-priority wave go first. Default is `0`.
 *
 * Affects this wave only -- other waves in the same workgroup, even those
 * on the same SIMD, keep their existing priority. Takes a few cycles to
 * take effect.
 *
 * Template-parameterized because `s_setprio` requires a compile-time
 * constant operand.
 *
 * @tparam PRIO new priority in [0, 3].
 */
template<int PRIO>
__device__ __forceinline__ void set_priority() {
    static_assert(PRIO >= 0 && PRIO <= 3, "s_setprio takes a 2-bit constant");
    __builtin_amdgcn_s_setprio(static_cast<short>(PRIO));
}

/**
 * @brief Raise the issue priority by `DELTA` for every wave of this
 *        workgroup currently sharing the **same SIMD** as the caller.
 *
 * Lowers to `s_setprio_inc_wg N`. Useful when one warp wants to pull the
 * rest of its workgroup forward together -- e.g. a producer warp boosting
 * the consumer warps once the prologue has loaded the first K-tile.
 *
 * The "same SIMD" qualifier matters: a gfx1250 workgroup's waves are spread
 * across the four SIMDs of a Compute Unit (CU), and this only bumps the
 * members on the caller's SIMD. Waves of the same workgroup on the other
 * three SIMDs keep their existing priority. Takes a few cycles to take
 * effect.
 *
 * @tparam DELTA priority increment in [0, 3].
 */
template<int DELTA>
__device__ __forceinline__ void boost_priority() {
    static_assert(DELTA >= 0 && DELTA <= 3, "s_setprio_inc_wg takes a 2-bit constant");
    __builtin_amdgcn_s_setprio_inc_wg(static_cast<short>(DELTA));
}

/**
 * @brief Compiler-only scheduling fence; emits no hardware op.
 *
 * Lowers to `__builtin_amdgcn_sched_barrier(0)`, which tells the compiler's
 * instruction scheduler not to move instructions across this point. Purely
 * a compile-time hint -- it costs nothing at runtime -- but it pins code
 * order that the scheduler would otherwise be free to rearrange.
 *
 * Typical use: between issuing async loads and the matrix multiplies that
 * consume them, so the compiler doesn't sink a load past its consumer or
 * regroup unrelated multiplies together.
 */
__device__ __forceinline__ void compiler_fence() {
    __builtin_amdgcn_sched_barrier(0);
}

} // namespace sched
} // namespace kittens

