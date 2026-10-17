/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in tensor memory.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
namespace detail {
namespace tcgen05 {
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-descriptor
template<typename D, typename AB, int M, int N, bool trans_a, bool trans_b, bool neg=false>
__device__ static inline uint32_t instruction_descriptor() {
    uint32_t desc = 0;
    if constexpr (sizeof(AB) == 2) { // kind::f16
        // either accumulate to float, or the input is half and the output is half
        static_assert(std::is_same_v<D, float> || std::is_same_v<AB, half>);
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b0       << 3;  // no saturate on fp types
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D matrix is FP32
        }
        else {
            desc |= 0b00  << 4; // D matrix is FP16
        }
        desc |= 0b0       << 6;  // reserved
        if constexpr (std::is_same_v<AB, half>) {
            desc |= 0b000 << 7;  // 16-bit A input type as FP16
            desc |= 0b000 << 10; // 16-bit B input type as FP16
        } else if constexpr (std::is_same_v<AB, bf16>) {
            desc |= 0b001 << 7;  // 16-bit A input type as BF16
            desc |= 0b001 << 10; // 16-bit B input type as BF16
        } else if constexpr (std::is_same_v<AB, fp8e4m3>) {
            desc |= 0b000 << 7;  // 8-bit A input type as FP8 e4m3
            desc |= 0b000 << 10; // 8-bit B input type as FP8 e4m3
        } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
            desc |= 0b001 << 7;  // 8-bit A input type as FP8 e5m2
            desc |= 0b001 << 10; // 8-bit B input type as FP8 e5m2
        }
        /* fp6 and fp4
        else if constexpr (std::is_same_v<AB, fp6e2m3>) {
            desc |= 0b011 << 7;  // 6-bit A input type as FP6 e2m3
            desc |= 0b011 << 10; // 6-bit B input type as FP6 e2m3
        }
        else if constexpr (std::is_same_v<AB, fp4e2m3>) {
            desc |= 0b100 << 7;  // 6-bit A input type as FP6 e3m2
            desc |= 0b100 << 10; // 6-bit B input type as FP6 e3m2
        }
        else if constexpr (std::is_same_v<AB, fp4e3m1>) {
            desc |= 0b101 << 7;  // 4-bit A input type as FP4 e3m1
            desc |= 0b101 << 10; // 4-bit B input type as FP4 e3m1
        }
        */
        if constexpr (neg) {
            desc |= 0b1   << 13; // Do negate A matrix
        }
        else {
            desc |= 0b0   << 13; // Don't negate A matrix
        }
        desc |= 0b0       << 14; // Don't negate B matrix (in all cases)
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // Transpose A matrix
        }
        else {
            desc |= 0b0   << 15; // Don't transpose A matrix
        }
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // Transpose B matrix
        }
        else {
            desc |= 0b0  << 16; // Don't transpose B matrix
        }
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // reserved
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // reserved
        desc |= 0b00     << 30; // no shift for B-matrix reuse
    } else if constexpr (sizeof(AB) == 1) { // kind::f8f6f4
        static_assert(std::is_same_v<D, float> || std::is_same_v<D, half>); // FP8/6/4 has to accumulate to float or half
        desc |= 0b00      << 0;  // sparsity bits unneeded
        desc |= 0b0       << 2;  // dense
        desc |= 0b0       << 3;  // no saturate on fp types
        if constexpr (std::is_same_v<D, float>) {
            desc |= 0b01  << 4; // D matrix is FP32
        }
        else {
            desc |= 0b00  << 4; // D matrix is FP16
        }
        desc |= 0b0       << 6;  // reserved
        if constexpr (std::is_same_v<AB, fp8e4m3>) {
            desc |= 0b000 << 7;  // 8-bit A input type as FP8 e4m3
            desc |= 0b000 << 10; // 8-bit B input type as FP8 e4m3
        } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
            desc |= 0b001 << 7;  // 8-bit A input type as FP8 e5m2
            desc |= 0b001 << 10; // 8-bit B input type as FP8 e5m2
        }
        /* fp6 and fp4
        else if constexpr (std::is_same_v<AB, fp6e2m3>) {
            desc |= 0b011 << 7;  // 6-bit A input type as FP6 e2m3
            desc |= 0b011 << 10; // 6-bit B input type as FP6 e2m3
        }
        else if constexpr (std::is_same_v<AB, fp4e2m3>) {
            desc |= 0b100 << 7;  // 6-bit A input type as FP6 e3m2
            desc |= 0b100 << 10; // 6-bit B input type as FP6 e3m2
        }
        else if constexpr (std::is_same_v<AB, fp4e3m1>) {
            desc |= 0b101 << 7;  // 4-bit A input type as FP4 e3m1
            desc |= 0b101 << 10; // 4-bit B input type as FP4 e3m1
        }
        */
        if constexpr (neg) {
            desc |= 0b1   << 13; // Do negate A matrix
        }
        else {
            desc |= 0b0   << 13; // Don't negate A matrix
        }
        desc |= 0b0       << 14; // Don't negate B matrix (in all cases)
        if constexpr (trans_a) {
            desc |= 0b1   << 15; // Transpose A matrix
        }
        else {
            desc |= 0b0   << 15; // Don't transpose A matrix
        }
        if constexpr (trans_b) {
            desc |= 0b1  << 16; // Transpose B matrix
        }
        else {
            desc |= 0b0  << 16; // Don't transpose B matrix
        }
        desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
        desc |= 0b0      << 23; // reserved
        desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
        desc |= 0b0      << 29; // reserved
        desc |= 0b00     << 30; // no shift for B-matrix reuse
    }
    else {
        static_assert(sizeof(AB) == 999, "Invalid AB type size; not implemented yet.");
    }
    return desc;
};

template<typename T_AB, int acc, int ncta=1>
__device__ static inline void tt_st(uint32_t d_tt_addr, uint32_t a_tt_addr, uint64_t b_desc, uint32_t idesc) {
    if constexpr (std::is_same_v<T_AB, fp8e4m3> || std::is_same_v<T_AB, fp8e5m2>) {
        // TODO(danfu): is there a better way to do this with string manipulation that the compiler likes?
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    } else {
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "r"(a_tt_addr), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    }
}

template<typename T_AB, int acc, int ncta=1>
__device__ static inline void st_st(uint32_t d_tt_addr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc) {
    if constexpr (std::is_same_v<T_AB, fp8e4m3> || std::is_same_v<T_AB, fp8e5m2>) {
        // TODO(danfu): is there a better way to do this with string manipulation that the compiler likes?
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    } else {
        if constexpr (ncta == 1) {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
        else {
            asm volatile(
                "{.reg .pred p;\n" \
                "setp.eq.u32 p, 1, %4;\n" \
                "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}\n"
            ::  "r"(d_tt_addr), "l"(a_desc), "l"(b_desc), "r"(idesc), "n"(acc)
            );
        }
    }
}

template<int ncta=1> __device__ static inline void commit(kittens::semaphore &sem) {
    if constexpr (ncta == 1) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
        ::  "l"(&sem)
        );
    }
    else {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
        ::  "l"(&sem), "h"((uint16_t)(0b11))
        );
    }
}

} // namespace tcgen05
} // namespace detail

template<typename T_AB> constexpr int reduction_dimension = sizeof(T_AB) == 2 ? 16 : sizeof(T_AB) == 4 ? 8 : 32; // haven't added fp4 yet.
// RS matmul equivalent
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::tt::all A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    constexpr int trans_b = 1 - n_trans_b;

    // Do everything here.
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    static_assert(M == D::rows*ncta && ((ncta == 1 && (M == 64 || M == 128)) || (ncta == 2 && (M == 128 || M == 256)))); // output register is correctly sized

    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    static_assert(N == D::cols); // output register is correctly sized

    constexpr int K = trans_a ? A::rows : A::cols;
    static_assert((trans_b ? B::rows : B::cols) == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T; static_assert(std::is_same_v<T_AB, typename B::T>);
    using T_D  = D::T;

    constexpr int red_dim = reduction_dimension<T_AB>;
    static_assert(K%red_dim == 0, "K dimension must be divisible by red_dim.");

    static_assert(
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, half>) || 
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, fp8e4m3>) ||
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, fp8e5m2>) ||
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, bf16>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, half>) ||
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, fp8e4m3>) ||
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, fp8e5m2>),
        "Currently unsupported type combination for matrix multiply."
    );
    uint32_t idesc = detail::tcgen05::instruction_descriptor<T_D, T_AB, M, N, trans_a, trans_b, false>();
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

    detail::tcgen05::template tt_st<T_AB, acc, ncta>(
        d.addr,
        a.template chunk_addr<trans_a>(0),
        b_desc.chunk_descriptor(0),
        idesc
    );
    #pragma unroll
    for(int i = 1; i < K/red_dim; i++) {
        detail::tcgen05::template tt_st<T_AB, 1, ncta>(
            d.addr,
            a.template chunk_addr<trans_a>(i),
            b_desc.chunk_descriptor(i),
            idesc
        );
    }
}
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::tt::all A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b);
    detail::tcgen05::commit<ncta>(sem);
}
// SS matmul equivalent
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    constexpr int trans_b = 1 - n_trans_b;

    // Do everything here.
    constexpr int M = (trans_a ? A::cols : A::rows) * ncta;
    static_assert(M == D::rows*ncta && ((ncta == 1 && (M == 64 || M == 128)) || (ncta == 2 && (M == 128 || M == 256)))); // output register is correctly sized

    constexpr int N = (trans_b ? B::cols : B::rows) * ncta;
    static_assert(N == D::cols); // output register is correctly sized

    constexpr int K = trans_a ? A::rows : A::cols;
    static_assert((trans_b ? B::rows : B::cols) == K); // K dimension must match
    static_assert(std::is_same_v<typename A::T, typename B::T>); // A and B must match type.

    // Usings
    using T_AB = A::T; static_assert(std::is_same_v<T_AB, typename B::T>);
    using T_D  = D::T;

    constexpr int red_dim = reduction_dimension<T_AB>;
    static_assert(K%red_dim == 0, "K dimension must be divisible by red_dim.");

    static_assert(
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, half>) || 
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, fp8e4m3>) || 
        (std::is_same_v<T_D, half> && !std::is_same_v<T_AB, fp8e5m2>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, bf16>) || 
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, half>) ||
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, fp8e4m3>) ||
        (std::is_same_v<T_D, float> && !std::is_same_v<T_AB, fp8e5m2>),
        "Currently unsupported type combination for matrix multiply."
    );
    uint32_t idesc = detail::tcgen05::instruction_descriptor<T_D, T_AB, M, N, trans_a, trans_b, false>();
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<A>, trans_a> a_desc(a);
    kittens::st_descriptor<ducks::st_descriptor::detail::get_st<B>, trans_b> b_desc(b);

    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    
    detail::tcgen05::template st_st<T_AB, acc, ncta>(
        d.addr,
        a_desc.chunk_descriptor(0),
        b_desc.chunk_descriptor(0),
        idesc
    );
    #pragma unroll
    for(int i = 1; i < K/red_dim; i++) {
        detail::tcgen05::template st_st<T_AB, 1, ncta>(
            d.addr,
            a_desc.chunk_descriptor(i),
            b_desc.chunk_descriptor(i),
            idesc
        );
    }
}
template<int trans_a, int n_trans_b, ducks::tt::all D, ducks::st_descriptor::input A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b);
    detail::tcgen05::commit<ncta>(sem);
}
// Accumulator / numcta wrappers
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b);
}

// Transpose wrappers
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}


} // namespace kittens

