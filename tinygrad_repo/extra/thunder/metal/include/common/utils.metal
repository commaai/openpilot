/**
 * @file
 * @brief General utilities for Thundermittens.
 */
#pragma once // not done
/*
 TODO:
   shared allocator
   max shared mem for other hardware
 */

#include <metal_stdlib>
#include "base_types.metal"
/**
 * @namespace mittens
 *
 * @brief The main namespace of Thundermittens.
 */
namespace mittens {
/**
 * @namespace ore
 *
 * @brief The main namespace of Thundermittens Metal.
 */

/* ----------  GENERAL CONSTANTS FOR mittens  ---------- */

/**
 * @brief Tile dimension constant.
 */
constant constexpr const int TILE_DIM{8};
constant constexpr const int TILE_ELEMENTS{TILE_DIM*TILE_DIM};
constant constexpr const int SIMD_THREADS{32};
    
    
#ifdef M2_PRO
constant constexpr int MAX_SHARED_MEMORY = 32768;
#else
constant constexpr int MAX_SHARED_MEMORY = 32768;
#endif
/* ----------  TYPE HELPERS  ---------- */
/**
 * @namespace ducks
 *
 * @brief Thundermittens' namespace for template metaprogramming..
 *
 * This includes primarily dummy types and concept wrappers, along
 * with a few additional utilities.
 */
namespace ducks {

/**
 * @brief A type representing an empty default for a template.
 */
struct default_type {};

// This macro can't be done as a template, so it doesn't really have a location in mittens.
#define typeof(A) typename std::remove_const<typename std::remove_reference<decltype(A)>::type>::type

    
}
    
/* ----------  SHUFFLE UTILS  ---------- */
/**
 * @brief Mask constant for all active threads in a warp.
 */
constant static constexpr uint32_t MASK_ALL = 0xFFFFFFFF;
    
template<typename T>
static METAL_FUNC T shfl_sync(thread const T &f, const ushort laneid) {
    return metal::simd_shuffle(f, laneid);
}
    
template<>
METAL_FUNC bfloat shfl_sync<bfloat>(thread const bf16 &f, const ushort laneid) {
//    return as_type<bf16>(metal::simd_shuffle(*(thread half*)(&f), laneid));
    float f_val = (float)f;
    float shfl_val = metal::simd_shuffle(f_val, laneid);
    return (bf16)shfl_val;
}
    
template<>
METAL_FUNC bfloat2 shfl_sync<bfloat2>(thread const bf16_2 &f, const ushort laneid) {
//    return as_type<bf16_2>(metal::simd_shuffle(*(thread half2*)(&f), laneid));
    float2 f_val = (float2)f;
    float2 shfl_val = metal::simd_shuffle(f_val, laneid);
    return (bf16_2)shfl_val;
}
    
template<typename T>
static METAL_FUNC T shfl_down_fill_sync(thread const T &f, thread const T& fill_data, const ushort laneid) {
    return metal::simd_shuffle_and_fill_down(f, laneid, fill_data);
}
    
template<>
METAL_FUNC bfloat shfl_down_fill_sync<bfloat>(thread const bfloat &f, thread const bfloat &fill_data, const ushort laneid) {
//    return as_type<bf16>(metal::simd_shuffle_and_fill_down(*(thread half*)(&f), *(thread half*)(&fill_data), laneid));
    float f_val = (float)f;
    float fill_data_f = (float)fill_data;
    float shfl_val = metal::simd_shuffle_and_fill_down(f_val, fill_data_f, laneid);
    return (bf16)shfl_val;
}
template<>
METAL_FUNC bfloat2 shfl_down_fill_sync<bfloat2>(thread const bfloat2 &f, thread const bfloat2 &fill_data, const ushort laneid) {
//    return as_type<bf16_2>(metal::simd_shuffle_and_fill_down(*(thread half2*)(&f), *(thread half2*)(&fill_data), laneid));
    float2 f_val = (float2)f;
    float2 fill_data_f = (float2)fill_data;
    float2 shfl_val = metal::simd_shuffle_and_fill_down(f_val, fill_data_f, laneid);
    return (bf16_2)shfl_val;
}
/**
 * @brief Perform a shuffle down operation on a packed type synchronously across a warp.
 * @tparam T The type of the value to be shuffled.
 * @param mask[in] The mask of active threads.
 * @param f[in] The value to be shuffled.
 * @param delta[in] The number of positions to shuffle down.
 * @return The result of the shuffle operation.
 */
template<typename T>
static METAL_FUNC T shfl_down_sync(thread const T &f, int delta) {
    return metal::simd_shuffle_rotate_down(f, delta);
}

template<>
METAL_FUNC bfloat shfl_down_sync<bfloat>(thread const bf16 &f, int delta) {
//    return base_types::convertor<bf16, float>::convert(metal::simd_shuffle_rotate_down(base_types::convertor<float, bf16>::convert(f), delta));
//    return as_type<bf16>(metal::simd_shuffle_rotate_down(*(thread half*)(&f), delta));
    float f_val = (float)f;
    float shfl_val = metal::simd_shuffle_rotate_down(f_val, delta);
    return (bf16)shfl_val;
}

template<>
METAL_FUNC bfloat2 shfl_down_sync<bfloat2>(thread const bf16_2 &f, int delta) {
//    return as_type<bf16_2>(metal::simd_shuffle_rotate_down(*(thread const half2*)(&f), delta));
//    return base_types::convertor<bf16_2, float2>::convert(metal::simd_shuffle_rotate_down(base_types::convertor<float2, bf16_2>::convert(f), delta));
    
    float2 f_val = (float2)f;
    float2 shfl_val = metal::simd_shuffle_rotate_down(f_val, delta);
    return (bf16_2)shfl_val;
//    return as_type<bf16_2>(metal::simd_shuffle_rotate_down(*(thread half2*)(&f), delta));
}
    
    
/* ----------  LOOP UNROLLING UTILS  ---------- */
    
namespace meta {
template <int Start, int End, int Stride, bool = (Start < End)>
struct unroll_i_in_range {
    template<class F, typename... Args>
    static METAL_FUNC void run(F f, Args... args) {
        f(Start, args...);
        unroll_i_in_range<Start + Stride, End, Stride>::run(f, args...);
    }
};

template <int Start, int End, int Stride>
struct unroll_i_in_range<Start, End, Stride, false> {
    template<class F, typename... Args>
    static METAL_FUNC void run(F, Args...) {
    }
};


template <int Start, int End, int Stride, bool = (Start < End)>
struct unroll_i_j_in_range_inner {
    template<class F, typename... Args>
    static METAL_FUNC void run(F f, int outerIndex, Args... args) {
        f(outerIndex, Start, args...);
        unroll_i_j_in_range_inner<Start + Stride, End, Stride>::run(f, outerIndex, args...);
    }
};

template <int Start, int End, int Stride>
struct unroll_i_j_in_range_inner<Start, End, Stride, false> {
    template<class F, typename... Args>
    static METAL_FUNC void run(F, int, Args...) {
    }
};

template <int StartOuter, int EndOuter, int StrideOuter,
          int StartInner, int EndInner, int StrideInner,
          bool = (StartOuter < EndOuter)>
struct unroll_i_j_in_range {
    template<class F, typename... Args>
    static METAL_FUNC void run(F f, Args... args) {
        unroll_i_j_in_range_inner<StartInner, EndInner, StrideInner>::run(
            f, StartOuter, args...
        );
        unroll_i_j_in_range<
            StartOuter + StrideOuter, EndOuter, StrideOuter,
            StartInner, EndInner, StrideInner
        >::run(f, args...);
    }
};

template <int StartOuter, int EndOuter, int StrideOuter,
          int StartInner, int EndInner, int StrideInner>
struct unroll_i_j_in_range<StartOuter, EndOuter, StrideOuter,
                          StartInner, EndInner, StrideInner, false> {
    template<class F, typename... Args>
    static METAL_FUNC void run(F, Args...) {
    }
};
    
}


template <int N>
struct ReadVector {
    float _[N];
};

/* ----------  SHARED MEMORY UTILS  ---------- */

#define mittens_ALIGN_AS(n) alignas(n)
#define mittens_DEFAULT_ALIGN mittens_ALIGN_AS(16)
    
/**
 * @brief Dummy structure for alignment purposes. Needed for WGMMA and TMA calls.
 */
struct mittens_DEFAULT_ALIGN alignment_dummy { int dummy; };
}


