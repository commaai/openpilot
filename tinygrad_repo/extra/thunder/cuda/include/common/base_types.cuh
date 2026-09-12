/**
 * @file
 * @brief Declarations, manipulations, and wrappers for basic types.
 * 
 * This file is a bunch of utilities for going back and forth between different types.
 * 
 * Many of them are for the compiler, so as to clean up the code. It unfortunately
 * seems necessary when we have types we really care about that are less than word width.
 */

#pragma once

#ifdef KITTENS_HOPPER
#include <cuda_fp8.h>
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <string>
#include <bit>


namespace kittens {

/**
 * @brief Bfloat16 floating-point type.
 */
using bf16 = __nv_bfloat16;
/**
 * @brief Half-precision floating-point type.
 */
using half = __half;
/**
 * @brief Packed word of two bfloat16 floating-point values.
 */
using bf16_2 = __nv_bfloat162;
/**
 * @brief Packed word of two half-precision floating-point values.
 */
using half_2 = __half2;
#ifdef KITTENS_HOPPER
/**
 * @brief float8 floating-point type.
 */
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
#ifdef KITTENS_BLACKWELL
using fp8e8m0 = __nv_fp8_e8m0;
#endif
/**
 * @brief 2-packed float8 floating-point type.
 */
using fp8e4m3_2 = __nv_fp8x2_e4m3;
using fp8e5m2_2 = __nv_fp8x2_e5m2;
#ifdef KITTENS_BLACKWELL
using fp8e8m0_2 = __nv_fp8x2_e8m0;
#endif
/**
 * @brief 4-packed float8 floating-point type.
 */
using fp8e4m3_4 = __nv_fp8x4_e4m3;
using fp8e5m2_4 = __nv_fp8x4_e5m2;
#ifdef KITTENS_BLACKWELL
using fp8e8m0_4 = __nv_fp8x4_e8m0;
#endif
#endif

namespace ducks {
/**
 * @namespace base_types
 *
 * @brief A namespace for concepts for basic data types.
 */
namespace base_types {

#ifdef KITTENS_HOPPER
#ifdef KITTENS_BLACKWELL
template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2> || std::is_same_v<T, fp8e4m3_4> || std::is_same_v<T, fp8e5m2_4> || std::is_same_v<T, fp8e8m0_4>; // could add half_2 later if implemented.
template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half> || std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2> || std::is_same_v<T, fp8e8m0>; // could add half_2 later if implemented.
#else
template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2> || std::is_same_v<T, fp8e4m3_4> || std::is_same_v<T, fp8e5m2_4>;
template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half> || std::is_same_v<T, fp8e4m3> || std::is_same_v<T, fp8e5m2>;
#endif
#else
template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2>;
template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half>;
#endif

} // namespace base_types
} // namespace ducks

/**
 * @namespace base_types
 *
 * @brief A namespace for ThunderKittens basic data types.
 */
namespace base_types {

/**
 * @brief Provides compile-time constants for different types.
 *
 * @tparam T The type for which to provide constants.
 */
template<typename T> struct constants {
    /**
     * @brief Zero
     * @return Constexpr zero with type T
     */
    static __device__ inline constexpr T zero()      { return T{0}; }
    /**
     * @brief One
     * @return Constexpr one with type T
     */
    static __device__ inline constexpr T one()       { return T{1}; }
    /**
     * @brief Positive infinity. Particularly useful for initializing before a min op.
     * @return Constexpr positive infinity with type T
     */
    static __device__ inline constexpr T pos_infty() { return T{INFINITY}; } // I'll find a better way at some point but this appears to work.
    /**
     * @brief Negative infinity. Particularly useful for initializing before a max op.
     * @return Constexpr negative infinity with type T
     */
    static __device__ inline constexpr T neg_infty() { return T{-INFINITY}; }
};
template<> struct constants<float2> {
    static __device__ inline constexpr float2 zero()      { return float2{0.f, 0.f}; }
    static __device__ inline constexpr float2 one()       { return float2{1.f, 1.f}; }
    static __device__ inline constexpr float2 pos_infty() { return float2{constants<float>::pos_infty(), constants<float>::pos_infty()}; }
    static __device__ inline constexpr float2 neg_infty() { return float2{constants<float>::neg_infty(), constants<float>::neg_infty()}; }
};
template<> struct constants<bf16> {
    static __device__ inline constexpr bf16 zero()      { return std::bit_cast<__nv_bfloat16>(uint16_t(0x0000)); } // unfortunately __float2bf16_rn is not constexpr
    static __device__ inline constexpr bf16 one()       { return std::bit_cast<__nv_bfloat16>(uint16_t(0x3F80)); }
    static __device__ inline constexpr bf16 pos_infty() { return std::bit_cast<__nv_bfloat16>(uint16_t(0x7F80)); }
    static __device__ inline constexpr bf16 neg_infty() { return std::bit_cast<__nv_bfloat16>(uint16_t(0xFF80)); }
};
template<> struct constants<bf16_2> {
    static __device__ inline constexpr bf16_2 zero()      { return bf16_2{constants<bf16>::zero(),      constants<bf16>::zero()};      }
    static __device__ inline constexpr bf16_2 one()       { return bf16_2{constants<bf16>::one(),       constants<bf16>::one()};       }
    static __device__ inline constexpr bf16_2 pos_infty() { return bf16_2{constants<bf16>::pos_infty(), constants<bf16>::pos_infty()}; }
    static __device__ inline constexpr bf16_2 neg_infty() { return bf16_2{constants<bf16>::neg_infty(), constants<bf16>::neg_infty()}; }
};
template<> struct constants<half> {
    static __device__ inline constexpr half zero()      { return std::bit_cast<__half>(uint16_t(0x0000)); }
    static __device__ inline constexpr half one()       { return std::bit_cast<__half>(uint16_t(0x3C00)); }
    static __device__ inline constexpr half pos_infty() { return std::bit_cast<__half>(uint16_t(0x7C00)); }
    static __device__ inline constexpr half neg_infty() { return std::bit_cast<__half>(uint16_t(0xFC00)); }
};
template<> struct constants<half_2> {
    static __device__ inline constexpr half_2 zero()      { return half_2{constants<half>::zero(),      constants<half>::zero()};      }
    static __device__ inline constexpr half_2 one()       { return half_2{constants<half>::one(),       constants<half>::one()};       }
    static __device__ inline constexpr half_2 pos_infty() { return half_2{constants<half>::pos_infty(), constants<half>::pos_infty()}; }
    static __device__ inline constexpr half_2 neg_infty() { return half_2{constants<half>::neg_infty(), constants<half>::neg_infty()}; }
};
#ifdef KITTENS_HOPPER
template<> struct constants<fp8e4m3> {
    static __device__ inline constexpr fp8e4m3 zero() { return std::bit_cast<__nv_fp8_e4m3>(uint8_t(0x00)); }
    static __device__ inline constexpr fp8e4m3 one() { return std::bit_cast<__nv_fp8_e4m3>(uint8_t(0x38)); }
};
template<> struct constants<fp8e4m3_2> {
    static __device__ inline constexpr fp8e4m3_2 zero() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x0000)); }
    static __device__ inline constexpr fp8e4m3_2 one() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x3838)); }
};
template<> struct constants<fp8e4m3_4> {
    static __device__ inline constexpr fp8e4m3_4 zero() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x00000000)); }
    static __device__ inline constexpr fp8e4m3_4 one() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x38383838)); }
};
template<> struct constants<fp8e5m2> {
    static __device__ inline constexpr fp8e5m2 zero() { return std::bit_cast<__nv_fp8_e5m2>(uint8_t(0x00)); }
    static __device__ inline constexpr fp8e5m2 one() { return std::bit_cast<__nv_fp8_e5m2>(uint8_t(0x3C)); }
};
template<> struct constants<fp8e5m2_2> {
    static __device__ inline constexpr fp8e5m2_2 zero() { return std::bit_cast<fp8e5m2_2>(uint16_t(0x0000)); }
    static __device__ inline constexpr fp8e5m2_2 one() { return std::bit_cast<fp8e5m2_2>(uint16_t(0x3C3C)); }
};
template<> struct constants<fp8e5m2_4> {
    static __device__ inline constexpr fp8e5m2_4 zero() { return std::bit_cast<fp8e5m2_4>(uint32_t(0x00000000)); }
    static __device__ inline constexpr fp8e5m2_4 one() { return std::bit_cast<fp8e5m2_4>(uint32_t(0x3C3C3C3C)); }
};
#endif

template<> struct constants<int> {
    static __device__ inline constexpr int zero()      { return 0; }
    static __device__ inline constexpr int one()       { return 1; }
};
template<> struct constants<int2> {
    static __device__ inline constexpr int2 zero()      { return int2{0, 0}; }
    static __device__ inline constexpr int2 one()       { return int2{1, 1}; }
};

/**
 * @brief Provides information about packing of elements for a given type.
 *
 * @tparam T The type for which to provide packing information.
 */
template<typename T> struct packing {
    /**
     * @brief The number of elements packed together.
     *
     * @return constexpr int representing number of elements within the type.
     */
    static __device__ inline constexpr int num() { return 1; }
    /**
     * @brief Packs a single T element twice (replicated) into its packed type.
     *
     * @param i[in] The element to pack.
     * @return The packed type.
     */
    static __device__ inline constexpr T pack(const bf16 &i);
};
template<> struct packing<bf16> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<bf16_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<half> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};
template<> struct packing<half_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct packing<float2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<char> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = char;
    using packed_type = char2;
    static __device__ inline constexpr char2 pack(const char &i) { return char2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<char2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = char;
    using packed_type = char2;
    static __device__ inline constexpr char2 pack(const char &i) { return char2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<uint> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = uint;
    using packed_type = uint2;
    static __device__ inline constexpr uint2 pack(const uint &i) { return uint2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<uint2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = uint;
    using packed_type = uint2;
    static __device__ inline constexpr uint2 pack(const uint &i) { return uint2{i, i}; } // this replication makes code cleaner later.
};
struct uint64_2 { uint64_t x, y; };
template<> struct packing<uint64_t> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = uint64_t;
    using packed_type = uint64_2;
    static __device__ inline constexpr uint64_2 pack(const uint64_t &i) { return uint64_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<uint64_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = uint64_t;
    using packed_type = uint64_2;
    static __device__ inline constexpr uint64_2 pack(const uint64_t &i) { return uint64_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float4> {
    static __device__ inline constexpr int num() { return 4; }
};
template<> struct packing<int4> {
    static __device__ inline constexpr int num() { return 4; }
};
#ifdef KITTENS_HOPPER
template<> struct packing<fp8e4m3> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e4m3_4> {
    static __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e5m2> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e5m2;
    using packed_type = fp8e5m2_4;
};
template<> struct packing<fp8e5m2_4> {
    static __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e5m2;
    using packed_type = fp8e5m2_4;
};
#ifdef KITTENS_BLACKWELL
template<> struct packing<fp8e8m0> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e8m0;
    using packed_type = fp8e8m0_4;
};
template<> struct packing<fp8e8m0_4> {
    static __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e8m0;
    using packed_type = fp8e8m0_4;
};
#endif
#endif


/**
 * @brief Provides templated functionality to convert between different types.
 *
 * @tparam T The target type for conversion.
 * @tparam U The source type for conversion.
 */
template<typename T, typename U> struct convertor {
    /**
     * @brief Converts a value of type U to type T.
     *
     * @param u[in] The value of type U to convert.
     * @return T The converted value of type T.
     */
    static __host__ __device__ inline T convert(const U & u) {
        return (T)u;
    }
};
template<> struct convertor<float, bf16> {
    static __host__ __device__ inline float convert(const bf16 & u) {
        return 	__bfloat162float(u);
    }
};
template<> struct convertor<bf16, float> {
    static __host__ __device__ inline bf16 convert(const float & u) {
        return 	__float2bfloat16_rn(u);
    }
};
template<> struct convertor<float2, bf16_2> {
    static __host__ __device__ inline float2 convert(const bf16_2 & u) {
        return 	__bfloat1622float2(u);
    }
};
template<> struct convertor<bf16_2, float2> {
    static __host__ __device__ inline bf16_2 convert(const float2 & u) {
        return 	__float22bfloat162_rn(u);
    }
};
template<> struct convertor<float, half> {
    static __host__ __device__ inline float convert(const half & u) {
        return __half2float(u);
    }
};
template<> struct convertor<half, float> {
    static __host__ __device__ inline half convert(const float & u) {
        return __float2half(u);
    }
};
template<> struct convertor<float2, half_2> {
    static __host__ __device__ inline float2 convert(const half_2 & u) {
        return __half22float2(u);
    }
};
template<> struct convertor<half_2, float2> {
    static __host__ __device__ inline half_2 convert(const float2 & u) {
        return __float22half2_rn(u);
    }
};
template<> struct convertor<bf16, half> {
    static __host__ __device__ inline bf16 convert(const half & u) {
        return __float2bfloat16_rn(__half2float(u));
    }
};
template<> struct convertor<half, bf16> {
    static __host__ __device__ inline half convert(const bf16 & u) {
        return __float2half(__bfloat162float(u));
    }
};
template<> struct convertor<bf16_2, half_2> {
    static __host__ __device__ inline bf16_2 convert(const half_2 & u) {
        return __float22bfloat162_rn(__half22float2(u));
    }
};
template<> struct convertor<half_2, bf16_2> {
    static __host__ __device__ inline half_2 convert(const bf16_2 & u) {
        return __float22half2_rn(__bfloat1622float2(u));
    }
};
#ifdef KITTENS_HOPPER
// fp8e4m3
template<> struct convertor<fp8e4m3_4, float4> {
    static __host__ __device__ inline fp8e4m3_4 convert(const float4& u) {
        return __nv_fp8x4_e4m3(u); 
    }
};
template<> struct convertor<float4, fp8e4m3_4> {
    static __host__ __device__ inline float4 convert(const fp8e4m3_4& u) {
        __nv_fp8_e4m3 *vals = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<__nv_fp8x4_e4m3*>(&u));
        return make_float4(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]));
    }
};
template<> struct convertor<fp8e4m3_2, float2> {
    static __host__ __device__ inline fp8e4m3_2 convert(const float2& u) {
        return __nv_fp8x2_e4m3(u); 
    }
};
template<> struct convertor<float2, fp8e4m3_2> {
    static __host__ __device__ inline float2 convert(const fp8e4m3_2& u) {
        __nv_fp8_e4m3 *vals = reinterpret_cast<__nv_fp8_e4m3*>(const_cast<__nv_fp8x2_e4m3*>(&u));
        return make_float2(float(vals[0]), float(vals[1]));
    }
};
template<> struct convertor<fp8e4m3, float> {
    static __host__ __device__ inline fp8e4m3 convert(const float & u) {
        return __nv_fp8_e4m3(u);
    }
};
template<> struct convertor<float, fp8e4m3> {
    static __host__ __device__ inline float convert(const fp8e4m3 & u) {
        return float(u);
    }
};
template<> struct convertor<bf16_2, fp8e4m3_4> {
    static __host__ __device__ inline bf16_2 convert(const fp8e4m3_4 & u) {
        float4 f4 = convertor<float4, fp8e4m3_4>::convert(u);
        float2 f2 = make_float2(f4.x, f4.y);
        return __float22bfloat162_rn(f2);
    }
};
template<> struct convertor<fp8e4m3_4, bf16_2> {
    static __host__ __device__ inline fp8e4m3_4 convert(const bf16_2 & u) {
        float2 f2 = __bfloat1622float2(u);
        float4 f4 = make_float4(f2.x, f2.y, 0.0f, 0.0f);
        return __nv_fp8x4_e4m3(f4);
    }
};
// fp8e5m2
template<> struct convertor<fp8e5m2_4, float4> {
    static __host__ __device__ inline fp8e5m2_4 convert(const float4& u) {
        return __nv_fp8x4_e5m2(u); 
    }
};
template<> struct convertor<float4, fp8e5m2_4> {
    static __host__ __device__ inline float4 convert(const fp8e5m2_4& u) {
        __nv_fp8_e5m2 *vals = reinterpret_cast<__nv_fp8_e5m2*>(const_cast<__nv_fp8x4_e5m2*>(&u));
        return make_float4(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]));
    }
};
template<> struct convertor<fp8e5m2_2, float2> {
    static __host__ __device__ inline fp8e5m2_2 convert(const float2& u) {
        return __nv_fp8x2_e5m2(u); 
    }
};
template<> struct convertor<float2, fp8e5m2_2> {
    static __host__ __device__ inline float2 convert(const fp8e5m2_2& u) {
        __nv_fp8_e5m2 *vals = reinterpret_cast<__nv_fp8_e5m2*>(const_cast<__nv_fp8x2_e5m2*>(&u));
        return make_float2(float(vals[0]), float(vals[1]));
    }
};
template<> struct convertor<fp8e5m2, float> {
    static __host__ __device__ inline fp8e5m2 convert(const float & u) {
        return __nv_fp8_e5m2(u);
    }
};
template<> struct convertor<float, fp8e5m2> {
    static __host__ __device__ inline float convert(const fp8e5m2 & u) {
        return float(u);
    }
};
template<> struct convertor<bf16_2, fp8e5m2_4> {
    static __host__ __device__ inline bf16_2 convert(const fp8e5m2_4 & u) {
        float4 f4 = convertor<float4, fp8e5m2_4>::convert(u);
        float2 f2 = make_float2(f4.x, f4.y);
        return __float22bfloat162_rn(f2);
    }
};
template<> struct convertor<fp8e5m2_4, bf16_2> {
    static __host__ __device__ inline fp8e5m2_4 convert(const bf16_2 & u) {
        float2 f2 = __bfloat1622float2(u);
        float4 f4 = make_float4(f2.x, f2.y, 0.0f, 0.0f);
        return __nv_fp8x4_e5m2(f4);
    }
};
#endif
}
}
