
#pragma once

namespace mittens {
    
using bf16 = bfloat;
using bf16_2 = bfloat2;
using bf16_4 = bfloat4;
//using half_2 = half2;
     
namespace ducks {
namespace base_types {
template <typename T>
static METAL_FUNC constexpr const bool isT1() {
    return metal::is_same<typename T::dtype, float>::value ||
           metal::is_same<typename T::dtype, bf16 >::value ||
           metal::is_same<typename T::dtype, half>::value;
}
template <typename T>
static METAL_FUNC constexpr const bool isT2() {
    return metal::is_same<typename T::dtype, float2>::value ||
           metal::is_same<typename T::dtype, bf16_2>::value ||
           metal::is_same<typename T::dtype, half2>::value;
}
    
template <typename T>
static METAL_FUNC constexpr const bool isT1Type() {
    return metal::is_same<T, float>::value ||
           metal::is_same<T, bf16 >::value ||
           metal::is_same<T, half>::value;
}
template <typename T>
static METAL_FUNC constexpr const bool isT2Type() {
    return metal::is_same<T, float2>::value ||
           metal::is_same<T, bf16_2>::value ||
           metal::is_same<T, half2>::value;
}
    
template <typename T>
static METAL_FUNC constexpr const bool isT1Ptr() {
    return metal::is_same<T, device      float*>::value ||
           metal::is_same<T, threadgroup float*>::value ||
           metal::is_same<T, thread      float*>::value ||
           metal::is_same<T, device      bf16*>::value ||
           metal::is_same<T, threadgroup bf16*>::value ||
           metal::is_same<T, thread      bf16*>::value ||
           metal::is_same<T, device      half*>::value ||
           metal::is_same<T, threadgroup half*>::value ||
           metal::is_same<T, thread      half*>::value;
}
template <typename T>
static METAL_FUNC constexpr const bool isT2Ptr() {
    return metal::is_same<T, device      float2*>::value ||
           metal::is_same<T, threadgroup float2*>::value ||
           metal::is_same<T, thread      float2*>::value ||
           metal::is_same<T, device      bf16_2*>::value ||
           metal::is_same<T, threadgroup bf16_2*>::value ||
           metal::is_same<T, thread      bf16_2*>::value ||
           metal::is_same<T, device      half2*>::value ||
           metal::is_same<T, threadgroup half2*>::value ||
           metal::is_same<T, thread      half2*>::value;
}

template <typename T>
static METAL_FUNC constexpr const bool isTKType() { // good enough
    return !isT1Type<T>() && !isT2Type<T>() && !isT1Ptr<T>() && !isT2Ptr<T>();
}
    
} // namespace base_types
} // namespace ducks
    
/**
 * @namespace base_types
 *
 * @brief A namespace for Thundermittens basic data types.
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
    static METAL_FUNC constexpr T zero()      { return T{0}; }
    /**
     * @brief One
     * @return Constexpr one with type T
     */
    static METAL_FUNC constexpr T one()       { return T{1}; }
    /**
     * @brief Positive infinity. Particularly useful for initializing before a min op.
     * @return Constexpr positive infinity with type T
     */
    static METAL_FUNC constexpr T pos_infty() { return T{INFINITY}; } // I'll find a better way at some point but this appears to work.
    /**
     * @brief Negative infinity. Particularly useful for initializing before a max op.
     * @return Constexpr negative infinity with type T
     */
    static METAL_FUNC constexpr T neg_infty() { return T{-INFINITY}; }
};
template<> struct constants<float> {
    static METAL_FUNC constexpr float zero()      { return 0.f; }
    static METAL_FUNC constexpr float one()       { return 1.f; }
    static METAL_FUNC constexpr float pos_infty() { return INFINITY; }
    static METAL_FUNC constexpr float neg_infty() { return -INFINITY; }
};
template<> struct constants<float2> {
    static METAL_FUNC constexpr float2 zero()      { return float2(0.f, 0.f); }
    static METAL_FUNC constexpr float2 one()       { return float2(1.f, 1.f); }
    static METAL_FUNC constexpr float2 pos_infty() { return float2(constants<float>::pos_infty(), constants<float>::pos_infty()); }
    static METAL_FUNC constexpr float2 neg_infty() { return float2(constants<float>::neg_infty(), constants<float>::neg_infty()); }
};
template<> struct constants<bf16> {
    static METAL_FUNC constexpr bf16 zero()      { return 0.bf; }
    static METAL_FUNC constexpr bf16 one()       { return 1.bf; }
    static METAL_FUNC constexpr bf16 pos_infty() { return HUGE_VALBF; }
    static METAL_FUNC constexpr bf16 neg_infty() { return -HUGE_VALBF; }
};
template<> struct constants<bf16_2> {
    static METAL_FUNC constexpr bf16_2 zero()      { return bf16_2(constants<bf16>::zero(),      constants<bf16>::zero());      }
    static METAL_FUNC constexpr bf16_2 one()       { return bf16_2(constants<bf16>::one(),       constants<bf16>::one());       }
    static METAL_FUNC constexpr bf16_2 pos_infty() { return bf16_2(constants<bf16>::pos_infty(), constants<bf16>::pos_infty()); }
    static METAL_FUNC constexpr bf16_2 neg_infty() { return bf16_2(constants<bf16>::neg_infty(), constants<bf16>::neg_infty()); }
};
template<> struct constants<half> {
    static METAL_FUNC constexpr half zero()      { return half(0.h); }
    static METAL_FUNC constexpr half one()       { return half(1.h); }
    static METAL_FUNC constexpr half pos_infty() { return HUGE_VALH; }
    static METAL_FUNC constexpr half neg_infty() { return -HUGE_VALH; }
};
    
template<> struct constants<half2> {
    static METAL_FUNC constexpr half2 zero()      { return half2(constants<half>::zero(),      constants<half>::zero());      }
    static METAL_FUNC constexpr half2 one()       { return half2(constants<half>::one(),       constants<half>::one());       }
    static METAL_FUNC constexpr half2 pos_infty() { return half2(constants<half>::pos_infty(), constants<half>::pos_infty()); }
    static METAL_FUNC constexpr half2 neg_infty() { return half2(constants<half>::neg_infty(), constants<half>::neg_infty()); }
};

    

/**
 * @brief Provides information about packing of elements for a given type.
 *
 * @tparam T The type for which to provide packing information.
 */
template<typename T> struct packing {
//    /**
//     * @brief The number of elements packed together.
//     *
//     * @return constexpr int representing number of elements within the type.
//     */
//    static METAL_FUNC constexpr int num() { return 1; }
//    /**
//     * @brief Packs a single T element twice (replicated) into its packed type.
//     *
//     * @param i[in] The element to pack.
//     * @return The packed type.
//     */
//    static METAL_FUNC constexpr T pack(device const bf16 &i);
//    static METAL_FUNC constexpr T pack(threadgroup const bf16 &i);
//    static METAL_FUNC constexpr T pack(thread const bf16 &i);
};

#define PACK_FUNCTIONS(T1, T2) \
    static METAL_FUNC constexpr T2 pack(device const T1 &i) { return T2{i, i};      } \
    static METAL_FUNC constexpr T2 pack(threadgroup const T1 &i) { return T2{i, i}; } \
    static METAL_FUNC constexpr T2 pack(thread const T1 &i) { return T2{i, i};      }

template<> struct packing<bf16> {
    static METAL_FUNC constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    using packed_four = bf16_4;
    PACK_FUNCTIONS(unpacked_type, packed_type)
};
template<> struct packing<half> {
    static METAL_FUNC constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type = half2;
    using packed_four = half4;
    PACK_FUNCTIONS(unpacked_type, packed_type)
};
template<> struct packing<float> {
    static METAL_FUNC constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = float2;
    using packed_four = float4;
    
    PACK_FUNCTIONS(unpacked_type, packed_type)
};
template<> struct packing<bf16_2> {
    static METAL_FUNC constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    using packed_four = bf16_4;
    PACK_FUNCTIONS(unpacked_type, packed_type)
};
template<> struct packing<half2> {
    static METAL_FUNC constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type = half2;
    using packed_four = half4;
    PACK_FUNCTIONS(unpacked_type, packed_type)
};
template<> struct packing<float2> {
    static METAL_FUNC constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = float2;
    using packed_four = float4;
    PACK_FUNCTIONS(unpacked_type, packed_type)
};
template<> struct packing<int2> {
    static METAL_FUNC constexpr int num() { return 2; }
};
template<> struct packing<float4> {
    static METAL_FUNC constexpr int num() { return 4; }
};
template<> struct packing<int4> {
    static METAL_FUNC constexpr int num() { return 4; }
};

    
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
    static METAL_FUNC T convert(device const U & u)      { return (T)u; }
    static METAL_FUNC T convert(threadgroup const U & u) { return (T)u; }
    static METAL_FUNC T convert(thread const U & u)      { return (T)u; }
};
    
template<> struct convertor<float, bf16> {
    // fptrunc float %_ to bfloat
    static METAL_FUNC float convert(device const bf16 & u)      { return float(u);}
    static METAL_FUNC float convert(threadgroup const bf16 & u) { return float(u);}
    static METAL_FUNC float convert(thread const bf16 & u)      { return float(u);}
};
template<> struct convertor<bf16, float> {
    // fpext bfloat %_ to float
    static METAL_FUNC bf16 convert(device const float & u)      { return bf16(u); }
    static METAL_FUNC bf16 convert(threadgroup const float & u) { return bf16(u); }
    static METAL_FUNC bf16 convert(thread const float & u)      { return bf16(u); }
};
template<> struct convertor<float2, bf16_2> {
    // tail call fast <2 x float> @air.convert.f.v2f32.f.v2bf16(<2 x bfloat> %_)
    static METAL_FUNC float2 convert(device const bf16_2 & u)      { return float2(u); }
    static METAL_FUNC float2 convert(threadgroup const bf16_2 & u) { return float2(u); }
    static METAL_FUNC float2 convert(thread const bf16_2 & u)      { return float2(u); }
};
template<> struct convertor<bf16_2, float2> {
    // tail call fast <2 x bfloat> @air.convert.f.v2bf16.f.v2f32(<2 x float> %_)
    static METAL_FUNC bf16_2 convert(device const float2 & u)      { return bf16_2(u); }
    static METAL_FUNC bf16_2 convert(threadgroup const float2 & u) { return bf16_2(u); }
    static METAL_FUNC bf16_2 convert(thread const float2 & u)      { return bf16_2(u); }
};
    
template<> struct convertor<float, half> {
    // fptrunc float %_ to half
    static METAL_FUNC float convert(device const half & u)      { return float(u); }
    static METAL_FUNC float convert(threadgroup const half & u) { return float(u); }
    static METAL_FUNC float convert(thread const half & u)      { return float(u); }
};
template<> struct convertor<half, float> {
    //fpext half %_ to float
    static METAL_FUNC half convert(device const float & u)      { return half(u); }
    static METAL_FUNC half convert(threadgroup const float & u) { return half(u); }
    static METAL_FUNC half convert(thread const float & u)      { return half(u); }
};
template<> struct convertor<float2, half2> {
    // tail call fast <2 x float> @air.convert.f.v2f32.f.v2f16(<2 x half> %_)
    static METAL_FUNC float2 convert(device const half2 & u)      { return float2(u); }
    static METAL_FUNC float2 convert(threadgroup const half2 & u) { return float2(u); }
    static METAL_FUNC float2 convert(thread const half2 & u)      { return float2(u); }
};
template<> struct convertor<half2, float2> {
    // tail call fast <2 x half> @air.convert.f.v2f16.f.v2f32(<2 x float> %_)
    static METAL_FUNC half2 convert(device const float2 & u)      { return half2(u); }
    static METAL_FUNC half2 convert(threadgroup const float2 & u) { return half2(u); }
    static METAL_FUNC half2 convert(thread const float2 & u)      { return half2(u); }
};
template<> struct convertor<bf16, half> {
    static METAL_FUNC bf16 convert(device const half & u)      { return bf16(u); }
    static METAL_FUNC bf16 convert(threadgroup const half & u) { return bf16(u); }
    static METAL_FUNC bf16 convert(thread const half & u)      { return bf16(u); }
};
template<> struct convertor<half, bf16> {
    static METAL_FUNC half convert(device const bf16 & u)      { return half(u); }
    static METAL_FUNC half convert(threadgroup const bf16 & u) { return half(u); }
    static METAL_FUNC half convert(thread const bf16 & u)      { return half(u); }
};
template<> struct convertor<bf16_2, half2> {
    // tail call fast <2 x bfloat> @air.convert.f.v2bf16.f.v2f16(<2 x half> %_)
    static METAL_FUNC bf16_2 convert(device const half2 & u)      { return bf16_2(u); }
    static METAL_FUNC bf16_2 convert(threadgroup const half2 & u) { return bf16_2(u); }
    static METAL_FUNC bf16_2 convert(thread const half2 & u)      { return bf16_2(u); }
};
template<> struct convertor<half2, bf16_2> {
    // tail call fast <2 x half> @air.convert.f.v2f16.f.v2bf16(<2 x bfloat> %_)
    static METAL_FUNC half2 convert(device const bf16_2 & u)      { return half2(u); }
    static METAL_FUNC half2 convert(threadgroup const bf16_2 & u) { return half2(u); }
    static METAL_FUNC half2 convert(thread const bf16_2 & u)      { return half2(u); }
};
    
    
    
} // base_types

} // mittens
