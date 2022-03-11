/*
 * Copyright 2013 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <math/vec2.h>
#include <math/half.h>
#include <stdint.h>
#include <sys/types.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"

namespace android {
// -------------------------------------------------------------------------------------

namespace details {

template <typename T>
class TVec3 :   public TVecProductOperators<TVec3, T>,
                public TVecAddOperators<TVec3, T>,
                public TVecUnaryOperators<TVec3, T>,
                public TVecComparisonOperators<TVec3, T>,
                public TVecFunctions<TVec3, T>,
                public TVecDebug<TVec3, T> {
public:
    enum no_init { NO_INIT };
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef size_t size_type;

    union {
        struct { T x, y, z; };
        struct { T s, t, p; };
        struct { T r, g, b; };
        TVec2<T> xy;
        TVec2<T> st;
        TVec2<T> rg;
    };

    static constexpr size_t SIZE = 3;
    inline constexpr size_type size() const { return SIZE; }

    // array access
    inline constexpr T const& operator[](size_t i) const {
#if __cplusplus >= 201402L
        // only possible in C++0x14 with constexpr
        assert(i < SIZE);
#endif
        return (&x)[i];
    }

    inline T& operator[](size_t i) {
        assert(i < SIZE);
        return (&x)[i];
    }

    // -----------------------------------------------------------------------
    // we want the compiler generated versions for these...
    TVec3(const TVec3&) = default;
    ~TVec3() = default;
    TVec3& operator = (const TVec3&) = default;

    // constructors
    // leaves object uninitialized. use with caution.
    explicit
    constexpr TVec3(no_init) { }

    // default constructor
    constexpr TVec3() : x(0), y(0), z(0) { }

    // handles implicit conversion to a tvec4. must not be explicit.
    template<typename A, typename = typename std::enable_if<std::is_arithmetic<A>::value >::type>
    constexpr TVec3(A v) : x(v), y(v), z(v) { }

    template<typename A, typename B, typename C>
    constexpr TVec3(A x, B y, C z) : x(x), y(y), z(z) { }

    template<typename A, typename B>
    constexpr TVec3(const TVec2<A>& v, B z) : x(v.x), y(v.y), z(z) { }

    template<typename A>
    explicit
    constexpr TVec3(const TVec3<A>& v) : x(v.x), y(v.y), z(v.z) { }

    // cross product works only on vectors of size 3
    template <typename RT>
    friend inline
    constexpr TVec3 cross(const TVec3& u, const TVec3<RT>& v) {
        return TVec3(
                u.y*v.z - u.z*v.y,
                u.z*v.x - u.x*v.z,
                u.x*v.y - u.y*v.x);
    }
};

}  // namespace details

// ----------------------------------------------------------------------------------------

typedef details::TVec3<double> double3;
typedef details::TVec3<float> float3;
typedef details::TVec3<float> vec3;
typedef details::TVec3<half> half3;
typedef details::TVec3<int32_t> int3;
typedef details::TVec3<uint32_t> uint3;
typedef details::TVec3<int16_t> short3;
typedef details::TVec3<uint16_t> ushort3;
typedef details::TVec3<int8_t> byte3;
typedef details::TVec3<uint8_t> ubyte3;
typedef details::TVec3<bool> bool3;

// ----------------------------------------------------------------------------------------
}  // namespace android

#pragma clang diagnostic pop
