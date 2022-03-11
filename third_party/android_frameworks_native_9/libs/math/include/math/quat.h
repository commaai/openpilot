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

#include <math/half.h>
#include <math/TQuatHelpers.h>
#include <math/vec3.h>
#include <math/vec4.h>

#include <stdint.h>
#include <sys/types.h>

#ifndef PURE
#define PURE __attribute__((pure))
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"

namespace android {
// -------------------------------------------------------------------------------------

namespace details {

template <typename T>
class TQuaternion : public TVecAddOperators<TQuaternion, T>,
                    public TVecUnaryOperators<TQuaternion, T>,
                    public TVecComparisonOperators<TQuaternion, T>,
                    public TQuatProductOperators<TQuaternion, T>,
                    public TQuatFunctions<TQuaternion, T>,
                    public TQuatDebug<TQuaternion, T> {
public:
    enum no_init { NO_INIT };
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef size_t size_type;

    /*
     * quaternion internals stored as:
     *
     * q = w + xi + yj + zk
     *
     *  q[0] = x;
     *  q[1] = y;
     *  q[2] = z;
     *  q[3] = w;
     *
     */
    union {
        struct { T x, y, z, w; };
        TVec4<T> xyzw;
        TVec3<T> xyz;
        TVec2<T> xy;
    };

    enum { SIZE = 4 };
    inline constexpr static size_type size() { return SIZE; }

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
    TQuaternion(const TQuaternion&) = default;
    ~TQuaternion() = default;
    TQuaternion& operator = (const TQuaternion&) = default;

    // constructors

    // leaves object uninitialized. use with caution.
    explicit
    constexpr TQuaternion(no_init) : xyzw(TVec4<T>::NO_INIT) {}

    // default constructor. sets all values to zero.
    constexpr TQuaternion() : x(0), y(0), z(0), w(0) { }

    // handles implicit conversion to a tvec4. must not be explicit.
    template<typename A>
    constexpr TQuaternion(A w) : x(0), y(0), z(0), w(w) {
        static_assert(std::is_arithmetic<A>::value, "requires arithmetic type");
    }

    // initialize from 4 values to w + xi + yj + zk
    template<typename A, typename B, typename C, typename D>
    constexpr TQuaternion(A w, B x, C y, D z) : x(x), y(y), z(z), w(w) { }

    // initialize from a vec3 + a value to : v.xi + v.yj + v.zk + w
    template<typename A, typename B>
    constexpr TQuaternion(const TVec3<A>& v, B w) : x(v.x), y(v.y), z(v.z), w(w) { }

    // initialize from a double4
    template<typename A>
    constexpr explicit TQuaternion(const TVec4<A>& v) : x(v.x), y(v.y), z(v.z), w(v.w) { }

    // initialize from a quaternion of a different type
    template<typename A>
    constexpr explicit TQuaternion(const TQuaternion<A>& v) : x(v.x), y(v.y), z(v.z), w(v.w) { }

    // conjugate operator
    constexpr TQuaternion operator~() const {
        return conj(*this);
    }

    // constructs a quaternion from an axis and angle
    template <typename A, typename B>
    constexpr static TQuaternion PURE fromAxisAngle(const TVec3<A>& axis, B angle) {
        return TQuaternion(std::sin(angle*0.5) * normalize(axis), std::cos(angle*0.5));
    }
};

}  // namespace details

// ----------------------------------------------------------------------------------------

typedef details::TQuaternion<double> quatd;
typedef details::TQuaternion<float> quat;
typedef details::TQuaternion<float> quatf;
typedef details::TQuaternion<half> quath;

constexpr inline quat operator"" _i(long double v) {
    return quat(0, static_cast<float>(v), 0, 0);
}
constexpr inline quat operator"" _j(long double v) {
    return quat(0, 0, static_cast<float>(v), 0);
}
constexpr inline quat operator"" _k(long double v) {
    return quat(0, 0, 0, static_cast<float>(v));
}

constexpr inline quat operator"" _i(unsigned long long v) {  // NOLINT
    return quat(0, static_cast<float>(v), 0, 0);
}
constexpr inline quat operator"" _j(unsigned long long v) {  // NOLINT
    return quat(0, 0, static_cast<float>(v), 0);
}
constexpr inline quat operator"" _k(unsigned long long v) {  // NOLINT
    return quat(0, 0, 0, static_cast<float>(v));
}

constexpr inline quatd operator"" _id(long double v) {
    return quatd(0, static_cast<double>(v), 0, 0);
}
constexpr inline quatd operator"" _jd(long double v) {
    return quatd(0, 0, static_cast<double>(v), 0);
}
constexpr inline quatd operator"" _kd(long double v) {
    return quatd(0, 0, 0, static_cast<double>(v));
}

constexpr inline quatd operator"" _id(unsigned long long v) {  // NOLINT
    return quatd(0, static_cast<double>(v), 0, 0);
}
constexpr inline quatd operator"" _jd(unsigned long long v) {  // NOLINT
    return quatd(0, 0, static_cast<double>(v), 0);
}
constexpr inline quatd operator"" _kd(unsigned long long v) {  // NOLINT
    return quatd(0, 0, 0, static_cast<double>(v));
}

// ----------------------------------------------------------------------------------------
}  // namespace android

#pragma clang diagnostic pop

#undef PURE
