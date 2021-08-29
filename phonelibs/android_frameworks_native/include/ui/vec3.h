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

#ifndef UI_VEC3_H
#define UI_VEC3_H

#include <stdint.h>
#include <sys/types.h>

#include <ui/vec2.h>

namespace android {
// -------------------------------------------------------------------------------------

template <typename T>
class tvec3 :   public TVecProductOperators<tvec3, T>,
                public TVecAddOperators<tvec3, T>,
                public TVecUnaryOperators<tvec3, T>,
                public TVecComparisonOperators<tvec3, T>,
                public TVecFunctions<tvec3, T>
{
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
        Impersonator< tvec2<T> > xy;
        Impersonator< tvec2<T> > st;
        Impersonator< tvec2<T> > rg;
    };

    enum { SIZE = 3 };
    inline static size_type size() { return SIZE; }

    // array access
    inline T const& operator [] (size_t i) const { return (&x)[i]; }
    inline T&       operator [] (size_t i)       { return (&x)[i]; }

    // -----------------------------------------------------------------------
    // we don't provide copy-ctor and operator= on purpose
    // because we want the compiler generated versions

    // constructors
    // leaves object uninitialized. use with caution.
    explicit tvec3(no_init) { }

    // default constructor
    tvec3() : x(0), y(0), z(0) { }

    // handles implicit conversion to a tvec4. must not be explicit.
    template<typename A>
    tvec3(A v) : x(v), y(v), z(v) { }

    template<typename A, typename B, typename C>
    tvec3(A x, B y, C z) : x(x), y(y), z(z) { }

    template<typename A, typename B>
    tvec3(const tvec2<A>& v, B z) : x(v.x), y(v.y), z(z) { }

    template<typename A>
    explicit tvec3(const tvec3<A>& v) : x(v.x), y(v.y), z(v.z) { }

    template<typename A>
    tvec3(const Impersonator< tvec3<A> >& v)
        : x(((const tvec3<A>&)v).x),
          y(((const tvec3<A>&)v).y),
          z(((const tvec3<A>&)v).z) { }

    template<typename A, typename B>
    tvec3(const Impersonator< tvec2<A> >& v, B z)
        : x(((const tvec2<A>&)v).x),
          y(((const tvec2<A>&)v).y),
          z(z) { }

    // cross product works only on vectors of size 3
    template <typename RT>
    friend inline
    tvec3 __attribute__((pure)) cross(const tvec3& u, const tvec3<RT>& v) {
        return tvec3(
                u.y*v.z - u.z*v.y,
                u.z*v.x - u.x*v.z,
                u.x*v.y - u.y*v.x);
    }
};


// ----------------------------------------------------------------------------------------

typedef tvec3<float> vec3;

// ----------------------------------------------------------------------------------------
}; // namespace android

#endif /* UI_VEC4_H */
