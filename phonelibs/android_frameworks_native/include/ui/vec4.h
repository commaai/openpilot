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

#ifndef UI_VEC4_H
#define UI_VEC4_H

#include <stdint.h>
#include <sys/types.h>

#include <ui/vec3.h>

namespace android {
// -------------------------------------------------------------------------------------

template <typename T>
class tvec4 :   public TVecProductOperators<tvec4, T>,
                public TVecAddOperators<tvec4, T>,
                public TVecUnaryOperators<tvec4, T>,
                public TVecComparisonOperators<tvec4, T>,
                public TVecFunctions<tvec4, T>
{
public:
    enum no_init { NO_INIT };
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef size_t size_type;

    union {
        struct { T x, y, z, w; };
        struct { T s, t, p, q; };
        struct { T r, g, b, a; };
        Impersonator< tvec2<T> > xy;
        Impersonator< tvec2<T> > st;
        Impersonator< tvec2<T> > rg;
        Impersonator< tvec3<T> > xyz;
        Impersonator< tvec3<T> > stp;
        Impersonator< tvec3<T> > rgb;
    };

    enum { SIZE = 4 };
    inline static size_type size() { return SIZE; }

    // array access
    inline T const& operator [] (size_t i) const { return (&x)[i]; }
    inline T&       operator [] (size_t i)       { return (&x)[i]; }

    // -----------------------------------------------------------------------
    // we don't provide copy-ctor and operator= on purpose
    // because we want the compiler generated versions

    // constructors

    // leaves object uninitialized. use with caution.
    explicit tvec4(no_init) { }

    // default constructor
    tvec4() : x(0), y(0), z(0), w(0) { }

    // handles implicit conversion to a tvec4. must not be explicit.
    template<typename A>
    tvec4(A v) : x(v), y(v), z(v), w(v) { }

    template<typename A, typename B, typename C, typename D>
    tvec4(A x, B y, C z, D w) : x(x), y(y), z(z), w(w) { }

    template<typename A, typename B, typename C>
    tvec4(const tvec2<A>& v, B z, C w) : x(v.x), y(v.y), z(z), w(w) { }

    template<typename A, typename B>
    tvec4(const tvec3<A>& v, B w) : x(v.x), y(v.y), z(v.z), w(w) { }

    template<typename A>
    explicit tvec4(const tvec4<A>& v) : x(v.x), y(v.y), z(v.z), w(v.w) { }

    template<typename A>
    tvec4(const Impersonator< tvec4<A> >& v)
        : x(((const tvec4<A>&)v).x),
          y(((const tvec4<A>&)v).y),
          z(((const tvec4<A>&)v).z),
          w(((const tvec4<A>&)v).w) { }

    template<typename A, typename B>
    tvec4(const Impersonator< tvec3<A> >& v, B w)
        : x(((const tvec3<A>&)v).x),
          y(((const tvec3<A>&)v).y),
          z(((const tvec3<A>&)v).z),
          w(w) { }

    template<typename A, typename B, typename C>
    tvec4(const Impersonator< tvec2<A> >& v, B z, C w)
        : x(((const tvec2<A>&)v).x),
          y(((const tvec2<A>&)v).y),
          z(z),
          w(w) { }
};

// ----------------------------------------------------------------------------------------

typedef tvec4<float> vec4;

// ----------------------------------------------------------------------------------------
}; // namespace android

#endif /* UI_VEC4_H */
