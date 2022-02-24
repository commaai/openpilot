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

#ifndef UI_VEC2_H
#define UI_VEC2_H

#include <stdint.h>
#include <sys/types.h>

#define TVEC_IMPLEMENTATION
#include <ui/TVecHelpers.h>

namespace android {
// -------------------------------------------------------------------------------------

template <typename T>
class tvec2 :   public TVecProductOperators<tvec2, T>,
                public TVecAddOperators<tvec2, T>,
                public TVecUnaryOperators<tvec2, T>,
                public TVecComparisonOperators<tvec2, T>,
                public TVecFunctions<tvec2, T>
{
public:
    enum no_init { NO_INIT };
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef size_t size_type;

    union {
        struct { T x, y; };
        struct { T s, t; };
        struct { T r, g; };
    };

    enum { SIZE = 2 };
    inline static size_type size() { return SIZE; }

    // array access
    inline T const& operator [] (size_t i) const { return (&x)[i]; }
    inline T&       operator [] (size_t i)       { return (&x)[i]; }

    // -----------------------------------------------------------------------
    // we don't provide copy-ctor and operator= on purpose
    // because we want the compiler generated versions

    // constructors

    // leaves object uninitialized. use with caution.
    explicit tvec2(no_init) { }

    // default constructor
    tvec2() : x(0), y(0) { }

    // handles implicit conversion to a tvec4. must not be explicit.
    template<typename A>
    tvec2(A v) : x(v), y(v) { }

    template<typename A, typename B>
    tvec2(A x, B y) : x(x), y(y) { }

    template<typename A>
    explicit tvec2(const tvec2<A>& v) : x(v.x), y(v.y) { }

    template<typename A>
    tvec2(const Impersonator< tvec2<A> >& v)
        : x(((const tvec2<A>&)v).x),
          y(((const tvec2<A>&)v).y) { }
};

// ----------------------------------------------------------------------------------------

typedef tvec2<float> vec2;

// ----------------------------------------------------------------------------------------
}; // namespace android

#endif /* UI_VEC4_H */
