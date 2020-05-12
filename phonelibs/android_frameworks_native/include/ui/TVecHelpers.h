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

#ifndef TVEC_IMPLEMENTATION
#error "Don't include TVecHelpers.h directly. use ui/vec*.h instead"
#else
#undef TVEC_IMPLEMENTATION
#endif


#ifndef UI_TVEC_HELPERS_H
#define UI_TVEC_HELPERS_H

#include <stdint.h>
#include <sys/types.h>

#define PURE __attribute__((pure))

namespace android {
// -------------------------------------------------------------------------------------

/*
 * No user serviceable parts here.
 *
 * Don't use this file directly, instead include ui/vec{2|3|4}.h
 */

/*
 * This class casts itself into anything and assign itself from anything!
 * Use with caution!
 */
template <typename TYPE>
struct Impersonator {
    Impersonator& operator = (const TYPE& rhs) {
        reinterpret_cast<TYPE&>(*this) = rhs;
        return *this;
    }
    operator TYPE& () {
        return reinterpret_cast<TYPE&>(*this);
    }
    operator TYPE const& () const {
        return reinterpret_cast<TYPE const&>(*this);
    }
};

/*
 * TVec{Add|Product}Operators implements basic arithmetic and basic compound assignments
 * operators on a vector of type BASE<T>.
 *
 * BASE only needs to implement operator[] and size().
 * By simply inheriting from TVec{Add|Product}Operators<BASE, T> BASE will automatically
 * get all the functionality here.
 */

template <template<typename T> class BASE, typename T>
class TVecAddOperators {
public:
    /* compound assignment from a another vector of the same size but different
     * element type.
     */
    template <typename OTHER>
    BASE<T>& operator += (const BASE<OTHER>& v) {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            rhs[i] += v[i];
        }
        return rhs;
    }
    template <typename OTHER>
    BASE<T>& operator -= (const BASE<OTHER>& v) {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            rhs[i] -= v[i];
        }
        return rhs;
    }

    /* compound assignment from a another vector of the same type.
     * These operators can be used for implicit conversion and  handle operations
     * like "vector *= scalar" by letting the compiler implicitly convert a scalar
     * to a vector (assuming the BASE<T> allows it).
     */
    BASE<T>& operator += (const BASE<T>& v) {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            rhs[i] += v[i];
        }
        return rhs;
    }
    BASE<T>& operator -= (const BASE<T>& v) {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            rhs[i] -= v[i];
        }
        return rhs;
    }

    /*
     * NOTE: the functions below ARE NOT member methods. They are friend functions
     * with they definition inlined with their declaration. This makes these
     * template functions available to the compiler when (and only when) this class
     * is instantiated, at which point they're only templated on the 2nd parameter
     * (the first one, BASE<T> being known).
     */

    /* The operators below handle operation between vectors of the same side
     * but of a different element type.
     */
    template<typename RT>
    friend inline
    BASE<T> PURE operator +(const BASE<T>& lv, const BASE<RT>& rv) {
        return BASE<T>(lv) += rv;
    }
    template<typename RT>
    friend inline
    BASE<T> PURE operator -(const BASE<T>& lv, const BASE<RT>& rv) {
        return BASE<T>(lv) -= rv;
    }

    /* The operators below (which are not templates once this class is instanced,
     * i.e.: BASE<T> is known) can be used for implicit conversion on both sides.
     * These handle operations like "vector * scalar" and "scalar * vector" by
     * letting the compiler implicitly convert a scalar to a vector (assuming
     * the BASE<T> allows it).
     */
    friend inline
    BASE<T> PURE operator +(const BASE<T>& lv, const BASE<T>& rv) {
        return BASE<T>(lv) += rv;
    }
    friend inline
    BASE<T> PURE operator -(const BASE<T>& lv, const BASE<T>& rv) {
        return BASE<T>(lv) -= rv;
    }
};

template <template<typename T> class BASE, typename T>
class TVecProductOperators {
public:
    /* compound assignment from a another vector of the same size but different
     * element type.
     */
    template <typename OTHER>
    BASE<T>& operator *= (const BASE<OTHER>& v) {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            rhs[i] *= v[i];
        }
        return rhs;
    }
    template <typename OTHER>
    BASE<T>& operator /= (const BASE<OTHER>& v) {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            rhs[i] /= v[i];
        }
        return rhs;
    }

    /* compound assignment from a another vector of the same type.
     * These operators can be used for implicit conversion and  handle operations
     * like "vector *= scalar" by letting the compiler implicitly convert a scalar
     * to a vector (assuming the BASE<T> allows it).
     */
    BASE<T>& operator *= (const BASE<T>& v) {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            rhs[i] *= v[i];
        }
        return rhs;
    }
    BASE<T>& operator /= (const BASE<T>& v) {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            rhs[i] /= v[i];
        }
        return rhs;
    }

    /*
     * NOTE: the functions below ARE NOT member methods. They are friend functions
     * with they definition inlined with their declaration. This makes these
     * template functions available to the compiler when (and only when) this class
     * is instantiated, at which point they're only templated on the 2nd parameter
     * (the first one, BASE<T> being known).
     */

    /* The operators below handle operation between vectors of the same side
     * but of a different element type.
     */
    template<typename RT>
    friend inline
    BASE<T> PURE operator *(const BASE<T>& lv, const BASE<RT>& rv) {
        return BASE<T>(lv) *= rv;
    }
    template<typename RT>
    friend inline
    BASE<T> PURE operator /(const BASE<T>& lv, const BASE<RT>& rv) {
        return BASE<T>(lv) /= rv;
    }

    /* The operators below (which are not templates once this class is instanced,
     * i.e.: BASE<T> is known) can be used for implicit conversion on both sides.
     * These handle operations like "vector * scalar" and "scalar * vector" by
     * letting the compiler implicitly convert a scalar to a vector (assuming
     * the BASE<T> allows it).
     */
    friend inline
    BASE<T> PURE operator *(const BASE<T>& lv, const BASE<T>& rv) {
        return BASE<T>(lv) *= rv;
    }
    friend inline
    BASE<T> PURE operator /(const BASE<T>& lv, const BASE<T>& rv) {
        return BASE<T>(lv) /= rv;
    }
};

/*
 * TVecUnaryOperators implements unary operators on a vector of type BASE<T>.
 *
 * BASE only needs to implement operator[] and size().
 * By simply inheriting from TVecUnaryOperators<BASE, T> BASE will automatically
 * get all the functionality here.
 *
 * These operators are implemented as friend functions of TVecUnaryOperators<BASE, T>
 */
template <template<typename T> class BASE, typename T>
class TVecUnaryOperators {
public:
    BASE<T>& operator ++ () {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            ++rhs[i];
        }
        return rhs;
    }
    BASE<T>& operator -- () {
        BASE<T>& rhs = static_cast<BASE<T>&>(*this);
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            --rhs[i];
        }
        return rhs;
    }
    BASE<T> operator - () const {
        BASE<T> r(BASE<T>::NO_INIT);
        BASE<T> const& rv(static_cast<BASE<T> const&>(*this));
        for (size_t i=0 ; i<BASE<T>::size() ; i++) {
            r[i] = -rv[i];
        }
        return r;
    }
};


/*
 * TVecComparisonOperators implements relational/comparison operators
 * on a vector of type BASE<T>.
 *
 * BASE only needs to implement operator[] and size().
 * By simply inheriting from TVecComparisonOperators<BASE, T> BASE will automatically
 * get all the functionality here.
 */
template <template<typename T> class BASE, typename T>
class TVecComparisonOperators {
public:
    /*
     * NOTE: the functions below ARE NOT member methods. They are friend functions
     * with they definition inlined with their declaration. This makes these
     * template functions available to the compiler when (and only when) this class
     * is instantiated, at which point they're only templated on the 2nd parameter
     * (the first one, BASE<T> being known).
     */
    template<typename RT>
    friend inline
    bool PURE operator ==(const BASE<T>& lv, const BASE<RT>& rv) {
        for (size_t i = 0; i < BASE<T>::size(); i++)
            if (lv[i] != rv[i])
                return false;
        return true;
    }

    template<typename RT>
    friend inline
    bool PURE operator !=(const BASE<T>& lv, const BASE<RT>& rv) {
        return !operator ==(lv, rv);
    }

    template<typename RT>
    friend inline
    bool PURE operator >(const BASE<T>& lv, const BASE<RT>& rv) {
        for (size_t i = 0; i < BASE<T>::size(); i++)
            if (lv[i] <= rv[i])
                return false;
        return true;
    }

    template<typename RT>
    friend inline
    bool PURE operator <=(const BASE<T>& lv, const BASE<RT>& rv) {
        return !(lv > rv);
    }

    template<typename RT>
    friend inline
    bool PURE operator <(const BASE<T>& lv, const BASE<RT>& rv) {
        for (size_t i = 0; i < BASE<T>::size(); i++)
            if (lv[i] >= rv[i])
                return false;
        return true;
    }

    template<typename RT>
    friend inline
    bool PURE operator >=(const BASE<T>& lv, const BASE<RT>& rv) {
        return !(lv < rv);
    }
};


/*
 * TVecFunctions implements functions on a vector of type BASE<T>.
 *
 * BASE only needs to implement operator[] and size().
 * By simply inheriting from TVecFunctions<BASE, T> BASE will automatically
 * get all the functionality here.
 */
template <template<typename T> class BASE, typename T>
class TVecFunctions {
public:
    /*
     * NOTE: the functions below ARE NOT member methods. They are friend functions
     * with they definition inlined with their declaration. This makes these
     * template functions available to the compiler when (and only when) this class
     * is instantiated, at which point they're only templated on the 2nd parameter
     * (the first one, BASE<T> being known).
     */
    template<typename RT>
    friend inline
    T PURE dot(const BASE<T>& lv, const BASE<RT>& rv) {
        T r(0);
        for (size_t i = 0; i < BASE<T>::size(); i++)
            r += lv[i]*rv[i];
        return r;
    }

    friend inline
    T PURE length(const BASE<T>& lv) {
        return sqrt( dot(lv, lv) );
    }

    template<typename RT>
    friend inline
    T PURE distance(const BASE<T>& lv, const BASE<RT>& rv) {
        return length(rv - lv);
    }

    friend inline
    BASE<T> PURE normalize(const BASE<T>& lv) {
        return lv * (1 / length(lv));
    }
};

#undef PURE

// -------------------------------------------------------------------------------------
}; // namespace android


#endif /* UI_TVEC_HELPERS_H */
