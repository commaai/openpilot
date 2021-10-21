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

#ifndef TMAT_IMPLEMENTATION
#error "Don't include TMatHelpers.h directly. use ui/mat*.h instead"
#else
#undef TMAT_IMPLEMENTATION
#endif


#ifndef UI_TMAT_HELPERS_H
#define UI_TMAT_HELPERS_H

#include <stdint.h>
#include <sys/types.h>
#include <math.h>
#include <utils/Debug.h>
#include <utils/String8.h>

#define PURE __attribute__((pure))

namespace android {
// -------------------------------------------------------------------------------------

/*
 * No user serviceable parts here.
 *
 * Don't use this file directly, instead include ui/mat*.h
 */


/*
 * Matrix utilities
 */

namespace matrix {

inline int     PURE transpose(int v)    { return v; }
inline float   PURE transpose(float v)  { return v; }
inline double  PURE transpose(double v) { return v; }

inline int     PURE trace(int v)    { return v; }
inline float   PURE trace(float v)  { return v; }
inline double  PURE trace(double v) { return v; }

template<typename MATRIX>
MATRIX PURE inverse(const MATRIX& src) {

    COMPILE_TIME_ASSERT_FUNCTION_SCOPE( MATRIX::COL_SIZE == MATRIX::ROW_SIZE );

    typename MATRIX::value_type t;
    const size_t N = MATRIX::col_size();
    size_t swap;
    MATRIX tmp(src);
    MATRIX inverse(1);

    for (size_t i=0 ; i<N ; i++) {
        // look for largest element in column
        swap = i;
        for (size_t j=i+1 ; j<N ; j++) {
            if (fabs(tmp[j][i]) > fabs(tmp[i][i])) {
                swap = j;
            }
        }

        if (swap != i) {
            /* swap rows. */
            for (size_t k=0 ; k<N ; k++) {
                t = tmp[i][k];
                tmp[i][k] = tmp[swap][k];
                tmp[swap][k] = t;

                t = inverse[i][k];
                inverse[i][k] = inverse[swap][k];
                inverse[swap][k] = t;
            }
        }

        t = 1 / tmp[i][i];
        for (size_t k=0 ; k<N ; k++) {
            tmp[i][k] *= t;
            inverse[i][k] *= t;
        }
        for (size_t j=0 ; j<N ; j++) {
            if (j != i) {
                t = tmp[j][i];
                for (size_t k=0 ; k<N ; k++) {
                    tmp[j][k] -= tmp[i][k] * t;
                    inverse[j][k] -= inverse[i][k] * t;
                }
            }
        }
    }
    return inverse;
}

template<typename MATRIX_R, typename MATRIX_A, typename MATRIX_B>
MATRIX_R PURE multiply(const MATRIX_A& lhs, const MATRIX_B& rhs) {
    // pre-requisite:
    //  lhs : D columns, R rows
    //  rhs : C columns, D rows
    //  res : C columns, R rows

    COMPILE_TIME_ASSERT_FUNCTION_SCOPE( MATRIX_A::ROW_SIZE == MATRIX_B::COL_SIZE );
    COMPILE_TIME_ASSERT_FUNCTION_SCOPE( MATRIX_R::ROW_SIZE == MATRIX_B::ROW_SIZE );
    COMPILE_TIME_ASSERT_FUNCTION_SCOPE( MATRIX_R::COL_SIZE == MATRIX_A::COL_SIZE );

    MATRIX_R res(MATRIX_R::NO_INIT);
    for (size_t r=0 ; r<MATRIX_R::row_size() ; r++) {
        res[r] = lhs * rhs[r];
    }
    return res;
}

// transpose. this handles matrices of matrices
template <typename MATRIX>
MATRIX PURE transpose(const MATRIX& m) {
    // for now we only handle square matrix transpose
    COMPILE_TIME_ASSERT_FUNCTION_SCOPE( MATRIX::ROW_SIZE == MATRIX::COL_SIZE );
    MATRIX result(MATRIX::NO_INIT);
    for (size_t r=0 ; r<MATRIX::row_size() ; r++)
        for (size_t c=0 ; c<MATRIX::col_size() ; c++)
            result[c][r] = transpose(m[r][c]);
    return result;
}

// trace. this handles matrices of matrices
template <typename MATRIX>
typename MATRIX::value_type PURE trace(const MATRIX& m) {
    COMPILE_TIME_ASSERT_FUNCTION_SCOPE( MATRIX::ROW_SIZE == MATRIX::COL_SIZE );
    typename MATRIX::value_type result(0);
    for (size_t r=0 ; r<MATRIX::row_size() ; r++)
        result += trace(m[r][r]);
    return result;
}

// trace. this handles matrices of matrices
template <typename MATRIX>
typename MATRIX::col_type PURE diag(const MATRIX& m) {
    COMPILE_TIME_ASSERT_FUNCTION_SCOPE( MATRIX::ROW_SIZE == MATRIX::COL_SIZE );
    typename MATRIX::col_type result(MATRIX::col_type::NO_INIT);
    for (size_t r=0 ; r<MATRIX::row_size() ; r++)
        result[r] = m[r][r];
    return result;
}

template <typename MATRIX>
String8 asString(const MATRIX& m) {
    String8 s;
    for (size_t c=0 ; c<MATRIX::col_size() ; c++) {
        s.append("|  ");
        for (size_t r=0 ; r<MATRIX::row_size() ; r++) {
            s.appendFormat("%7.2f  ", m[r][c]);
        }
        s.append("|\n");
    }
    return s;
}

}; // namespace matrix

// -------------------------------------------------------------------------------------

/*
 * TMatProductOperators implements basic arithmetic and basic compound assignments
 * operators on a vector of type BASE<T>.
 *
 * BASE only needs to implement operator[] and size().
 * By simply inheriting from TMatProductOperators<BASE, T> BASE will automatically
 * get all the functionality here.
 */

template <template<typename T> class BASE, typename T>
class TMatProductOperators {
public:
    // multiply by a scalar
    BASE<T>& operator *= (T v) {
        BASE<T>& lhs(static_cast< BASE<T>& >(*this));
        for (size_t r=0 ; r<lhs.row_size() ; r++) {
            lhs[r] *= v;
        }
        return lhs;
    }

    // divide by a scalar
    BASE<T>& operator /= (T v) {
        BASE<T>& lhs(static_cast< BASE<T>& >(*this));
        for (size_t r=0 ; r<lhs.row_size() ; r++) {
            lhs[r] /= v;
        }
        return lhs;
    }

    // matrix * matrix, result is a matrix of the same type than the lhs matrix
    template<typename U>
    friend BASE<T> PURE operator *(const BASE<T>& lhs, const BASE<U>& rhs) {
        return matrix::multiply<BASE<T> >(lhs, rhs);
    }
};


/*
 * TMatSquareFunctions implements functions on a matrix of type BASE<T>.
 *
 * BASE only needs to implement:
 *  - operator[]
 *  - col_type
 *  - row_type
 *  - COL_SIZE
 *  - ROW_SIZE
 *
 * By simply inheriting from TMatSquareFunctions<BASE, T> BASE will automatically
 * get all the functionality here.
 */

template<template<typename U> class BASE, typename T>
class TMatSquareFunctions {
public:
    /*
     * NOTE: the functions below ARE NOT member methods. They are friend functions
     * with they definition inlined with their declaration. This makes these
     * template functions available to the compiler when (and only when) this class
     * is instantiated, at which point they're only templated on the 2nd parameter
     * (the first one, BASE<T> being known).
     */
    friend BASE<T> PURE inverse(const BASE<T>& m)   { return matrix::inverse(m); }
    friend BASE<T> PURE transpose(const BASE<T>& m) { return matrix::transpose(m); }
    friend T       PURE trace(const BASE<T>& m)     { return matrix::trace(m); }
};

template <template<typename T> class BASE, typename T>
class TMatDebug {
public:
    String8 asString() const {
        return matrix::asString( static_cast< const BASE<T>& >(*this) );
    }
};

// -------------------------------------------------------------------------------------
}; // namespace android

#undef PURE

#endif /* UI_TMAT_HELPERS_H */
