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

#include <math/TMatHelpers.h>
#include <math/vec2.h>
#include <stdint.h>
#include <sys/types.h>

#define PURE __attribute__((pure))

#if __cplusplus >= 201402L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

namespace android {
// -------------------------------------------------------------------------------------
namespace details {

/**
 * A 2x2 column-major matrix class.
 *
 * Conceptually a 2x2 matrix is a an array of 2 column vec2:
 *
 * mat2 m =
 *      \f$
 *      \left(
 *      \begin{array}{cc}
 *      m[0] & m[1] \\
 *      \end{array}
 *      \right)
 *      \f$
 *      =
 *      \f$
 *      \left(
 *      \begin{array}{cc}
 *      m[0][0] & m[1][0] \\
 *      m[0][1] & m[1][1] \\
 *      \end{array}
 *      \right)
 *      \f$
 *      =
 *      \f$
 *      \left(
 *      \begin{array}{cc}
 *      m(0,0) & m(0,1) \\
 *      m(1,0) & m(1,1) \\
 *      \end{array}
 *      \right)
 *      \f$
 *
 * m[n] is the \f$ n^{th} \f$ column of the matrix and is a vec2.
 *
 */
template <typename T>
class TMat22 :  public TVecUnaryOperators<TMat22, T>,
                public TVecComparisonOperators<TMat22, T>,
                public TVecAddOperators<TMat22, T>,
                public TMatProductOperators<TMat22, T>,
                public TMatSquareFunctions<TMat22, T>,
                public TMatHelpers<TMat22, T>,
                public TMatDebug<TMat22, T> {
public:
    enum no_init { NO_INIT };
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef size_t size_type;
    typedef TVec2<T> col_type;
    typedef TVec2<T> row_type;

    static constexpr size_t COL_SIZE = col_type::SIZE;  // size of a column (i.e.: number of rows)
    static constexpr size_t ROW_SIZE = row_type::SIZE;  // size of a row (i.e.: number of columns)
    static constexpr size_t NUM_ROWS = COL_SIZE;
    static constexpr size_t NUM_COLS = ROW_SIZE;

private:
    /*
     *  <--  N columns  -->
     *
     *  a[0][0] a[1][0] a[2][0] ... a[N][0]    ^
     *  a[0][1] a[1][1] a[2][1] ... a[N][1]    |
     *  a[0][2] a[1][2] a[2][2] ... a[N][2]  M rows
     *  ...                                    |
     *  a[0][M] a[1][M] a[2][M] ... a[N][M]    v
     *
     *  COL_SIZE = M
     *  ROW_SIZE = N
     *  m[0] = [ a[0][0] a[0][1] a[0][2] ... a[0][M] ]
     */

    col_type m_value[NUM_COLS];

public:
    // array access
    inline constexpr col_type const& operator[](size_t column) const {
#if __cplusplus >= 201402L
        // only possible in C++0x14 with constexpr
        assert(column < NUM_COLS);
#endif
        return m_value[column];
    }

    inline col_type& operator[](size_t column) {
        assert(column < NUM_COLS);
        return m_value[column];
    }

    // -----------------------------------------------------------------------
    // we want the compiler generated versions for these...
    TMat22(const TMat22&) = default;
    ~TMat22() = default;
    TMat22& operator = (const TMat22&) = default;

    /**
     *  constructors
     */

    /**
     * leaves object uninitialized. use with caution.
     */
    explicit constexpr TMat22(no_init)
            : m_value{ col_type(col_type::NO_INIT),
                       col_type(col_type::NO_INIT) } {}


    /**
     * initialize to identity.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cc}
     *      1 & 0 \\
     *      0 & 1 \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    CONSTEXPR TMat22();

    /**
     * initialize to Identity*scalar.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cc}
     *      v & 0 \\
     *      0 & v \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template<typename U>
    explicit CONSTEXPR TMat22(U v);

    /**
     * sets the diagonal to a vector.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cc}
     *      v[0] & 0 \\
     *      0 & v[1] \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <typename U>
    explicit CONSTEXPR TMat22(const TVec2<U>& v);

    /**
     * construct from another matrix of the same size
     */
    template <typename U>
    explicit CONSTEXPR TMat22(const TMat22<U>& rhs);

    /**
     * construct from 2 column vectors.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cc}
     *      v0 & v1 \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <typename A, typename B>
    CONSTEXPR TMat22(const TVec2<A>& v0, const TVec2<B>& v1);

    /** construct from 4 elements in column-major form.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cc}
     *      m[0][0] & m[1][0] \\
     *      m[0][1] & m[1][1] \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <
        typename A, typename B,
        typename C, typename D>
    CONSTEXPR TMat22(A m00, B m01, C m10, D m11);

    /**
     * construct from a C array in column major form.
     */
    template <typename U>
    explicit CONSTEXPR TMat22(U const* rawArray);

    /**
     * Rotate by radians in the 2D plane
     */
    static CONSTEXPR TMat22<T> rotate(T radian) {
        TMat22<T> r(TMat22<T>::NO_INIT);
        T c = std::cos(radian);
        T s = std::sin(radian);
        r[0][0] = c;   r[1][1] = c;
        r[0][1] = s;   r[1][0] = -s;
        return r;
    }
};

// ----------------------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------------------

// Since the matrix code could become pretty big quickly, we don't inline most
// operations.

template <typename T>
CONSTEXPR TMat22<T>::TMat22() {
    m_value[0] = col_type(1, 0);
    m_value[1] = col_type(0, 1);
}

template <typename T>
template <typename U>
CONSTEXPR TMat22<T>::TMat22(U v) {
    m_value[0] = col_type(v, 0);
    m_value[1] = col_type(0, v);
}

template<typename T>
template<typename U>
CONSTEXPR TMat22<T>::TMat22(const TVec2<U>& v) {
    m_value[0] = col_type(v.x, 0);
    m_value[1] = col_type(0, v.y);
}

// construct from 4 scalars. Note that the arrangement
// of values in the constructor is the transpose of the matrix
// notation.
template<typename T>
template <
    typename A, typename B,
    typename C, typename D>
CONSTEXPR TMat22<T>::TMat22( A m00, B m01, C m10, D m11) {
    m_value[0] = col_type(m00, m01);
    m_value[1] = col_type(m10, m11);
}

template <typename T>
template <typename U>
CONSTEXPR TMat22<T>::TMat22(const TMat22<U>& rhs) {
    for (size_t col = 0; col < NUM_COLS; ++col) {
        m_value[col] = col_type(rhs[col]);
    }
}

// Construct from 2 column vectors.
template <typename T>
template <typename A, typename B>
CONSTEXPR TMat22<T>::TMat22(const TVec2<A>& v0, const TVec2<B>& v1) {
    m_value[0] = v0;
    m_value[1] = v1;
}

// Construct from raw array, in column-major form.
template <typename T>
template <typename U>
CONSTEXPR TMat22<T>::TMat22(U const* rawArray) {
    for (size_t col = 0; col < NUM_COLS; ++col) {
        for (size_t row = 0; row < NUM_ROWS; ++row) {
            m_value[col][row] = *rawArray++;
        }
    }
}

// ----------------------------------------------------------------------------------------
// Arithmetic operators outside of class
// ----------------------------------------------------------------------------------------

/* We use non-friend functions here to prevent the compiler from using
 * implicit conversions, for instance of a scalar to a vector. The result would
 * not be what the caller expects.
 *
 * Also note that the order of the arguments in the inner loop is important since
 * it determines the output type (only relevant when T != U).
 */

// matrix * column-vector, result is a vector of the same type than the input vector
template <typename T, typename U>
CONSTEXPR typename TMat22<U>::col_type PURE operator *(const TMat22<T>& lhs, const TVec2<U>& rhs) {
    // Result is initialized to zero.
    typename TMat22<U>::col_type result;
    for (size_t col = 0; col < TMat22<T>::NUM_COLS; ++col) {
        result += lhs[col] * rhs[col];
    }
    return result;
}

// row-vector * matrix, result is a vector of the same type than the input vector
template <typename T, typename U>
CONSTEXPR typename TMat22<U>::row_type PURE operator *(const TVec2<U>& lhs, const TMat22<T>& rhs) {
    typename TMat22<U>::row_type result(TMat22<U>::row_type::NO_INIT);
    for (size_t col = 0; col < TMat22<T>::NUM_COLS; ++col) {
        result[col] = dot(lhs, rhs[col]);
    }
    return result;
}

// matrix * scalar, result is a matrix of the same type than the input matrix
template<typename T, typename U>
constexpr typename std::enable_if<std::is_arithmetic<U>::value, TMat22<T>>::type PURE
operator*(TMat22<T> lhs, U rhs) {
    return lhs *= rhs;
}

// scalar * matrix, result is a matrix of the same type than the input matrix
template<typename T, typename U>
constexpr typename std::enable_if<std::is_arithmetic<U>::value, TMat22<T>>::type PURE
operator*(U lhs, const TMat22<T>& rhs) {
    return rhs * lhs;
}

// ----------------------------------------------------------------------------------------

/* FIXME: this should go into TMatSquareFunctions<> but for some reason
 * BASE<T>::col_type is not accessible from there (???)
 */
template<typename T>
CONSTEXPR typename TMat22<T>::col_type PURE diag(const TMat22<T>& m) {
    return matrix::diag(m);
}

}  // namespace details

// ----------------------------------------------------------------------------------------

typedef details::TMat22<double> mat2d;
typedef details::TMat22<float> mat2;
typedef details::TMat22<float> mat2f;

// ----------------------------------------------------------------------------------------
}  // namespace android

#undef PURE
#undef CONSTEXPR
