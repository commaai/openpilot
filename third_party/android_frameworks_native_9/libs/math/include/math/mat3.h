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

#include <math/quat.h>
#include <math/TMatHelpers.h>
#include <math/vec3.h>
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

template<typename T>
class TQuaternion;

/**
 * A 3x3 column-major matrix class.
 *
 * Conceptually a 3x3 matrix is a an array of 3 column vec3:
 *
 * mat3 m =
 *      \f$
 *      \left(
 *      \begin{array}{ccc}
 *      m[0] & m[1] & m[2] \\
 *      \end{array}
 *      \right)
 *      \f$
 *      =
 *      \f$
 *      \left(
 *      \begin{array}{ccc}
 *      m[0][0] & m[1][0] & m[2][0] \\
 *      m[0][1] & m[1][1] & m[2][1] \\
 *      m[0][2] & m[1][2] & m[2][2] \\
 *      \end{array}
 *      \right)
 *      \f$
 *      =
 *      \f$
 *      \left(
 *      \begin{array}{ccc}
 *      m(0,0) & m(0,1) & m(0,2) \\
 *      m(1,0) & m(1,1) & m(1,2) \\
 *      m(2,0) & m(2,1) & m(2,2) \\
 *      \end{array}
 *      \right)
 *      \f$
 *
 * m[n] is the \f$ n^{th} \f$ column of the matrix and is a vec3.
 *
 */
template <typename T>
class TMat33 :  public TVecUnaryOperators<TMat33, T>,
                public TVecComparisonOperators<TMat33, T>,
                public TVecAddOperators<TMat33, T>,
                public TMatProductOperators<TMat33, T>,
                public TMatSquareFunctions<TMat33, T>,
                public TMatTransform<TMat33, T>,
                public TMatHelpers<TMat33, T>,
                public TMatDebug<TMat33, T> {
public:
    enum no_init { NO_INIT };
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef size_t size_type;
    typedef TVec3<T> col_type;
    typedef TVec3<T> row_type;

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
    TMat33(const TMat33&) = default;
    ~TMat33() = default;
    TMat33& operator = (const TMat33&) = default;

    /**
     *  constructors
     */

    /**
     * leaves object uninitialized. use with caution.
     */
    explicit constexpr TMat33(no_init)
            : m_value{ col_type(col_type::NO_INIT),
                       col_type(col_type::NO_INIT),
                       col_type(col_type::NO_INIT) } {}


    /**
     * initialize to identity.
     *
     *      \f$
     *      \left(
     *      \begin{array}{ccc}
     *      1 & 0 & 0 \\
     *      0 & 1 & 0 \\
     *      0 & 0 & 1 \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    CONSTEXPR TMat33();

    /**
     * initialize to Identity*scalar.
     *
     *      \f$
     *      \left(
     *      \begin{array}{ccc}
     *      v & 0 & 0 \\
     *      0 & v & 0 \\
     *      0 & 0 & v \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template<typename U>
    explicit CONSTEXPR TMat33(U v);

    /**
     * sets the diagonal to a vector.
     *
     *      \f$
     *      \left(
     *      \begin{array}{ccc}
     *      v[0] & 0 & 0 \\
     *      0 & v[1] & 0 \\
     *      0 & 0 & v[2] \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <typename U>
    explicit CONSTEXPR TMat33(const TVec3<U>& v);

    /**
     * construct from another matrix of the same size
     */
    template <typename U>
    explicit CONSTEXPR TMat33(const TMat33<U>& rhs);

    /**
     * construct from 3 column vectors.
     *
     *      \f$
     *      \left(
     *      \begin{array}{ccc}
     *      v0 & v1 & v2 \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <typename A, typename B, typename C>
    CONSTEXPR TMat33(const TVec3<A>& v0, const TVec3<B>& v1, const TVec3<C>& v2);

    /** construct from 9 elements in column-major form.
     *
     *      \f$
     *      \left(
     *      \begin{array}{ccc}
     *      m[0][0] & m[1][0] & m[2][0] \\
     *      m[0][1] & m[1][1] & m[2][1] \\
     *      m[0][2] & m[1][2] & m[2][2] \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <
        typename A, typename B, typename C,
        typename D, typename E, typename F,
        typename G, typename H, typename I>
    CONSTEXPR TMat33(
           A m00, B m01, C m02,
           D m10, E m11, F m12,
           G m20, H m21, I m22);

    /**
     * construct from a quaternion
     */
    template <typename U>
    explicit CONSTEXPR TMat33(const TQuaternion<U>& q);

    /**
     * construct from a C array in column major form.
     */
    template <typename U>
    explicit CONSTEXPR TMat33(U const* rawArray);

    /**
     * orthogonalize only works on matrices of size 3x3
     */
    friend inline
    CONSTEXPR TMat33 orthogonalize(const TMat33& m) {
        TMat33 ret(TMat33::NO_INIT);
        ret[0] = normalize(m[0]);
        ret[2] = normalize(cross(ret[0], m[1]));
        ret[1] = normalize(cross(ret[2], ret[0]));
        return ret;
    }
};

// ----------------------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------------------

// Since the matrix code could become pretty big quickly, we don't inline most
// operations.

template <typename T>
CONSTEXPR TMat33<T>::TMat33() {
    m_value[0] = col_type(1, 0, 0);
    m_value[1] = col_type(0, 1, 0);
    m_value[2] = col_type(0, 0, 1);
}

template <typename T>
template <typename U>
CONSTEXPR TMat33<T>::TMat33(U v) {
    m_value[0] = col_type(v, 0, 0);
    m_value[1] = col_type(0, v, 0);
    m_value[2] = col_type(0, 0, v);
}

template<typename T>
template<typename U>
CONSTEXPR TMat33<T>::TMat33(const TVec3<U>& v) {
    m_value[0] = col_type(v.x, 0, 0);
    m_value[1] = col_type(0, v.y, 0);
    m_value[2] = col_type(0, 0, v.z);
}

// construct from 9 scalars. Note that the arrangement
// of values in the constructor is the transpose of the matrix
// notation.
template<typename T>
template <
    typename A, typename B, typename C,
    typename D, typename E, typename F,
    typename G, typename H, typename I>
CONSTEXPR TMat33<T>::TMat33(
        A m00, B m01, C m02,
        D m10, E m11, F m12,
        G m20, H m21, I m22) {
    m_value[0] = col_type(m00, m01, m02);
    m_value[1] = col_type(m10, m11, m12);
    m_value[2] = col_type(m20, m21, m22);
}

template <typename T>
template <typename U>
CONSTEXPR TMat33<T>::TMat33(const TMat33<U>& rhs) {
    for (size_t col = 0; col < NUM_COLS; ++col) {
        m_value[col] = col_type(rhs[col]);
    }
}

// Construct from 3 column vectors.
template <typename T>
template <typename A, typename B, typename C>
CONSTEXPR TMat33<T>::TMat33(const TVec3<A>& v0, const TVec3<B>& v1, const TVec3<C>& v2) {
    m_value[0] = v0;
    m_value[1] = v1;
    m_value[2] = v2;
}

// Construct from raw array, in column-major form.
template <typename T>
template <typename U>
CONSTEXPR TMat33<T>::TMat33(U const* rawArray) {
    for (size_t col = 0; col < NUM_COLS; ++col) {
        for (size_t row = 0; row < NUM_ROWS; ++row) {
            m_value[col][row] = *rawArray++;
        }
    }
}

template <typename T>
template <typename U>
CONSTEXPR TMat33<T>::TMat33(const TQuaternion<U>& q) {
    const U n = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    const U s = n > 0 ? 2/n : 0;
    const U x = s*q.x;
    const U y = s*q.y;
    const U z = s*q.z;
    const U xx = x*q.x;
    const U xy = x*q.y;
    const U xz = x*q.z;
    const U xw = x*q.w;
    const U yy = y*q.y;
    const U yz = y*q.z;
    const U yw = y*q.w;
    const U zz = z*q.z;
    const U zw = z*q.w;
    m_value[0] = col_type(1-yy-zz,    xy+zw,    xz-yw);  // NOLINT
    m_value[1] = col_type(  xy-zw,  1-xx-zz,    yz+xw);  // NOLINT
    m_value[2] = col_type(  xz+yw,    yz-xw,  1-xx-yy);  // NOLINT
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
CONSTEXPR typename TMat33<U>::col_type PURE operator *(const TMat33<T>& lhs, const TVec3<U>& rhs) {
    // Result is initialized to zero.
    typename TMat33<U>::col_type result;
    for (size_t col = 0; col < TMat33<T>::NUM_COLS; ++col) {
        result += lhs[col] * rhs[col];
    }
    return result;
}

// row-vector * matrix, result is a vector of the same type than the input vector
template <typename T, typename U>
CONSTEXPR typename TMat33<U>::row_type PURE operator *(const TVec3<U>& lhs, const TMat33<T>& rhs) {
    typename TMat33<U>::row_type result(TMat33<U>::row_type::NO_INIT);
    for (size_t col = 0; col < TMat33<T>::NUM_COLS; ++col) {
        result[col] = dot(lhs, rhs[col]);
    }
    return result;
}

// matrix * scalar, result is a matrix of the same type than the input matrix
template<typename T, typename U>
constexpr typename std::enable_if<std::is_arithmetic<U>::value, TMat33<T>>::type PURE
operator*(TMat33<T> lhs, U rhs) {
    return lhs *= rhs;
}

// scalar * matrix, result is a matrix of the same type than the input matrix
template<typename T, typename U>
constexpr typename std::enable_if<std::is_arithmetic<U>::value, TMat33<T>>::type PURE
operator*(U lhs, const TMat33<T>& rhs) {
    return rhs * lhs;
}

//------------------------------------------------------------------------------
template <typename T>
CONSTEXPR TMat33<T> orthogonalize(const TMat33<T>& m) {
    TMat33<T> ret(TMat33<T>::NO_INIT);
    ret[0] = normalize(m[0]);
    ret[2] = normalize(cross(ret[0], m[1]));
    ret[1] = normalize(cross(ret[2], ret[0]));
    return ret;
}

// ----------------------------------------------------------------------------------------

/* FIXME: this should go into TMatSquareFunctions<> but for some reason
 * BASE<T>::col_type is not accessible from there (???)
 */
template<typename T>
CONSTEXPR typename TMat33<T>::col_type PURE diag(const TMat33<T>& m) {
    return matrix::diag(m);
}

}  // namespace details

// ----------------------------------------------------------------------------------------

typedef details::TMat33<double> mat3d;
typedef details::TMat33<float> mat3;
typedef details::TMat33<float> mat3f;

// ----------------------------------------------------------------------------------------
}  // namespace android

#undef PURE
#undef CONSTEXPR
