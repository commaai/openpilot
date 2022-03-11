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

#include <math/mat3.h>
#include <math/quat.h>
#include <math/TMatHelpers.h>
#include <math/vec3.h>
#include <math/vec4.h>

#include <stdint.h>
#include <sys/types.h>
#include <limits>

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
 * A 4x4 column-major matrix class.
 *
 * Conceptually a 4x4 matrix is a an array of 4 column double4:
 *
 * mat4 m =
 *      \f$
 *      \left(
 *      \begin{array}{cccc}
 *      m[0] & m[1] & m[2] & m[3] \\
 *      \end{array}
 *      \right)
 *      \f$
 *      =
 *      \f$
 *      \left(
 *      \begin{array}{cccc}
 *      m[0][0] & m[1][0] & m[2][0] & m[3][0] \\
 *      m[0][1] & m[1][1] & m[2][1] & m[3][1] \\
 *      m[0][2] & m[1][2] & m[2][2] & m[3][2] \\
 *      m[0][3] & m[1][3] & m[2][3] & m[3][3] \\
 *      \end{array}
 *      \right)
 *      \f$
 *      =
 *      \f$
 *      \left(
 *      \begin{array}{cccc}
 *      m(0,0) & m(0,1) & m(0,2) & m(0,3) \\
 *      m(1,0) & m(1,1) & m(1,2) & m(1,3) \\
 *      m(2,0) & m(2,1) & m(2,2) & m(2,3) \\
 *      m(3,0) & m(3,1) & m(3,2) & m(3,3) \\
 *      \end{array}
 *      \right)
 *      \f$
 *
 * m[n] is the \f$ n^{th} \f$ column of the matrix and is a double4.
 *
 */
template <typename T>
class TMat44 :  public TVecUnaryOperators<TMat44, T>,
                public TVecComparisonOperators<TMat44, T>,
                public TVecAddOperators<TMat44, T>,
                public TMatProductOperators<TMat44, T>,
                public TMatSquareFunctions<TMat44, T>,
                public TMatTransform<TMat44, T>,
                public TMatHelpers<TMat44, T>,
                public TMatDebug<TMat44, T> {
public:
    enum no_init { NO_INIT };
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef size_t size_type;
    typedef TVec4<T> col_type;
    typedef TVec4<T> row_type;

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
    TMat44(const TMat44&) = default;
    ~TMat44() = default;
    TMat44& operator = (const TMat44&) = default;

    /*
     *  constructors
     */

    // leaves object uninitialized. use with caution.
    explicit constexpr TMat44(no_init)
            : m_value{ col_type(col_type::NO_INIT),
                       col_type(col_type::NO_INIT),
                       col_type(col_type::NO_INIT),
                       col_type(col_type::NO_INIT) } {}

    /** initialize to identity.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cccc}
     *      1 & 0 & 0 & 0 \\
     *      0 & 1 & 0 & 0 \\
     *      0 & 0 & 1 & 0 \\
     *      0 & 0 & 0 & 1 \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    CONSTEXPR TMat44();

    /** initialize to Identity*scalar.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cccc}
     *      v & 0 & 0 & 0 \\
     *      0 & v & 0 & 0 \\
     *      0 & 0 & v & 0 \\
     *      0 & 0 & 0 & v \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template<typename U>
    explicit CONSTEXPR TMat44(U v);

    /** sets the diagonal to a vector.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cccc}
     *      v[0] & 0 & 0 & 0 \\
     *      0 & v[1] & 0 & 0 \\
     *      0 & 0 & v[2] & 0 \\
     *      0 & 0 & 0 & v[3] \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <typename U>
    explicit CONSTEXPR TMat44(const TVec4<U>& v);

    // construct from another matrix of the same size
    template <typename U>
    explicit CONSTEXPR TMat44(const TMat44<U>& rhs);

    /** construct from 4 column vectors.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cccc}
     *      v0 & v1 & v2 & v3 \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <typename A, typename B, typename C, typename D>
    CONSTEXPR TMat44(const TVec4<A>& v0, const TVec4<B>& v1, const TVec4<C>& v2, const TVec4<D>& v3);

    /** construct from 16 elements in column-major form.
     *
     *      \f$
     *      \left(
     *      \begin{array}{cccc}
     *      m[0][0] & m[1][0] & m[2][0] & m[3][0] \\
     *      m[0][1] & m[1][1] & m[2][1] & m[3][1] \\
     *      m[0][2] & m[1][2] & m[2][2] & m[3][2] \\
     *      m[0][3] & m[1][3] & m[2][3] & m[3][3] \\
     *      \end{array}
     *      \right)
     *      \f$
     */
    template <
        typename A, typename B, typename C, typename D,
        typename E, typename F, typename G, typename H,
        typename I, typename J, typename K, typename L,
        typename M, typename N, typename O, typename P>
    CONSTEXPR TMat44(
            A m00, B m01, C m02, D m03,
            E m10, F m11, G m12, H m13,
            I m20, J m21, K m22, L m23,
            M m30, N m31, O m32, P m33);

    /**
     * construct from a quaternion
     */
    template <typename U>
    explicit CONSTEXPR TMat44(const TQuaternion<U>& q);

    /**
     * construct from a C array in column major form.
     */
    template <typename U>
    explicit CONSTEXPR TMat44(U const* rawArray);

    /**
     * construct from a 3x3 matrix
     */
    template <typename U>
    explicit CONSTEXPR TMat44(const TMat33<U>& matrix);

    /**
     * construct from a 3x3 matrix and 3d translation
     */
    template <typename U, typename V>
    CONSTEXPR TMat44(const TMat33<U>& matrix, const TVec3<V>& translation);

    /**
     * construct from a 3x3 matrix and 4d last column.
     */
    template <typename U, typename V>
    CONSTEXPR TMat44(const TMat33<U>& matrix, const TVec4<V>& column3);

    /*
     *  helpers
     */

    static CONSTEXPR TMat44 ortho(T left, T right, T bottom, T top, T near, T far);

    static CONSTEXPR TMat44 frustum(T left, T right, T bottom, T top, T near, T far);

    enum class Fov {
        HORIZONTAL,
        VERTICAL
    };
    static CONSTEXPR TMat44 perspective(T fov, T aspect, T near, T far, Fov direction = Fov::VERTICAL);

    template <typename A, typename B, typename C>
    static CONSTEXPR TMat44 lookAt(const TVec3<A>& eye, const TVec3<B>& center, const TVec3<C>& up);

    template <typename A>
    static CONSTEXPR TVec3<A> project(const TMat44& projectionMatrix, TVec3<A> vertice) {
        TVec4<A> r = projectionMatrix * TVec4<A>{ vertice, 1 };
        return r.xyz / r.w;
    }

    template <typename A>
    static CONSTEXPR TVec4<A> project(const TMat44& projectionMatrix, TVec4<A> vertice) {
        vertice = projectionMatrix * vertice;
        return { vertice.xyz / vertice.w, 1 };
    }

    /**
     * Constructs a 3x3 matrix from the upper-left corner of this 4x4 matrix
     */
    inline constexpr TMat33<T> upperLeft() const {
        return TMat33<T>(m_value[0].xyz, m_value[1].xyz, m_value[2].xyz);
    }
};

// ----------------------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------------------

// Since the matrix code could become pretty big quickly, we don't inline most
// operations.

template <typename T>
CONSTEXPR TMat44<T>::TMat44() {
    m_value[0] = col_type(1, 0, 0, 0);
    m_value[1] = col_type(0, 1, 0, 0);
    m_value[2] = col_type(0, 0, 1, 0);
    m_value[3] = col_type(0, 0, 0, 1);
}

template <typename T>
template <typename U>
CONSTEXPR TMat44<T>::TMat44(U v) {
    m_value[0] = col_type(v, 0, 0, 0);
    m_value[1] = col_type(0, v, 0, 0);
    m_value[2] = col_type(0, 0, v, 0);
    m_value[3] = col_type(0, 0, 0, v);
}

template<typename T>
template<typename U>
CONSTEXPR TMat44<T>::TMat44(const TVec4<U>& v) {
    m_value[0] = col_type(v.x, 0, 0, 0);
    m_value[1] = col_type(0, v.y, 0, 0);
    m_value[2] = col_type(0, 0, v.z, 0);
    m_value[3] = col_type(0, 0, 0, v.w);
}

// construct from 16 scalars
template<typename T>
template <
    typename A, typename B, typename C, typename D,
    typename E, typename F, typename G, typename H,
    typename I, typename J, typename K, typename L,
    typename M, typename N, typename O, typename P>
CONSTEXPR TMat44<T>::TMat44(
        A m00, B m01, C m02, D m03,
        E m10, F m11, G m12, H m13,
        I m20, J m21, K m22, L m23,
        M m30, N m31, O m32, P m33) {
    m_value[0] = col_type(m00, m01, m02, m03);
    m_value[1] = col_type(m10, m11, m12, m13);
    m_value[2] = col_type(m20, m21, m22, m23);
    m_value[3] = col_type(m30, m31, m32, m33);
}

template <typename T>
template <typename U>
CONSTEXPR TMat44<T>::TMat44(const TMat44<U>& rhs) {
    for (size_t col = 0; col < NUM_COLS; ++col) {
        m_value[col] = col_type(rhs[col]);
    }
}

// Construct from 4 column vectors.
template <typename T>
template <typename A, typename B, typename C, typename D>
CONSTEXPR TMat44<T>::TMat44(
        const TVec4<A>& v0, const TVec4<B>& v1,
        const TVec4<C>& v2, const TVec4<D>& v3) {
    m_value[0] = col_type(v0);
    m_value[1] = col_type(v1);
    m_value[2] = col_type(v2);
    m_value[3] = col_type(v3);
}

// Construct from raw array, in column-major form.
template <typename T>
template <typename U>
CONSTEXPR TMat44<T>::TMat44(U const* rawArray) {
    for (size_t col = 0; col < NUM_COLS; ++col) {
        for (size_t row = 0; row < NUM_ROWS; ++row) {
            m_value[col][row] = *rawArray++;
        }
    }
}

template <typename T>
template <typename U>
CONSTEXPR TMat44<T>::TMat44(const TQuaternion<U>& q) {
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
    m_value[0] = col_type(1-yy-zz,    xy+zw,    xz-yw,   0);
    m_value[1] = col_type(  xy-zw,  1-xx-zz,    yz+xw,   0);  // NOLINT
    m_value[2] = col_type(  xz+yw,    yz-xw,  1-xx-yy,   0);  // NOLINT
    m_value[3] = col_type(      0,        0,        0,   1);  // NOLINT
}

template <typename T>
template <typename U>
CONSTEXPR TMat44<T>::TMat44(const TMat33<U>& m) {
    m_value[0] = col_type(m[0][0], m[0][1], m[0][2], 0);
    m_value[1] = col_type(m[1][0], m[1][1], m[1][2], 0);
    m_value[2] = col_type(m[2][0], m[2][1], m[2][2], 0);
    m_value[3] = col_type(      0,       0,       0, 1);  // NOLINT
}

template <typename T>
template <typename U, typename V>
CONSTEXPR TMat44<T>::TMat44(const TMat33<U>& m, const TVec3<V>& v) {
    m_value[0] = col_type(m[0][0], m[0][1], m[0][2], 0);
    m_value[1] = col_type(m[1][0], m[1][1], m[1][2], 0);
    m_value[2] = col_type(m[2][0], m[2][1], m[2][2], 0);
    m_value[3] = col_type(   v[0],    v[1],    v[2], 1);  // NOLINT
}

template <typename T>
template <typename U, typename V>
CONSTEXPR TMat44<T>::TMat44(const TMat33<U>& m, const TVec4<V>& v) {
    m_value[0] = col_type(m[0][0], m[0][1], m[0][2],    0);  // NOLINT
    m_value[1] = col_type(m[1][0], m[1][1], m[1][2],    0);  // NOLINT
    m_value[2] = col_type(m[2][0], m[2][1], m[2][2],    0);  // NOLINT
    m_value[3] = col_type(   v[0],    v[1],    v[2], v[3]);  // NOLINT
}

// ----------------------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------------------

template <typename T>
CONSTEXPR TMat44<T> TMat44<T>::ortho(T left, T right, T bottom, T top, T near, T far) {
    TMat44<T> m;
    m[0][0] =  2 / (right - left);
    m[1][1] =  2 / (top   - bottom);
    m[2][2] = -2 / (far   - near);
    m[3][0] = -(right + left)   / (right - left);
    m[3][1] = -(top   + bottom) / (top   - bottom);
    m[3][2] = -(far   + near)   / (far   - near);
    return m;
}

template <typename T>
CONSTEXPR TMat44<T> TMat44<T>::frustum(T left, T right, T bottom, T top, T near, T far) {
    TMat44<T> m;
    m[0][0] =  (2 * near) / (right - left);
    m[1][1] =  (2 * near) / (top   - bottom);
    m[2][0] =  (right + left)   / (right - left);
    m[2][1] =  (top   + bottom) / (top   - bottom);
    m[2][2] = -(far   + near)   / (far   - near);
    m[2][3] = -1;
    m[3][2] = -(2 * far * near) / (far   - near);
    m[3][3] =  0;
    return m;
}

template <typename T>
CONSTEXPR TMat44<T> TMat44<T>::perspective(T fov, T aspect, T near, T far, TMat44::Fov direction) {
    T h;
    T w;

    if (direction == TMat44::Fov::VERTICAL) {
        h = std::tan(fov * M_PI / 360.0f) * near;
        w = h * aspect;
    } else {
        w = std::tan(fov * M_PI / 360.0f) * near;
        h = w / aspect;
    }
    return frustum(-w, w, -h, h, near, far);
}

/*
 * Returns a matrix representing the pose of a virtual camera looking towards -Z in its
 * local Y-up coordinate system. "eye" is where the camera is located, "center" is the points its
 * looking at and "up" defines where the Y axis of the camera's local coordinate system is.
 */
template <typename T>
template <typename A, typename B, typename C>
CONSTEXPR TMat44<T> TMat44<T>::lookAt(const TVec3<A>& eye, const TVec3<B>& center, const TVec3<C>& up) {
    TVec3<T> z_axis(normalize(center - eye));
    TVec3<T> norm_up(normalize(up));
    if (std::abs(dot(z_axis, norm_up)) > 0.999) {
        // Fix up vector if we're degenerate (looking straight up, basically)
        norm_up = { norm_up.z, norm_up.x, norm_up.y };
    }
    TVec3<T> x_axis(normalize(cross(z_axis, norm_up)));
    TVec3<T> y_axis(cross(x_axis, z_axis));
    return TMat44<T>(
            TVec4<T>(x_axis, 0),
            TVec4<T>(y_axis, 0),
            TVec4<T>(-z_axis, 0),
            TVec4<T>(eye, 1));
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
CONSTEXPR typename TMat44<T>::col_type PURE operator *(const TMat44<T>& lhs, const TVec4<U>& rhs) {
    // Result is initialized to zero.
    typename TMat44<T>::col_type result;
    for (size_t col = 0; col < TMat44<T>::NUM_COLS; ++col) {
        result += lhs[col] * rhs[col];
    }
    return result;
}

// mat44 * vec3, result is vec3( mat44 * {vec3, 1} )
template <typename T, typename U>
CONSTEXPR typename TMat44<T>::col_type PURE operator *(const TMat44<T>& lhs, const TVec3<U>& rhs) {
    return lhs * TVec4<U>{ rhs, 1 };
}


// row-vector * matrix, result is a vector of the same type than the input vector
template <typename T, typename U>
CONSTEXPR typename TMat44<U>::row_type PURE operator *(const TVec4<U>& lhs, const TMat44<T>& rhs) {
    typename TMat44<U>::row_type result(TMat44<U>::row_type::NO_INIT);
    for (size_t col = 0; col < TMat44<T>::NUM_COLS; ++col) {
        result[col] = dot(lhs, rhs[col]);
    }
    return result;
}

// matrix * scalar, result is a matrix of the same type than the input matrix
template <typename T, typename U>
constexpr typename std::enable_if<std::is_arithmetic<U>::value, TMat44<T>>::type PURE
operator *(TMat44<T> lhs, U rhs) {
    return lhs *= rhs;
}

// scalar * matrix, result is a matrix of the same type than the input matrix
template <typename T, typename U>
constexpr typename std::enable_if<std::is_arithmetic<U>::value, TMat44<T>>::type PURE
operator *(U lhs, const TMat44<T>& rhs) {
    return rhs * lhs;
}

// ----------------------------------------------------------------------------------------

/* FIXME: this should go into TMatSquareFunctions<> but for some reason
 * BASE<T>::col_type is not accessible from there (???)
 */
template<typename T>
typename TMat44<T>::col_type PURE diag(const TMat44<T>& m) {
    return matrix::diag(m);
}

} // namespace details

// ----------------------------------------------------------------------------------------

typedef details::TMat44<double> mat4d;
typedef details::TMat44<float> mat4;
typedef details::TMat44<float> mat4f;

// ----------------------------------------------------------------------------------------
}  // namespace android

#undef PURE
#undef CONSTEXPR
