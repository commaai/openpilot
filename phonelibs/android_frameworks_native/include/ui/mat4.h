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

#ifndef UI_MAT4_H
#define UI_MAT4_H

#include <stdint.h>
#include <sys/types.h>

#include <ui/vec4.h>
#include <utils/String8.h>

#define TMAT_IMPLEMENTATION
#include <ui/TMatHelpers.h>

#define PURE __attribute__((pure))

namespace android {
// -------------------------------------------------------------------------------------

template <typename T>
class tmat44 :  public TVecUnaryOperators<tmat44, T>,
                public TVecComparisonOperators<tmat44, T>,
                public TVecAddOperators<tmat44, T>,
                public TMatProductOperators<tmat44, T>,
                public TMatSquareFunctions<tmat44, T>,
                public TMatDebug<tmat44, T>
{
public:
    enum no_init { NO_INIT };
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef size_t size_type;
    typedef tvec4<T> col_type;
    typedef tvec4<T> row_type;

    // size of a column (i.e.: number of rows)
    enum { COL_SIZE = col_type::SIZE };
    static inline size_t col_size() { return COL_SIZE; }

    // size of a row (i.e.: number of columns)
    enum { ROW_SIZE = row_type::SIZE };
    static inline size_t row_size() { return ROW_SIZE; }
    static inline size_t size()     { return row_size(); }  // for TVec*<>

private:

    /*
     *  <--  N columns  -->
     *
     *  a00 a10 a20 ... aN0    ^
     *  a01 a11 a21 ... aN1    |
     *  a02 a12 a22 ... aN2  M rows
     *  ...                    |
     *  a0M a1M a2M ... aNM    v
     *
     *  COL_SIZE = M
     *  ROW_SIZE = N
     *  m[0] = [a00 a01 a02 ... a01M]
     */

    col_type mValue[ROW_SIZE];

public:
    // array access
    inline col_type const& operator [] (size_t i) const { return mValue[i]; }
    inline col_type&       operator [] (size_t i)       { return mValue[i]; }

    T const* asArray() const { return &mValue[0][0]; }

    // -----------------------------------------------------------------------
    // we don't provide copy-ctor and operator= on purpose
    // because we want the compiler generated versions

    /*
     *  constructors
     */

    // leaves object uninitialized. use with caution.
    explicit tmat44(no_init) { }

    // initialize to identity
    tmat44();

    // initialize to Identity*scalar.
    template<typename U>
    explicit tmat44(U v);

    // sets the diagonal to the passed vector
    template <typename U>
    explicit tmat44(const tvec4<U>& rhs);

    // construct from another matrix of the same size
    template <typename U>
    explicit tmat44(const tmat44<U>& rhs);

    // construct from 4 column vectors
    template <typename A, typename B, typename C, typename D>
    tmat44(const tvec4<A>& v0, const tvec4<B>& v1, const tvec4<C>& v2, const tvec4<D>& v3);

    // construct from 16 scalars
    template <
        typename A, typename B, typename C, typename D,
        typename E, typename F, typename G, typename H,
        typename I, typename J, typename K, typename L,
        typename M, typename N, typename O, typename P>
    tmat44( A m00, B m01, C m02, D m03,
            E m10, F m11, G m12, H m13,
            I m20, J m21, K m22, L m23,
            M m30, N m31, O m32, P m33);

    // construct from a C array
    template <typename U>
    explicit tmat44(U const* rawArray);

    /*
     *  helpers
     */

    static tmat44 ortho(T left, T right, T bottom, T top, T near, T far);

    static tmat44 frustum(T left, T right, T bottom, T top, T near, T far);

    template <typename A, typename B, typename C>
    static tmat44 lookAt(const tvec3<A>& eye, const tvec3<B>& center, const tvec3<C>& up);

    template <typename A>
    static tmat44 translate(const tvec4<A>& t);

    template <typename A>
    static tmat44 scale(const tvec4<A>& s);

    template <typename A, typename B>
    static tmat44 rotate(A radian, const tvec3<B>& about);
};

// ----------------------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------------------

/*
 * Since the matrix code could become pretty big quickly, we don't inline most
 * operations.
 */

template <typename T>
tmat44<T>::tmat44() {
    mValue[0] = col_type(1,0,0,0);
    mValue[1] = col_type(0,1,0,0);
    mValue[2] = col_type(0,0,1,0);
    mValue[3] = col_type(0,0,0,1);
}

template <typename T>
template <typename U>
tmat44<T>::tmat44(U v) {
    mValue[0] = col_type(v,0,0,0);
    mValue[1] = col_type(0,v,0,0);
    mValue[2] = col_type(0,0,v,0);
    mValue[3] = col_type(0,0,0,v);
}

template<typename T>
template<typename U>
tmat44<T>::tmat44(const tvec4<U>& v) {
    mValue[0] = col_type(v.x,0,0,0);
    mValue[1] = col_type(0,v.y,0,0);
    mValue[2] = col_type(0,0,v.z,0);
    mValue[3] = col_type(0,0,0,v.w);
}

// construct from 16 scalars
template<typename T>
template <
    typename A, typename B, typename C, typename D,
    typename E, typename F, typename G, typename H,
    typename I, typename J, typename K, typename L,
    typename M, typename N, typename O, typename P>
tmat44<T>::tmat44(  A m00, B m01, C m02, D m03,
                    E m10, F m11, G m12, H m13,
                    I m20, J m21, K m22, L m23,
                    M m30, N m31, O m32, P m33) {
    mValue[0] = col_type(m00, m01, m02, m03);
    mValue[1] = col_type(m10, m11, m12, m13);
    mValue[2] = col_type(m20, m21, m22, m23);
    mValue[3] = col_type(m30, m31, m32, m33);
}

template <typename T>
template <typename U>
tmat44<T>::tmat44(const tmat44<U>& rhs) {
    for (size_t r=0 ; r<row_size() ; r++)
        mValue[r] = rhs[r];
}

template <typename T>
template <typename A, typename B, typename C, typename D>
tmat44<T>::tmat44(const tvec4<A>& v0, const tvec4<B>& v1, const tvec4<C>& v2, const tvec4<D>& v3) {
    mValue[0] = v0;
    mValue[1] = v1;
    mValue[2] = v2;
    mValue[3] = v3;
}

template <typename T>
template <typename U>
tmat44<T>::tmat44(U const* rawArray) {
    for (size_t r=0 ; r<row_size() ; r++)
        for (size_t c=0 ; c<col_size() ; c++)
            mValue[r][c] = *rawArray++;
}

// ----------------------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------------------

template <typename T>
tmat44<T> tmat44<T>::ortho(T left, T right, T bottom, T top, T near, T far) {
    tmat44<T> m;
    m[0][0] =  2 / (right - left);
    m[1][1] =  2 / (top   - bottom);
    m[2][2] = -2 / (far   - near);
    m[3][0] = -(right + left)   / (right - left);
    m[3][1] = -(top   + bottom) / (top   - bottom);
    m[3][2] = -(far   + near)   / (far   - near);
    return m;
}

template <typename T>
tmat44<T> tmat44<T>::frustum(T left, T right, T bottom, T top, T near, T far) {
    tmat44<T> m;
    T A = (right + left)   / (right - left);
    T B = (top   + bottom) / (top   - bottom);
    T C = (far   + near)   / (far   - near);
    T D = (2 * far * near) / (far   - near);
    m[0][0] = (2 * near) / (right - left);
    m[1][1] = (2 * near) / (top   - bottom);
    m[2][0] = A;
    m[2][1] = B;
    m[2][2] = C;
    m[2][3] =-1;
    m[3][2] = D;
    m[3][3] = 0;
    return m;
}

template <typename T>
template <typename A, typename B, typename C>
tmat44<T> tmat44<T>::lookAt(const tvec3<A>& eye, const tvec3<B>& center, const tvec3<C>& up) {
    tvec3<T> L(normalize(center - eye));
    tvec3<T> S(normalize( cross(L, up) ));
    tvec3<T> U(cross(S, L));
    return tmat44<T>(
            tvec4<T>( S, 0),
            tvec4<T>( U, 0),
            tvec4<T>(-L, 0),
            tvec4<T>(-eye, 1));
}

template <typename T>
template <typename A>
tmat44<T> tmat44<T>::translate(const tvec4<A>& t) {
    tmat44<T> r;
    r[3] = t;
    return r;
}

template <typename T>
template <typename A>
tmat44<T> tmat44<T>::scale(const tvec4<A>& s) {
    tmat44<T> r;
    r[0][0] = s[0];
    r[1][1] = s[1];
    r[2][2] = s[2];
    r[3][3] = s[3];
    return r;
}

template <typename T>
template <typename A, typename B>
tmat44<T> tmat44<T>::rotate(A radian, const tvec3<B>& about) {
    tmat44<T> rotation;
    T* r = const_cast<T*>(rotation.asArray());
    T c = cos(radian);
    T s = sin(radian);
    if (about.x==1 && about.y==0 && about.z==0) {
        r[5] = c;   r[10]= c;
        r[6] = s;   r[9] = -s;
    } else if (about.x==0 && about.y==1 && about.z==0) {
        r[0] = c;   r[10]= c;
        r[8] = s;   r[2] = -s;
    } else if (about.x==0 && about.y==0 && about.z==1) {
        r[0] = c;   r[5] = c;
        r[1] = s;   r[4] = -s;
    } else {
        tvec3<B> nabout = normalize(about);
        B x = nabout.x;
        B y = nabout.y;
        B z = nabout.z;
        T nc = 1 - c;
        T xy = x * y;
        T yz = y * z;
        T zx = z * x;
        T xs = x * s;
        T ys = y * s;
        T zs = z * s;
        r[ 0] = x*x*nc +  c;    r[ 4] =  xy*nc - zs;    r[ 8] =  zx*nc + ys;
        r[ 1] =  xy*nc + zs;    r[ 5] = y*y*nc +  c;    r[ 9] =  yz*nc - xs;
        r[ 2] =  zx*nc - ys;    r[ 6] =  yz*nc + xs;    r[10] = z*z*nc +  c;
    }
    return rotation;
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

// matrix * vector, result is a vector of the same type than the input vector
template <typename T, typename U>
typename tmat44<U>::col_type PURE operator *(const tmat44<T>& lv, const tvec4<U>& rv) {
    typename tmat44<U>::col_type result;
    for (size_t r=0 ; r<tmat44<T>::row_size() ; r++)
        result += rv[r]*lv[r];
    return result;
}

// vector * matrix, result is a vector of the same type than the input vector
template <typename T, typename U>
typename tmat44<U>::row_type PURE operator *(const tvec4<U>& rv, const tmat44<T>& lv) {
    typename tmat44<U>::row_type result(tmat44<U>::row_type::NO_INIT);
    for (size_t r=0 ; r<tmat44<T>::row_size() ; r++)
        result[r] = dot(rv, lv[r]);
    return result;
}

// matrix * scalar, result is a matrix of the same type than the input matrix
template <typename T, typename U>
tmat44<T> PURE operator *(const tmat44<T>& lv, U rv) {
    tmat44<T> result(tmat44<T>::NO_INIT);
    for (size_t r=0 ; r<tmat44<T>::row_size() ; r++)
        result[r] = lv[r]*rv;
    return result;
}

// scalar * matrix, result is a matrix of the same type than the input matrix
template <typename T, typename U>
tmat44<T> PURE operator *(U rv, const tmat44<T>& lv) {
    tmat44<T> result(tmat44<T>::NO_INIT);
    for (size_t r=0 ; r<tmat44<T>::row_size() ; r++)
        result[r] = lv[r]*rv;
    return result;
}

// ----------------------------------------------------------------------------------------

/* FIXME: this should go into TMatSquareFunctions<> but for some reason
 * BASE<T>::col_type is not accessible from there (???)
 */
template<typename T>
typename tmat44<T>::col_type PURE diag(const tmat44<T>& m) {
    return matrix::diag(m);
}

// ----------------------------------------------------------------------------------------

typedef tmat44<float> mat4;

// ----------------------------------------------------------------------------------------
}; // namespace android

#undef PURE

#endif /* UI_MAT4_H */
