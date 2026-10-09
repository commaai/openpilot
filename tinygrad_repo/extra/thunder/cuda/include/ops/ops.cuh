/**
 * @file
 * @brief A collection of all of the operations that ThunderKittens defines.
 */

#pragma once

#include "thread/thread.cuh"
#include "group/group.cuh"
#include "device/device.cuh"

namespace kittens {

// Operator overloading, which defaults to warp scope.

// Tile operators

template<ducks::rt::all T, typename U>
__device__ static inline T operator+(const T &lhs, const U &rhs) {
    T dst;
    warp::add(dst, lhs, rhs);
    return dst;
}
template<ducks::rt::all T, typename U>
__device__ static inline void operator+=(T &lhs, const U &rhs) {
    warp::add(lhs, lhs, rhs);
}
template<ducks::rt::all T, typename U>
__device__ static inline T operator-(const T &lhs, const U &rhs) {
    T dst;
    warp::sub(dst, lhs, rhs);
    return dst;
}
template<ducks::rt::all T, typename U>
__device__ static inline void operator-=(T &lhs, const U &rhs) {
    warp::sub(lhs, lhs, rhs);
}
template<ducks::rt::all T, typename U>
__device__ static inline T operator*(const T &lhs, const U &rhs) {
    T dst;
    warp::mul(dst, lhs, rhs);
    return dst;
}
template<ducks::rt::all T, typename U>
__device__ static inline void operator*=(T &lhs, const U &rhs) {
    warp::mul(lhs, lhs, rhs);
}
template<ducks::rt::all T, typename U>
__device__ static inline T operator/(const T &lhs, const U &rhs) {
    T dst;
    warp::div(dst, lhs, rhs);
    return dst;
}
template<ducks::rt::all T, typename U>
__device__ static inline void operator/=(T &lhs, const U &rhs) {
    warp::div(lhs, lhs, rhs);
}
template<ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator+(const T &src, const V &row_values) {
    T dst;
    warp::add_row(dst, src, row_values);
    return dst;
}
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline T operator+(const T &src, const V &row_values) {
    T dst;
    warp::add_row(dst, src, row_values);
    return dst;
}
template<ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator+=(T &lhs, const V &row_values) {
    warp::add_row(lhs, lhs, row_values);
}
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void operator+=(T &lhs, const V &row_values) {
    warp::add_row(lhs, lhs, row_values);
}
template<ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator-(const T &src, const V &row_values) {
    T dst;
    warp::sub_row(dst, src, row_values);
    return dst;
}
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline T operator-(const T &src, const V &row_values) {
    T dst;
    warp::sub_row(dst, src, row_values);
    return dst;
}
template<ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator-=(T &lhs, const V &row_values) {
    warp::sub_row(lhs, lhs, row_values);
}
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void operator-=(T &lhs, const V &row_values) {
    warp::sub_row(lhs, lhs, row_values);
}
template<ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator*(const T &src, const V &row_values) {
    T dst;
    warp::mul_row(dst, src, row_values);
    return dst;
}
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline T operator*(const T &src, const V &row_values) {
    T dst;
    warp::mul_row(dst, src, row_values);
    return dst;
}
template<ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator*=(T &lhs, const V &row_values) {
    warp::mul_row(lhs, lhs, row_values);
}
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void operator*=(T &lhs, const V &row_values) {
    warp::mul_row(lhs, lhs, row_values);
}
template<ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator/(const T &src, const V &row_values) {
    T dst;
    warp::div_row(dst, src, row_values);
    return dst;
}
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline T operator/(const T &src, const V &row_values) {
    T dst;
    warp::div_row(dst, src, row_values);
    return dst;
}
template<ducks::rt::row_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator/=(T &lhs, const V &row_values) {
    warp::div_row(lhs, lhs, row_values);
}
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void operator/=(T &lhs, const V &row_values) {
    warp::div_row(lhs, lhs, row_values);
}
template<ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline T operator+(const T &src, const V &col_values) {
    T dst;
    warp::add_col(dst, src, col_values);
    return dst;
}
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator+(const T &src, const V &col_values) {
    T dst;
    warp::add_col(dst, src, col_values);
    return dst;
}
template<ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline void operator+=(T &lhs, const V &col_values) {
    warp::add_col(lhs, lhs, col_values);
}
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator+=(T &lhs, const V &col_values) {
    warp::add_col(lhs, lhs, col_values);
}
template<ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline T operator-(const T &src, const V &col_values) {
    T dst;
    warp::sub_col(dst, src, col_values);
    return dst;
}
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator-(const T &src, const V &col_values) {
    T dst;
    warp::sub_col(dst, src, col_values);
    return dst;
}
template<ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline void operator-=(T &lhs, const V &col_values) {
    warp::sub_col(lhs, lhs, col_values);
}
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator-=(T &lhs, const V &col_values) {
    warp::sub_col(lhs, lhs, col_values);
}
template<ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline T operator*(const T &src, const V &col_values) {
    T dst;
    warp::mul_col(dst, src, col_values);
    return dst;
}
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator*(const T &src, const V &col_values) {
    T dst;
    warp::mul_col(dst, src, col_values);
    return dst;
}
template<ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline void operator*=(T &lhs, const V &col_values) {
    warp::mul_col(lhs, lhs, col_values);
}
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator*=(T &lhs, const V &col_values) {
    warp::mul_col(lhs, lhs, col_values);
}
template<ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline T operator/(const T &src, const V &col_values) {
    T dst;
    warp::div_col(dst, src, col_values);
    return dst;
}
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline T operator/(const T &src, const V &col_values) {
    T dst;
    warp::div_col(dst, src, col_values);
    return dst;
}
template<ducks::rt::row_layout T, ducks::rv::align_layout V>
__device__ static inline void operator/=(T &lhs, const V &col_values) {
    warp::div_col(lhs, lhs, col_values);
}
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void operator/=(T &lhs, const V &col_values) {
    warp::div_col(lhs, lhs, col_values);
}

// Vector operators

template<ducks::rv::all T, typename U>
__device__ static inline T operator+(const T &lhs, const U &rhs) {
    T dst;
    warp::add(dst, lhs, rhs);
    return dst;
}
template<ducks::rv::all T, typename U>
__device__ static inline void operator+=(T &lhs, const U &rhs) {
    warp::add(lhs, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T operator-(const T &lhs, const U &rhs) {
    T dst;
    warp::sub(dst, lhs, rhs);
    return dst;
}
template<ducks::rv::all T, typename U>
__device__ static inline void operator-=(T &lhs, const U &rhs) {
    warp::sub(lhs, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T operator*(const T &lhs, const U &rhs) {
    T dst;
    warp::mul(dst, lhs, rhs);
    return dst;
}
template<ducks::rv::all T, typename U>
__device__ static inline void operator*=(T &lhs, const U &rhs) {
    warp::mul(lhs, lhs, rhs);
}
template<ducks::rv::all T, typename U>
__device__ static inline T operator/(const T &lhs, const U &rhs) {
    T dst;
    warp::div(dst, lhs, rhs);
    return dst;
}
template<ducks::rv::all T, typename U>
__device__ static inline void operator/=(T &lhs, const U &rhs) {
    warp::div(lhs, lhs, rhs);
}

}
