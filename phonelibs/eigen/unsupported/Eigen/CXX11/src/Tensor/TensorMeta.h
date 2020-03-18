// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_META_H
#define EIGEN_CXX11_TENSOR_TENSOR_META_H

namespace Eigen {

template<bool cond> struct Cond {};

template<typename T1, typename T2> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
const T1& choose(Cond<true>, const T1& first, const T2&) {
  return first;
}

template<typename T1, typename T2> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
const T2& choose(Cond<false>, const T1&, const T2& second) {
  return second;
}


template <typename T, typename X, typename Y>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T divup(const X x, const Y y) {
  return static_cast<T>((x + y - 1) / y);
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T divup(const T x, const T y) {
  return static_cast<T>((x + y - 1) / y);
}

template <size_t n> struct max_n_1 {
  static const size_t size = n;
};
template <> struct max_n_1<0> {
  static const size_t size = 1;
};


// Default packet types
template <typename Scalar, typename Device>
struct PacketType : internal::packet_traits<Scalar> {
  typedef typename internal::packet_traits<Scalar>::type type;
};

// For CUDA packet types when using a GpuDevice
#if defined(EIGEN_USE_GPU) && defined(__CUDACC__) && defined(EIGEN_HAS_CUDA_FP16)
template <>
struct PacketType<half, GpuDevice> {
  typedef half2 type;
  static const int size = 2;
  enum {
    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasNegate = 1,
    HasAbs    = 1,
    HasArg    = 0,
    HasAbs2   = 0,
    HasMin    = 1,
    HasMax    = 1,
    HasConj   = 0,
    HasSetLinear = 0,
    HasBlend  = 0,

    HasDiv    = 1,
    HasSqrt   = 1,
    HasRsqrt  = 1,
    HasExp    = 1,
    HasLog    = 1,
    HasLog1p  = 0,
    HasLog10  = 0,
    HasPow    = 1,
  };
};
#endif

#if defined(EIGEN_USE_SYCL)
template <typename T>
  struct PacketType<T, SyclDevice> {
  typedef T type;
  static const int size = 1;
  enum {
    HasAdd    = 0,
    HasSub    = 0,
    HasMul    = 0,
    HasNegate = 0,
    HasAbs    = 0,
    HasArg    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0,
    HasBlend  = 0
  };
};
#endif


// Tuple mimics std::pair but works on e.g. nvcc.
template <typename U, typename V> struct Tuple {
 public:
  U first;
  V second;

  typedef U first_type;
  typedef V second_type;

  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple() : first(), second() {}

  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple(const U& f, const V& s) : first(f), second(s) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple& operator= (const Tuple& rhs) {
    if (&rhs == this) return *this;
    first = rhs.first;
    second = rhs.second;
    return *this;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void swap(Tuple& rhs) {
    using numext::swap;
    swap(first, rhs.first);
    swap(second, rhs.second);
  }
};

template <typename U, typename V>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
bool operator==(const Tuple<U, V>& x, const Tuple<U, V>& y) {
  return (x.first == y.first && x.second == y.second);
}

template <typename U, typename V>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
bool operator!=(const Tuple<U, V>& x, const Tuple<U, V>& y) {
  return !(x == y);
}


// Can't use std::pairs on cuda devices
template <typename Idx> struct IndexPair {
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE IndexPair() : first(0), second(0) {}
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE IndexPair(Idx f, Idx s) : first(f), second(s) {}

  EIGEN_DEVICE_FUNC void set(IndexPair<Idx> val) {
    first = val.first;
    second = val.second;
  }

  Idx first;
  Idx second;
};


#ifdef EIGEN_HAS_SFINAE
namespace internal {

  template<typename IndexType, Index... Is>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  array<Index, sizeof...(Is)> customIndices2Array(IndexType& idx, numeric_list<Index, Is...>) {
    return { idx[Is]... };
  }
  template<typename IndexType>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  array<Index, 0> customIndices2Array(IndexType&, numeric_list<Index>) {
    return array<Index, 0>();
  }

  /** Make an array (for index/dimensions) out of a custom index */
  template<typename Index, std::size_t NumIndices, typename IndexType>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  array<Index, NumIndices> customIndices2Array(IndexType& idx) {
    return customIndices2Array(idx, typename gen_numeric_list<Index, NumIndices>::type{});
  }


  template <typename B, typename D>
  struct is_base_of
  {

    typedef char (&yes)[1];
    typedef char (&no)[2];

    template <typename BB, typename DD>
    struct Host
    {
      operator BB*() const;
      operator DD*();
    };

    template<typename T>
    static yes check(D*, T);
    static no check(B*, int);

    static const bool value = sizeof(check(Host<B,D>(), int())) == sizeof(yes);
  };

}
#endif



}  // namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_META_H
