// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H
#define EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H

namespace Eigen {
namespace internal {


/** \internal
 * \brief Template functor to compute the modulo between an array and a scalar.
 */
template <typename Scalar>
struct scalar_mod_op {
  EIGEN_DEVICE_FUNC scalar_mod_op(const Scalar& divisor) : m_divisor(divisor) {}
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a) const { return a % m_divisor; }
  const Scalar m_divisor;
};
template <typename Scalar>
struct functor_traits<scalar_mod_op<Scalar> >
{ enum { Cost = scalar_div_cost<Scalar,false>::value, PacketAccess = false }; };


/** \internal
 * \brief Template functor to compute the modulo between 2 arrays.
 */
template <typename Scalar>
struct scalar_mod2_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_mod2_op);
  EIGEN_DEVICE_FUNC inline Scalar operator() (const Scalar& a, const Scalar& b) const { return a % b; }
};
template <typename Scalar>
struct functor_traits<scalar_mod2_op<Scalar> >
{ enum { Cost = scalar_div_cost<Scalar,false>::value, PacketAccess = false }; };

template <typename Scalar>
struct scalar_fmod_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_fmod_op);
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& a, const Scalar& b) const {
    return numext::fmod(a, b);
  }
};
template <typename Scalar>
struct functor_traits<scalar_fmod_op<Scalar> > {
  enum { Cost = 13,  // Reciprocal throughput of FPREM on Haswell.
         PacketAccess = false };
};


/** \internal
  * \brief Template functor to compute the sigmoid of a scalar
  * \sa class CwiseUnaryOp, ArrayBase::sigmoid()
  */
template <typename T>
struct scalar_sigmoid_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sigmoid_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const {
    const T one = T(1);
    return one / (one + numext::exp(-x));
  }

  template <typename Packet> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Packet packetOp(const Packet& x) const {
    const Packet one = pset1<Packet>(T(1));
    return pdiv(one, padd(one, pexp(pnegate(x))));
  }
};

template <typename T>
struct functor_traits<scalar_sigmoid_op<T> > {
  enum {
    Cost = NumTraits<T>::AddCost * 2 + NumTraits<T>::MulCost * 6,
    PacketAccess = packet_traits<T>::HasAdd && packet_traits<T>::HasDiv &&
                   packet_traits<T>::HasNegate && packet_traits<T>::HasExp
  };
};


template<typename Reducer, typename Device>
struct reducer_traits {
  enum {
    Cost = 1,
    PacketAccess = false
  };
};

// Standard reduction functors
template <typename T> struct SumReducer
{
  static const bool PacketAccess = packet_traits<T>::HasAdd;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    internal::scalar_sum_op<T> sum_op;
    *accum = sum_op(*accum, t);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = padd<Packet>(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    internal::scalar_cast_op<int, T> conv;
    return conv(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(initialize());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    internal::scalar_sum_op<T> sum_op;
    return sum_op(saccum, predux(vaccum));
  }
};

template <typename T, typename Device>
struct reducer_traits<SumReducer<T>, Device> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = PacketType<T, Device>::HasAdd
  };
};


template <typename T> struct MeanReducer
{
  static const bool PacketAccess = packet_traits<T>::HasAdd && !NumTraits<T>::IsInteger;
  static const bool IsStateful = true;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  MeanReducer() : scalarCount_(0), packetCount_(0) { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) {
    internal::scalar_sum_op<T> sum_op;
    *accum = sum_op(*accum, t);
    scalarCount_++;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) {
    (*accum) = padd<Packet>(*accum, p);
    packetCount_++;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    internal::scalar_cast_op<int, T> conv;
    return conv(0);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(initialize());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum / scalarCount_;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return pdiv(vaccum, pset1<Packet>(packetCount_));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    internal::scalar_sum_op<T> sum_op;
    return sum_op(saccum, predux(vaccum)) / (scalarCount_ + packetCount_ * unpacket_traits<Packet>::size);
  }

  protected:
    DenseIndex scalarCount_;
    DenseIndex packetCount_;
};

template <typename T, typename Device>
struct reducer_traits<MeanReducer<T>, Device> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = PacketType<T, Device>::HasAdd
  };
};


template <typename T, bool IsMax = true, bool IsInteger = true>
struct MinMaxBottomValue {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T bottom_value() {
    return Eigen::NumTraits<T>::lowest();
  }
};
template <typename T>
struct MinMaxBottomValue<T, true, false> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T bottom_value() {
    return -Eigen::NumTraits<T>::infinity();
  }
};
template <typename T>
struct MinMaxBottomValue<T, false, true> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T bottom_value() {
    return Eigen::NumTraits<T>::highest();
  }
};
template <typename T>
struct MinMaxBottomValue<T, false, false> {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE T bottom_value() {
    return Eigen::NumTraits<T>::infinity();
  }
};


template <typename T> struct MaxReducer
{
  static const bool PacketAccess = packet_traits<T>::HasMax;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    if (t > *accum) { *accum = t; }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = pmax<Packet>(*accum, p);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return MinMaxBottomValue<T, true, Eigen::NumTraits<T>::IsInteger>::bottom_value();
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(initialize());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    return numext::maxi(saccum, predux_max(vaccum));
  }
};

template <typename T, typename Device>
struct reducer_traits<MaxReducer<T>, Device> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = PacketType<T, Device>::HasMax
  };
};


template <typename T> struct MinReducer
{
  static const bool PacketAccess = packet_traits<T>::HasMin;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    if (t < *accum) { *accum = t; }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = pmin<Packet>(*accum, p);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return MinMaxBottomValue<T, false, Eigen::NumTraits<T>::IsInteger>::bottom_value();
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(initialize());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    return numext::mini(saccum, predux_min(vaccum));
  }
};

template <typename T, typename Device>
struct reducer_traits<MinReducer<T>, Device> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = PacketType<T, Device>::HasMin
  };
};


template <typename T> struct ProdReducer
{
  static const bool PacketAccess = packet_traits<T>::HasMul;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    internal::scalar_product_op<T> prod_op;
    (*accum) = prod_op(*accum, t);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p, Packet* accum) const {
    (*accum) = pmul<Packet>(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    internal::scalar_cast_op<int, T> conv;
    return conv(1);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return pset1<Packet>(initialize());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalizeBoth(const T saccum, const Packet& vaccum) const {
    internal::scalar_product_op<T> prod_op;
    return prod_op(saccum, predux_mul(vaccum));
  }
};

template <typename T, typename Device>
struct reducer_traits<ProdReducer<T>, Device> {
  enum {
    Cost = NumTraits<T>::MulCost,
    PacketAccess = PacketType<T, Device>::HasMul
  };
};


struct AndReducer
{
  static const bool PacketAccess = false;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(bool t, bool* accum) const {
    *accum = *accum && t;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool initialize() const {
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool finalize(bool accum) const {
    return accum;
  }
};

template <typename Device>
struct reducer_traits<AndReducer, Device> {
  enum {
    Cost = 1,
    PacketAccess = false
  };
};


struct OrReducer {
  static const bool PacketAccess = false;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(bool t, bool* accum) const {
    *accum = *accum || t;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool initialize() const {
    return false;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool finalize(bool accum) const {
    return accum;
  }
};

template <typename Device>
struct reducer_traits<OrReducer, Device> {
  enum {
    Cost = 1,
    PacketAccess = false
  };
};


// Argmin/Argmax reducers
template <typename T> struct ArgMaxTupleReducer
{
  static const bool PacketAccess = false;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    if (t.second > accum->second) { *accum = t; }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return T(0, NumTraits<typename T::second_type>::lowest());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T& accum) const {
    return accum;
  }
};

template <typename T, typename Device>
struct reducer_traits<ArgMaxTupleReducer<T>, Device> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = false
  };
};


template <typename T> struct ArgMinTupleReducer
{
  static const bool PacketAccess = false;
  static const bool IsStateful = false;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T& t, T* accum) const {
    if (t.second < accum->second) { *accum = t; }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return T(0, NumTraits<typename T::second_type>::highest());
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T& accum) const {
    return accum;
  }
};

template <typename T, typename Device>
struct reducer_traits<ArgMinTupleReducer<T>, Device> {
  enum {
    Cost = NumTraits<T>::AddCost,
    PacketAccess = false
  };
};


template <typename T, typename Index, size_t NumDims>
class GaussianGenerator {
 public:
  static const bool PacketAccess = false;

  EIGEN_DEVICE_FUNC GaussianGenerator(const array<T, NumDims>& means,
                                      const array<T, NumDims>& std_devs)
      : m_means(means)
  {
    for (size_t i = 0; i < NumDims; ++i) {
      m_two_sigmas[i] = std_devs[i] * std_devs[i] * 2;
    }
  }

  EIGEN_DEVICE_FUNC T operator()(const array<Index, NumDims>& coordinates) const {
    T tmp = T(0);
    for (size_t i = 0; i < NumDims; ++i) {
      T offset = coordinates[i] - m_means[i];
      tmp += offset * offset / m_two_sigmas[i];
    }
    return numext::exp(-tmp);
  }

 private:
  array<T, NumDims> m_means;
  array<T, NumDims> m_two_sigmas;
};

template <typename T, typename Index, size_t NumDims>
struct functor_traits<GaussianGenerator<T, Index, NumDims> > {
  enum {
    Cost = NumDims * (2 * NumTraits<T>::AddCost + NumTraits<T>::MulCost +
                      functor_traits<scalar_quotient_op<T, T> >::Cost) +
           functor_traits<scalar_exp_op<T> >::Cost,
    PacketAccess = GaussianGenerator<T, Index, NumDims>::PacketAccess
  };
};

} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_FUNCTORS_H
