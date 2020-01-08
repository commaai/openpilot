// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONVERSION_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONVERSION_H

namespace Eigen {

/** \class TensorConversionOp
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor conversion class. This class makes it possible to vectorize
  * type casting operations when the number of scalars per packet in the source
  * and the destination type differ
  */
namespace internal {
template<typename TargetType, typename XprType>
struct traits<TensorConversionOp<TargetType, XprType> >
{
  // Type promotion to handle the case where the types of the lhs and the rhs are different.
  typedef TargetType Scalar;
  typedef typename traits<XprType>::StorageKind StorageKind;
  typedef typename traits<XprType>::Index Index;
  typedef typename XprType::Nested Nested;
  typedef typename remove_reference<Nested>::type _Nested;
  static const int NumDimensions = traits<XprType>::NumDimensions;
  static const int Layout = traits<XprType>::Layout;
  enum { Flags = 0 };
};

template<typename TargetType, typename XprType>
struct eval<TensorConversionOp<TargetType, XprType>, Eigen::Dense>
{
  typedef const TensorConversionOp<TargetType, XprType>& type;
};

template<typename TargetType, typename XprType>
struct nested<TensorConversionOp<TargetType, XprType>, 1, typename eval<TensorConversionOp<TargetType, XprType> >::type>
{
  typedef TensorConversionOp<TargetType, XprType> type;
};

}  // end namespace internal


template <typename TensorEvaluator, typename SrcPacket, typename TgtPacket, int SrcCoeffRatio, int TgtCoeffRatio>
struct PacketConverter {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketConverter(const TensorEvaluator& impl)
      : m_impl(impl) {}

  template<int LoadMode, typename Index>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TgtPacket packet(Index index) const {
    return internal::pcast<SrcPacket, TgtPacket>(m_impl.template packet<LoadMode>(index));
  }

 private:
  const TensorEvaluator& m_impl;
};


template <typename TensorEvaluator, typename SrcPacket, typename TgtPacket>
struct PacketConverter<TensorEvaluator, SrcPacket, TgtPacket, 2, 1> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketConverter(const TensorEvaluator& impl)
      : m_impl(impl) {}

  template<int LoadMode, typename Index>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TgtPacket packet(Index index) const {
    const int SrcPacketSize = internal::unpacket_traits<SrcPacket>::size;

    SrcPacket src1 = m_impl.template packet<LoadMode>(index);
    SrcPacket src2 = m_impl.template packet<LoadMode>(index + SrcPacketSize);
    TgtPacket result = internal::pcast<SrcPacket, TgtPacket>(src1, src2);
    return result;
  }

 private:
  const TensorEvaluator& m_impl;
};

template <typename TensorEvaluator, typename SrcPacket, typename TgtPacket>
struct PacketConverter<TensorEvaluator, SrcPacket, TgtPacket, 4, 1> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketConverter(const TensorEvaluator& impl)
      : m_impl(impl) {}

  template<int LoadMode, typename Index>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TgtPacket packet(Index index) const {
    const int SrcPacketSize = internal::unpacket_traits<SrcPacket>::size;

    SrcPacket src1 = m_impl.template packet<LoadMode>(index);
    SrcPacket src2 = m_impl.template packet<LoadMode>(index + SrcPacketSize);
    SrcPacket src3 = m_impl.template packet<LoadMode>(index + 2 * SrcPacketSize);
    SrcPacket src4 = m_impl.template packet<LoadMode>(index + 3 * SrcPacketSize);
    TgtPacket result = internal::pcast<SrcPacket, TgtPacket>(src1, src2, src3, src4);
    return result;
  }

 private:
  const TensorEvaluator& m_impl;
};

template <typename TensorEvaluator, typename SrcPacket, typename TgtPacket>
struct PacketConverter<TensorEvaluator, SrcPacket, TgtPacket, 1, 2> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketConverter(const TensorEvaluator& impl)
      : m_impl(impl), m_maxIndex(impl.dimensions().TotalSize()) {}

  template<int LoadMode, typename Index>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TgtPacket packet(Index index) const {
    const int SrcPacketSize = internal::unpacket_traits<SrcPacket>::size;
    // Only call m_impl.packet() when we have direct access to the underlying data. This
    // ensures that we don't compute the subexpression twice. We may however load some
    // coefficients twice, but in practice this doesn't negatively impact performance.
    if (m_impl.data() && (index + SrcPacketSize < m_maxIndex)) {
      // Force unaligned memory loads since we can't ensure alignment anymore
      return internal::pcast<SrcPacket, TgtPacket>(m_impl.template packet<Unaligned>(index));
    } else {
      const int TgtPacketSize = internal::unpacket_traits<TgtPacket>::size;
      typedef typename internal::unpacket_traits<SrcPacket>::type SrcType;
      typedef typename internal::unpacket_traits<TgtPacket>::type TgtType;
      internal::scalar_cast_op<SrcType, TgtType> converter;
      EIGEN_ALIGN_MAX typename internal::unpacket_traits<TgtPacket>::type values[TgtPacketSize];
      for (int i = 0; i < TgtPacketSize; ++i) {
        values[i] = converter(m_impl.coeff(index+i));
      }
      TgtPacket rslt = internal::pload<TgtPacket>(values);
      return rslt;
    }
  }

 private:
  const TensorEvaluator& m_impl;
  const typename TensorEvaluator::Index m_maxIndex;
};

template<typename TargetType, typename XprType>
class TensorConversionOp : public TensorBase<TensorConversionOp<TargetType, XprType>, ReadOnlyAccessors>
{
  public:
    typedef typename internal::traits<TensorConversionOp>::Scalar Scalar;
    typedef typename internal::traits<TensorConversionOp>::StorageKind StorageKind;
    typedef typename internal::traits<TensorConversionOp>::Index Index;
    typedef typename internal::nested<TensorConversionOp>::type Nested;
    typedef Scalar CoeffReturnType;
    typedef typename NumTraits<Scalar>::Real RealScalar;

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorConversionOp(const XprType& xpr)
        : m_xpr(xpr) {}

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    expression() const { return m_xpr; }

  protected:
    typename XprType::Nested m_xpr;
};

template <bool SameType, typename Eval, typename Scalar> struct ConversionSubExprEval {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(Eval& impl, Scalar*) {
    impl.evalSubExprsIfNeeded(NULL);
    return true;
  }
};

template <typename Eval, typename Scalar> struct ConversionSubExprEval<true, Eval, Scalar> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool run(Eval& impl, Scalar* data) {
    return impl.evalSubExprsIfNeeded(data);
  }
};


// Eval as rvalue
template<typename TargetType, typename ArgType, typename Device>
struct TensorEvaluator<const TensorConversionOp<TargetType, ArgType>, Device>
{
  typedef TensorConversionOp<TargetType, ArgType> XprType;
  typedef typename XprType::Index Index;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;
  typedef TargetType Scalar;
  typedef TargetType CoeffReturnType;
  typedef typename internal::remove_all<typename internal::traits<ArgType>::Scalar>::type SrcType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename PacketType<SrcType, Device>::type PacketSourceType;
  static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;

  enum {
    IsAligned = false,
    PacketAccess = true,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    RawAccess = false
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const XprType& op, const Device& device)
    : m_impl(op.expression(), device)
  {
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_impl.dimensions(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data)
  {
    return ConversionSubExprEval<internal::is_same<TargetType, SrcType>::value, TensorEvaluator<ArgType, Device>, Scalar>::run(m_impl, data);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup()
  {
    m_impl.cleanup();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const
  {
    internal::scalar_cast_op<SrcType, TargetType> converter;
    return converter(m_impl.coeff(index));
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    const bool Vectorizable = TensorEvaluator<ArgType, Device>::PacketAccess &
        internal::type_casting_traits<SrcType, TargetType>::VectorizedCast;
    return PacketConv<LoadMode, Vectorizable>::run(m_impl, index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool vectorized) const {
    const double cast_cost = TensorOpCost::CastCost<SrcType, TargetType>();
    if (vectorized) {
      const double SrcCoeffRatio =
          internal::type_casting_traits<SrcType, TargetType>::SrcCoeffRatio;
      const double TgtCoeffRatio =
          internal::type_casting_traits<SrcType, TargetType>::TgtCoeffRatio;
      return m_impl.costPerCoeff(vectorized) * (SrcCoeffRatio / PacketSize) +
          TensorOpCost(0, 0, TgtCoeffRatio * (cast_cost / PacketSize));
    } else {
      return m_impl.costPerCoeff(vectorized) + TensorOpCost(0, 0, cast_cost);
    }
  }

  EIGEN_DEVICE_FUNC Scalar* data() const { return NULL; }

  protected:
  template <int LoadMode, bool ActuallyVectorize>
  struct PacketConv {
    static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType run(const TensorEvaluator<ArgType, Device>& impl, Index index) {
      internal::scalar_cast_op<SrcType, TargetType> converter;
      EIGEN_ALIGN_MAX typename internal::remove_const<CoeffReturnType>::type values[PacketSize];
      for (int i = 0; i < PacketSize; ++i) {
        values[i] = converter(impl.coeff(index+i));
      }
      PacketReturnType rslt = internal::pload<PacketReturnType>(values);
      return rslt;
    }
  };

  template <int LoadMode>
  struct PacketConv<LoadMode, true> {
    static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType run(const TensorEvaluator<ArgType, Device>& impl, Index index) {
      const int SrcCoeffRatio = internal::type_casting_traits<SrcType, TargetType>::SrcCoeffRatio;
      const int TgtCoeffRatio = internal::type_casting_traits<SrcType, TargetType>::TgtCoeffRatio;
      PacketConverter<TensorEvaluator<ArgType, Device>, PacketSourceType, PacketReturnType,
                      SrcCoeffRatio, TgtCoeffRatio> converter(impl);
      return converter.template packet<LoadMode>(index);
    }
  };

  TensorEvaluator<ArgType, Device> m_impl;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONVERSION_H
