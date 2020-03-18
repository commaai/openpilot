// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
#define EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H

namespace Eigen {

/** \class TensorEvaluator
  * \ingroup CXX11_Tensor_Module
  *
  * \brief The tensor evaluator classes.
  *
  * These classes are responsible for the evaluation of the tensor expression.
  *
  * TODO: add support for more types of expressions, in particular expressions
  * leading to lvalues (slicing, reshaping, etc...)
  */

// Generic evaluator
template<typename Derived, typename Device>
struct TensorEvaluator
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  // NumDimensions is -1 for variable dim tensors
  static const int NumCoords = internal::traits<Derived>::NumDimensions > 0 ?
                               internal::traits<Derived>::NumDimensions : 0;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = (internal::unpacket_traits<PacketReturnType>::size > 1),
    Layout = Derived::Layout,
    CoordAccess = NumCoords > 0,
    RawAccess = true
  };

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const Derived& m, const Device& device)
      : m_data(const_cast<typename internal::traits<Derived>::template MakePointer<Scalar>::Type>(m.data())), m_dims(m.dimensions()), m_device(device), m_impl(m)
  { }

  // Used for accessor extraction in SYCL Managed TensorMap:
  const Derived& derived() const { return m_impl; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* dest) {
    if (dest) {
      m_device.memcpy((void*)dest, m_data, sizeof(Scalar) * m_dims.TotalSize());
      return false;
    }
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return m_data[index];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(Index index) {
    eigen_assert(m_data);
    return m_data[index];
  }

  template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt<PacketReturnType, LoadMode>(m_data + index);
  }

  template <int StoreMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void writePacket(Index index, const PacketReturnType& x)
  {
    return internal::pstoret<Scalar, PacketReturnType, StoreMode>(m_data + index, x);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(const array<DenseIndex, NumCoords>& coords) const {
    eigen_assert(m_data);
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return m_data[m_dims.IndexOfColMajor(coords)];
    } else {
      return m_data[m_dims.IndexOfRowMajor(coords)];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar& coeffRef(const array<DenseIndex, NumCoords>& coords) {
    eigen_assert(m_data);
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      return m_data[m_dims.IndexOfColMajor(coords)];
    } else {
      return m_data[m_dims.IndexOfRowMajor(coords)];
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized,
                        internal::unpacket_traits<PacketReturnType>::size);
  }

  EIGEN_DEVICE_FUNC typename internal::traits<Derived>::template MakePointer<Scalar>::Type data() const { return m_data; }

  /// required by sycl in order to construct sycl buffer from raw pointer
  const Device& device() const{return m_device;}

 protected:
  typename internal::traits<Derived>::template MakePointer<Scalar>::Type m_data;
  Dimensions m_dims;
  const Device& m_device;
  const Derived& m_impl;
};

namespace {
template <typename T> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T loadConstant(const T* address) {
  return *address;
}
// Use the texture cache on CUDA devices whenever possible
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float loadConstant(const float* address) {
  return __ldg(address);
}
template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double loadConstant(const double* address) {
  return __ldg(address);
}
template <> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
Eigen::half loadConstant(const Eigen::half* address) {
  return Eigen::half(half_impl::raw_uint16_to_half(__ldg(&address->x)));
}
#endif
}


// Default evaluator for rvalues
template<typename Derived, typename Device>
struct TensorEvaluator<const Derived, Device>
{
  typedef typename Derived::Index Index;
  typedef typename Derived::Scalar Scalar;
  typedef typename Derived::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  typedef typename Derived::Dimensions Dimensions;

  // NumDimensions is -1 for variable dim tensors
  static const int NumCoords = internal::traits<Derived>::NumDimensions > 0 ?
                               internal::traits<Derived>::NumDimensions : 0;

  enum {
    IsAligned = Derived::IsAligned,
    PacketAccess = (internal::unpacket_traits<PacketReturnType>::size > 1),
    Layout = Derived::Layout,
    CoordAccess = NumCoords > 0,
    RawAccess = true
  };

  // Used for accessor extraction in SYCL Managed TensorMap:
  const Derived& derived() const { return m_impl; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorEvaluator(const Derived& m, const Device& device)
      : m_data(m.data()), m_dims(m.dimensions()), m_device(device), m_impl(m)
  { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dims; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType* data) {
    if (!NumTraits<typename internal::remove_const<Scalar>::type>::RequireInitialization && data) {
      m_device.memcpy((void*)data, m_data, m_dims.TotalSize() * sizeof(Scalar));
      return false;
    }
    return true;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    eigen_assert(m_data);
    return loadConstant(m_data+index);
  }

  template<int LoadMode> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  PacketReturnType packet(Index index) const
  {
    return internal::ploadt_ro<PacketReturnType, LoadMode>(m_data + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(const array<DenseIndex, NumCoords>& coords) const {
    eigen_assert(m_data);
    const Index index = (static_cast<int>(Layout) == static_cast<int>(ColMajor)) ? m_dims.IndexOfColMajor(coords)
                        : m_dims.IndexOfRowMajor(coords);
    return loadConstant(m_data+index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized,
                        internal::unpacket_traits<PacketReturnType>::size);
  }

  EIGEN_DEVICE_FUNC typename internal::traits<Derived>::template MakePointer<const Scalar>::Type data() const { return m_data; }

  /// added for sycl in order to construct the buffer from the sycl device
  const Device& device() const{return m_device;}

 protected:
  typename internal::traits<Derived>::template MakePointer<const Scalar>::Type m_data;
  Dimensions m_dims;
  const Device& m_device;
  const Derived& m_impl;
};




// -------------------- CwiseNullaryOp --------------------

template<typename NullaryOp, typename ArgType, typename Device>
struct TensorEvaluator<const TensorCwiseNullaryOp<NullaryOp, ArgType>, Device>
{
  typedef TensorCwiseNullaryOp<NullaryOp, ArgType> XprType;

  enum {
    IsAligned = true,
    PacketAccess = internal::functor_traits<NullaryOp>::PacketAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };

  EIGEN_DEVICE_FUNC
  TensorEvaluator(const XprType& op, const Device& device)
      : m_functor(op.functor()), m_argImpl(op.nestedExpression(), device), m_wrapper()
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_argImpl.dimensions(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType*) { return true; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() { }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_wrapper(m_functor, index);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_wrapper.template packetOp<PacketReturnType, Index>(m_functor, index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool vectorized) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0, vectorized,
                        internal::unpacket_traits<PacketReturnType>::size);
  }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return NULL; }

  /// required by sycl in order to extract the accessor
  const TensorEvaluator<ArgType, Device>& impl() const { return m_argImpl; }
  /// required by sycl in order to extract the accessor
  NullaryOp functor() const { return m_functor; }


 private:
  const NullaryOp m_functor;
  TensorEvaluator<ArgType, Device> m_argImpl;
  const internal::nullary_wrapper<CoeffReturnType,NullaryOp> m_wrapper;
};



// -------------------- CwiseUnaryOp --------------------

template<typename UnaryOp, typename ArgType, typename Device>
struct TensorEvaluator<const TensorCwiseUnaryOp<UnaryOp, ArgType>, Device>
{
  typedef TensorCwiseUnaryOp<UnaryOp, ArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<ArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ArgType, Device>::PacketAccess & internal::functor_traits<UnaryOp>::PacketAccess,
    Layout = TensorEvaluator<ArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_functor(op.functor()),
      m_argImpl(op.nestedExpression(), device)
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;
  typedef typename TensorEvaluator<ArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const { return m_argImpl.dimensions(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar*) {
    m_argImpl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_argImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_argImpl.coeff(index));
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_argImpl.template packet<LoadMode>(index));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool vectorized) const {
    const double functor_cost = internal::functor_traits<UnaryOp>::Cost;
    return m_argImpl.costPerCoeff(vectorized) +
        TensorOpCost(0, 0, functor_cost, vectorized, PacketSize);
  }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return NULL; }

  /// required by sycl in order to extract the accessor
  const TensorEvaluator<ArgType, Device> & impl() const { return m_argImpl; }
  /// added for sycl in order to construct the buffer from sycl device
  UnaryOp functor() const { return m_functor; }


 private:
  const UnaryOp m_functor;
  TensorEvaluator<ArgType, Device> m_argImpl;
};


// -------------------- CwiseBinaryOp --------------------

template<typename BinaryOp, typename LeftArgType, typename RightArgType, typename Device>
struct TensorEvaluator<const TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType>, Device>
{
  typedef TensorCwiseBinaryOp<BinaryOp, LeftArgType, RightArgType> XprType;

  enum {
    IsAligned = TensorEvaluator<LeftArgType, Device>::IsAligned & TensorEvaluator<RightArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<LeftArgType, Device>::PacketAccess & TensorEvaluator<RightArgType, Device>::PacketAccess &
                   internal::functor_traits<BinaryOp>::PacketAccess,
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_functor(op.functor()),
      m_leftImpl(op.lhsExpression(), device),
      m_rightImpl(op.rhsExpression(), device)
  {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<LeftArgType, Device>::Layout) == static_cast<int>(TensorEvaluator<RightArgType, Device>::Layout) || internal::traits<XprType>::NumDimensions <= 1), YOU_MADE_A_PROGRAMMING_MISTAKE);
    eigen_assert(dimensions_match(m_leftImpl.dimensions(), m_rightImpl.dimensions()));
  }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;
  typedef typename TensorEvaluator<LeftArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const
  {
    // TODO: use right impl instead if right impl dimensions are known at compile time.
    return m_leftImpl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType*) {
    m_leftImpl.evalSubExprsIfNeeded(NULL);
    m_rightImpl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_leftImpl.cleanup();
    m_rightImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_leftImpl.coeff(index), m_rightImpl.coeff(index));
  }
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_leftImpl.template packet<LoadMode>(index), m_rightImpl.template packet<LoadMode>(index));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool vectorized) const {
    const double functor_cost = internal::functor_traits<BinaryOp>::Cost;
    return m_leftImpl.costPerCoeff(vectorized) +
           m_rightImpl.costPerCoeff(vectorized) +
           TensorOpCost(0, 0, functor_cost, vectorized, PacketSize);
  }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return NULL; }
  /// required by sycl in order to extract the accessor
  const TensorEvaluator<LeftArgType, Device>& left_impl() const { return m_leftImpl; }
  /// required by sycl in order to extract the accessor
  const TensorEvaluator<RightArgType, Device>& right_impl() const { return m_rightImpl; }
  /// required by sycl in order to extract the accessor
  BinaryOp functor() const { return m_functor; }

 private:
  const BinaryOp m_functor;
  TensorEvaluator<LeftArgType, Device> m_leftImpl;
  TensorEvaluator<RightArgType, Device> m_rightImpl;
};

// -------------------- CwiseTernaryOp --------------------

template<typename TernaryOp, typename Arg1Type, typename Arg2Type, typename Arg3Type, typename Device>
struct TensorEvaluator<const TensorCwiseTernaryOp<TernaryOp, Arg1Type, Arg2Type, Arg3Type>, Device>
{
  typedef TensorCwiseTernaryOp<TernaryOp, Arg1Type, Arg2Type, Arg3Type> XprType;

  enum {
    IsAligned = TensorEvaluator<Arg1Type, Device>::IsAligned & TensorEvaluator<Arg2Type, Device>::IsAligned & TensorEvaluator<Arg3Type, Device>::IsAligned,
    PacketAccess = TensorEvaluator<Arg1Type, Device>::PacketAccess & TensorEvaluator<Arg2Type, Device>::PacketAccess & TensorEvaluator<Arg3Type, Device>::PacketAccess &
                   internal::functor_traits<TernaryOp>::PacketAccess,
    Layout = TensorEvaluator<Arg1Type, Device>::Layout,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_functor(op.functor()),
      m_arg1Impl(op.arg1Expression(), device),
      m_arg2Impl(op.arg2Expression(), device),
      m_arg3Impl(op.arg3Expression(), device)
  {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<Arg1Type, Device>::Layout) == static_cast<int>(TensorEvaluator<Arg3Type, Device>::Layout) || internal::traits<XprType>::NumDimensions <= 1), YOU_MADE_A_PROGRAMMING_MISTAKE);

    EIGEN_STATIC_ASSERT((internal::is_same<typename internal::traits<Arg1Type>::StorageKind,
                         typename internal::traits<Arg2Type>::StorageKind>::value),
                        STORAGE_KIND_MUST_MATCH)
    EIGEN_STATIC_ASSERT((internal::is_same<typename internal::traits<Arg1Type>::StorageKind,
                         typename internal::traits<Arg3Type>::StorageKind>::value),
                        STORAGE_KIND_MUST_MATCH)
    EIGEN_STATIC_ASSERT((internal::is_same<typename internal::traits<Arg1Type>::Index,
                         typename internal::traits<Arg2Type>::Index>::value),
                        STORAGE_INDEX_MUST_MATCH)
    EIGEN_STATIC_ASSERT((internal::is_same<typename internal::traits<Arg1Type>::Index,
                         typename internal::traits<Arg3Type>::Index>::value),
                        STORAGE_INDEX_MUST_MATCH)

    eigen_assert(dimensions_match(m_arg1Impl.dimensions(), m_arg2Impl.dimensions()) && dimensions_match(m_arg1Impl.dimensions(), m_arg3Impl.dimensions()));
  }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;
  typedef typename TensorEvaluator<Arg1Type, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const
  {
    // TODO: use arg2 or arg3 dimensions if they are known at compile time.
    return m_arg1Impl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType*) {
    m_arg1Impl.evalSubExprsIfNeeded(NULL);
    m_arg2Impl.evalSubExprsIfNeeded(NULL);
    m_arg3Impl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_arg1Impl.cleanup();
    m_arg2Impl.cleanup();
    m_arg3Impl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_functor(m_arg1Impl.coeff(index), m_arg2Impl.coeff(index), m_arg3Impl.coeff(index));
  }
  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const
  {
    return m_functor.packetOp(m_arg1Impl.template packet<LoadMode>(index),
                              m_arg2Impl.template packet<LoadMode>(index),
                              m_arg3Impl.template packet<LoadMode>(index));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool vectorized) const {
    const double functor_cost = internal::functor_traits<TernaryOp>::Cost;
    return m_arg1Impl.costPerCoeff(vectorized) +
           m_arg2Impl.costPerCoeff(vectorized) +
           m_arg3Impl.costPerCoeff(vectorized) +
           TensorOpCost(0, 0, functor_cost, vectorized, PacketSize);
  }

  EIGEN_DEVICE_FUNC CoeffReturnType* data() const { return NULL; }

  /// required by sycl in order to extract the accessor
  const TensorEvaluator<Arg1Type, Device> & arg1Impl() const { return m_arg1Impl; }
  /// required by sycl in order to extract the accessor
  const TensorEvaluator<Arg2Type, Device>& arg2Impl() const { return m_arg2Impl; }
  /// required by sycl in order to extract the accessor
  const TensorEvaluator<Arg3Type, Device>& arg3Impl() const { return m_arg3Impl; }

 private:
  const TernaryOp m_functor;
  TensorEvaluator<Arg1Type, Device> m_arg1Impl;
  TensorEvaluator<Arg2Type, Device> m_arg2Impl;
  TensorEvaluator<Arg3Type, Device> m_arg3Impl;
};


// -------------------- SelectOp --------------------

template<typename IfArgType, typename ThenArgType, typename ElseArgType, typename Device>
struct TensorEvaluator<const TensorSelectOp<IfArgType, ThenArgType, ElseArgType>, Device>
{
  typedef TensorSelectOp<IfArgType, ThenArgType, ElseArgType> XprType;
  typedef typename XprType::Scalar Scalar;

  enum {
    IsAligned = TensorEvaluator<ThenArgType, Device>::IsAligned & TensorEvaluator<ElseArgType, Device>::IsAligned,
    PacketAccess = TensorEvaluator<ThenArgType, Device>::PacketAccess & TensorEvaluator<ElseArgType, Device>::PacketAccess &
                   internal::packet_traits<Scalar>::HasBlend,
    Layout = TensorEvaluator<IfArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
    RawAccess = false
  };

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device)
    : m_condImpl(op.ifExpression(), device),
      m_thenImpl(op.thenExpression(), device),
      m_elseImpl(op.elseExpression(), device)
  {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<IfArgType, Device>::Layout) == static_cast<int>(TensorEvaluator<ThenArgType, Device>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<IfArgType, Device>::Layout) == static_cast<int>(TensorEvaluator<ElseArgType, Device>::Layout)), YOU_MADE_A_PROGRAMMING_MISTAKE);
    eigen_assert(dimensions_match(m_condImpl.dimensions(), m_thenImpl.dimensions()));
    eigen_assert(dimensions_match(m_thenImpl.dimensions(), m_elseImpl.dimensions()));
  }

  typedef typename XprType::Index Index;
  typedef typename internal::traits<XprType>::Scalar CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;
  static const int PacketSize = internal::unpacket_traits<PacketReturnType>::size;
  typedef typename TensorEvaluator<IfArgType, Device>::Dimensions Dimensions;

  EIGEN_DEVICE_FUNC const Dimensions& dimensions() const
  {
    // TODO: use then or else impl instead if they happen to be known at compile time.
    return m_condImpl.dimensions();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(CoeffReturnType*) {
    m_condImpl.evalSubExprsIfNeeded(NULL);
    m_thenImpl.evalSubExprsIfNeeded(NULL);
    m_elseImpl.evalSubExprsIfNeeded(NULL);
    return true;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_condImpl.cleanup();
    m_thenImpl.cleanup();
    m_elseImpl.cleanup();
  }

  EIGEN_DEVICE_FUNC CoeffReturnType coeff(Index index) const
  {
    return m_condImpl.coeff(index) ? m_thenImpl.coeff(index) : m_elseImpl.coeff(index);
  }
  template<int LoadMode>
  EIGEN_DEVICE_FUNC PacketReturnType packet(Index index) const
  {
    internal::Selector<PacketSize> select;
    for (Index i = 0; i < PacketSize; ++i) {
      select.select[i] = m_condImpl.coeff(index+i);
    }
    return internal::pblend(select,
                            m_thenImpl.template packet<LoadMode>(index),
                            m_elseImpl.template packet<LoadMode>(index));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost
  costPerCoeff(bool vectorized) const {
    return m_condImpl.costPerCoeff(vectorized) +
           m_thenImpl.costPerCoeff(vectorized)
        .cwiseMax(m_elseImpl.costPerCoeff(vectorized));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType* data() const { return NULL; }
  /// required by sycl in order to extract the accessor
  const TensorEvaluator<IfArgType, Device> & cond_impl() const { return m_condImpl; }
  /// required by sycl in order to extract the accessor
  const TensorEvaluator<ThenArgType, Device>& then_impl() const { return m_thenImpl; }
  /// required by sycl in order to extract the accessor
  const TensorEvaluator<ElseArgType, Device>& else_impl() const { return m_elseImpl; }

 private:
  TensorEvaluator<IfArgType, Device> m_condImpl;
  TensorEvaluator<ThenArgType, Device> m_thenImpl;
  TensorEvaluator<ElseArgType, Device> m_elseImpl;
};


} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EVALUATOR_H
