// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H
#define EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H

namespace Eigen {

/** \class TensorContraction
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor contraction class.
  *
  *
  */
namespace internal {

template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct traits<TensorContractionOp<Dimensions, LhsXprType, RhsXprType> >
{
  // Type promotion to handle the case where the types of the lhs and the rhs are different.
  typedef typename gebp_traits<typename remove_const<typename LhsXprType::Scalar>::type,
                               typename remove_const<typename RhsXprType::Scalar>::type>::ResScalar Scalar;

  typedef typename promote_storage_type<typename traits<LhsXprType>::StorageKind,
                                        typename traits<RhsXprType>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<LhsXprType>::Index,
                                      typename traits<RhsXprType>::Index>::type Index;
  typedef typename LhsXprType::Nested LhsNested;
  typedef typename RhsXprType::Nested RhsNested;
  typedef typename remove_reference<LhsNested>::type _LhsNested;
  typedef typename remove_reference<RhsNested>::type _RhsNested;

  // From NumDims below.
  static const int NumDimensions = traits<RhsXprType>::NumDimensions + traits<RhsXprType>::NumDimensions - 2 * array_size<Dimensions>::value;
  static const int Layout = traits<LhsXprType>::Layout;

  enum {
    Flags = 0
  };
};

template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct eval<TensorContractionOp<Dimensions, LhsXprType, RhsXprType>, Eigen::Dense>
{
  typedef const TensorContractionOp<Dimensions, LhsXprType, RhsXprType>& type;
};

template<typename Dimensions, typename LhsXprType, typename RhsXprType>
struct nested<TensorContractionOp<Dimensions, LhsXprType, RhsXprType>, 1, typename eval<TensorContractionOp<Dimensions, LhsXprType, RhsXprType> >::type>
{
  typedef TensorContractionOp<Dimensions, LhsXprType, RhsXprType> type;
};

template<typename Indices_, typename LeftArgType_, typename RightArgType_, typename Device_>
struct traits<TensorEvaluator<const TensorContractionOp<Indices_, LeftArgType_, RightArgType_>, Device_> > {
  typedef Indices_ Indices;
  typedef LeftArgType_ LeftArgType;
  typedef RightArgType_ RightArgType;
  typedef Device_ Device;

  // From NumDims below.
  static const int NumDimensions = traits<LeftArgType_>::NumDimensions + traits<RightArgType_>::NumDimensions - 2 * array_size<Indices_>::value;
};

}  // end namespace internal

template<typename Indices, typename LhsXprType, typename RhsXprType>
class TensorContractionOp : public TensorBase<TensorContractionOp<Indices, LhsXprType, RhsXprType>, ReadOnlyAccessors>
{
  public:
  typedef typename Eigen::internal::traits<TensorContractionOp>::Scalar Scalar;
  typedef typename internal::gebp_traits<typename LhsXprType::CoeffReturnType,
                                                   typename RhsXprType::CoeffReturnType>::ResScalar CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorContractionOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorContractionOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorContractionOp>::Index Index;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorContractionOp(
      const LhsXprType& lhs, const RhsXprType& rhs, const Indices& dims)
      : m_lhs_xpr(lhs), m_rhs_xpr(rhs), m_indices(dims) {}

  EIGEN_DEVICE_FUNC
  const Indices& indices() const { return m_indices; }

  /** \returns the nested expressions */
  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename LhsXprType::Nested>::type&
  lhsExpression() const { return m_lhs_xpr; }

  EIGEN_DEVICE_FUNC
  const typename internal::remove_all<typename RhsXprType::Nested>::type&
  rhsExpression() const { return m_rhs_xpr; }

  protected:
    typename LhsXprType::Nested m_lhs_xpr;
    typename RhsXprType::Nested m_rhs_xpr;
    const Indices m_indices;
};


template<typename Derived>
struct TensorContractionEvaluatorBase
{
  typedef typename internal::traits<Derived>::Indices Indices;
  typedef typename internal::traits<Derived>::LeftArgType LeftArgType;
  typedef typename internal::traits<Derived>::RightArgType RightArgType;
  typedef typename internal::traits<Derived>::Device Device;

  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;

  enum {
    IsAligned = true,
    PacketAccess = (internal::unpacket_traits<PacketReturnType>::size > 1),
    Layout = TensorEvaluator<LeftArgType, Device>::Layout,
    CoordAccess = false,  // to be implemented
    RawAccess = true
  };

  // Most of the code is assuming that both input tensors are ColMajor. If the
  // inputs are RowMajor, we will "cheat" by swapping the LHS and RHS:
  // If we want to compute A * B = C, where A is LHS and B is RHS, the code
  // will pretend B is LHS and A is RHS.
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType>::type EvalLeftArgType;
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType>::type EvalRightArgType;

  static const int LDims =
      internal::array_size<typename TensorEvaluator<EvalLeftArgType, Device>::Dimensions>::value;
  static const int RDims =
      internal::array_size<typename TensorEvaluator<EvalRightArgType, Device>::Dimensions>::value;
  static const int ContractDims = internal::array_size<Indices>::value;
  static const int NumDims = LDims + RDims - 2 * ContractDims;

  typedef array<Index, ContractDims> contract_t;
  typedef array<Index, LDims - ContractDims> left_nocontract_t;
  typedef array<Index, RDims - ContractDims> right_nocontract_t;

  typedef DSizes<Index, NumDims> Dimensions;

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  TensorContractionEvaluatorBase(const XprType& op, const Device& device)
    : m_leftImpl(choose(Cond<static_cast<int>(Layout) == static_cast<int>(ColMajor)>(),
                          op.lhsExpression(), op.rhsExpression()), device),
    m_rightImpl(choose(Cond<static_cast<int>(Layout) == static_cast<int>(ColMajor)>(),
                          op.rhsExpression(), op.lhsExpression()), device),
        m_device(device),
        m_result(NULL) {
    EIGEN_STATIC_ASSERT((static_cast<int>(TensorEvaluator<LeftArgType, Device>::Layout) ==
			   static_cast<int>(TensorEvaluator<RightArgType, Device>::Layout)),
                        YOU_MADE_A_PROGRAMMING_MISTAKE);


    DSizes<Index, LDims> eval_left_dims;
    DSizes<Index, RDims> eval_right_dims;
    array<IndexPair<Index>, ContractDims> eval_op_indices;
    if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
      // For ColMajor, we keep using the existing dimensions
      for (int i = 0; i < LDims; i++) {
        eval_left_dims[i] = m_leftImpl.dimensions()[i];
      }
      for (int i = 0; i < RDims; i++) {
        eval_right_dims[i] = m_rightImpl.dimensions()[i];
      }
      // We keep the pairs of contracting indices.
      for (int i = 0; i < ContractDims; i++) {
        eval_op_indices[i].first = op.indices()[i].first;
        eval_op_indices[i].second = op.indices()[i].second;
      }
    } else {
      // For RowMajor, we need to reverse the existing dimensions
      for (int i = 0; i < LDims; i++) {
        eval_left_dims[i] = m_leftImpl.dimensions()[LDims - i - 1];
      }
      for (int i = 0; i < RDims; i++) {
        eval_right_dims[i] = m_rightImpl.dimensions()[RDims - i - 1];
      }
      // We need to flip all the pairs of contracting indices as well as
      // reversing the dimensions.
      for (int i = 0; i < ContractDims; i++) {
        eval_op_indices[i].first = LDims - 1 - op.indices()[ContractDims - 1 - i].second;
        eval_op_indices[i].second = RDims - 1 - op.indices()[ContractDims - 1 - i].first;
      }
    }

    // Check for duplicate axes and make sure the first index in eval_op_indices
    // is increasing. Using O(n^2) sorting is OK since ContractDims is small
    for (int i = 0; i < ContractDims; i++) {
      for (int j = i + 1; j < ContractDims; j++) {
        eigen_assert(eval_op_indices[j].first != eval_op_indices[i].first &&
                     eval_op_indices[j].second != eval_op_indices[i].second &&
                     "contraction axes should be unique");
        if (eval_op_indices[j].first < eval_op_indices[i].first) {
          numext::swap(eval_op_indices[j], eval_op_indices[i]);
        }
      }
    }

    array<Index, LDims> lhs_strides;
    lhs_strides[0] = 1;
    for (int i = 0; i < LDims-1; ++i) {
      lhs_strides[i+1] = lhs_strides[i] * eval_left_dims[i];
    }

    array<Index, RDims> rhs_strides;
    rhs_strides[0] = 1;
    for (int i = 0; i < RDims-1; ++i) {
      rhs_strides[i+1] = rhs_strides[i] * eval_right_dims[i];
    }

    if (m_i_strides.size() > 0) m_i_strides[0] = 1;
    if (m_j_strides.size() > 0) m_j_strides[0] = 1;
    if (m_k_strides.size() > 0) m_k_strides[0] = 1;

    m_i_size = 1;
    m_j_size = 1;
    m_k_size = 1;

    // To compute the dimension, we simply concatenate the non-contracting
    // dimensions of the left and then the right tensor. Additionally, we also
    // compute the strides corresponding to the left non-contracting
    // dimensions and right non-contracting dimensions.
    m_lhs_inner_dim_contiguous = true;
    int dim_idx = 0;
    unsigned int nocontract_idx = 0;

    for (int i = 0; i < LDims; i++) {
      // find if we are contracting on index i of left tensor
      bool contracting = false;
      for (int j = 0; j < ContractDims; j++) {
        if (eval_op_indices[j].first == i) {
          contracting = true;
          break;
        }
      }
      if (!contracting) {
        // add dimension size to output dimensions
        m_dimensions[dim_idx] = eval_left_dims[i];
        m_left_nocontract_strides[nocontract_idx] = lhs_strides[i];
        if (dim_idx != i) {
          m_lhs_inner_dim_contiguous = false;
        }
        if (nocontract_idx+1 < internal::array_size<left_nocontract_t>::value) {
          m_i_strides[nocontract_idx+1] =
              m_i_strides[nocontract_idx] * eval_left_dims[i];
        } else {
          m_i_size = m_i_strides[nocontract_idx] * eval_left_dims[i];
        }
        dim_idx++;
        nocontract_idx++;
      }
    }

    nocontract_idx = 0;
    for (int i = 0; i < RDims; i++) {
      bool contracting = false;
      // find if we are contracting on index i of right tensor
      for (int j = 0; j < ContractDims; j++) {
        if (eval_op_indices[j].second == i) {
          contracting = true;
          break;
        }
      }
      if (!contracting) {
        m_dimensions[dim_idx] = eval_right_dims[i];
        if (nocontract_idx+1 < internal::array_size<right_nocontract_t>::value) {
          m_j_strides[nocontract_idx+1] =
              m_j_strides[nocontract_idx] * eval_right_dims[i];
        } else {
          m_j_size = m_j_strides[nocontract_idx] * eval_right_dims[i];
        }
        m_right_nocontract_strides[nocontract_idx] = rhs_strides[i];
        dim_idx++;
        nocontract_idx++;
      }
    }

    // Now compute the strides corresponding to the contracting dimensions. We
    // assumed above that non-contracting axes are represented in the same order
    // in the matrix as they are in the tensor. This is not the case for
    // contracting axes. As the contracting axes must be of the same size in
    // each tensor, we'll only look at the first tensor here.
    m_rhs_inner_dim_contiguous = true;
    m_rhs_inner_dim_reordered = false;
    for (int i = 0; i < ContractDims; i++) {
      Index left = eval_op_indices[i].first;
      Index right = eval_op_indices[i].second;

      Index size = eval_left_dims[left];
      eigen_assert(size == eval_right_dims[right] &&
                   "Contraction axes must be same size");

      if (i+1 < static_cast<int>(internal::array_size<contract_t>::value)) {
        m_k_strides[i+1] = m_k_strides[i] * size;
      } else {
        m_k_size = m_k_strides[i] * size;
      }
      m_left_contracting_strides[i] = lhs_strides[left];
      m_right_contracting_strides[i] = rhs_strides[right];

      if (i > 0 && right < eval_op_indices[i-1].second) {
        m_rhs_inner_dim_reordered = true;
      }
      if (right != i) {
        m_rhs_inner_dim_contiguous = false;
      }
    }

    // If the layout is RowMajor, we need to reverse the m_dimensions
    if (static_cast<int>(Layout) == static_cast<int>(RowMajor)) {
      for (int i = 0, j = NumDims - 1; i < j; i++, j--) {
        numext::swap(m_dimensions[i], m_dimensions[j]);
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions& dimensions() const { return m_dimensions; }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool evalSubExprsIfNeeded(Scalar* data) {
    m_leftImpl.evalSubExprsIfNeeded(NULL);
    m_rightImpl.evalSubExprsIfNeeded(NULL);
    if (data) {
      evalTo(data);
      return false;
    } else {
      m_result = static_cast<Scalar *>(m_device.allocate(dimensions().TotalSize() * sizeof(Scalar)));
      evalTo(m_result);
      return true;
    }
  }

  EIGEN_DEVICE_FUNC void evalTo(Scalar* buffer) const {
    if (this->m_lhs_inner_dim_contiguous) {
      if (this->m_rhs_inner_dim_contiguous) {
        if (this->m_rhs_inner_dim_reordered) {
          static_cast<const Derived*>(this)->template evalProduct<true, true, true, Unaligned>(buffer);
        }
        else {
          static_cast<const Derived*>(this)->template evalProduct<true, true, false, Unaligned>(buffer);
        }
      }
      else {
       if (this->m_rhs_inner_dim_reordered) {
          static_cast<const Derived*>(this)->template evalProduct<true, false, true, Unaligned>(buffer);
        }
        else {
          static_cast<const Derived*>(this)->template evalProduct<true, false, false, Unaligned>(buffer);
        }
      }
    }
    else {
      if (this->m_rhs_inner_dim_contiguous) {
        if (this->m_rhs_inner_dim_reordered) {
          static_cast<const Derived*>(this)->template evalProduct<false, true, true, Unaligned>(buffer);
        }
        else {
          static_cast<const Derived*>(this)->template evalProduct<false, true, false, Unaligned>(buffer);
        }
      }
      else {
       if (this->m_rhs_inner_dim_reordered) {
          static_cast<const Derived*>(this)->template evalProduct<false, false, true, Unaligned>(buffer);
        }
        else {
          static_cast<const Derived*>(this)->template evalProduct<false, false, false, Unaligned>(buffer);
        }
      }
    }
  }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  EIGEN_DEVICE_FUNC void evalGemv(Scalar* buffer) const {
    const Index rows = m_i_size;
    const Index cols = m_k_size;

    typedef typename internal::remove_const<typename EvalLeftArgType::Scalar>::type LhsScalar;
    typedef typename internal::remove_const<typename EvalRightArgType::Scalar>::type RhsScalar;
    typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
    typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;
    const Index lhs_packet_size = internal::unpacket_traits<typename LeftEvaluator::PacketReturnType>::size;
    const Index rhs_packet_size = internal::unpacket_traits<typename RightEvaluator::PacketReturnType>::size;
    const int lhs_alignment = LeftEvaluator::IsAligned ? Aligned : Unaligned;
    const int rhs_alignment = RightEvaluator::IsAligned ? Aligned : Unaligned;
    typedef internal::TensorContractionInputMapper<LhsScalar, Index, internal::Lhs,
                                                   LeftEvaluator, left_nocontract_t,
                                                   contract_t, lhs_packet_size,
                                                   lhs_inner_dim_contiguous,
                                                   false, lhs_alignment> LhsMapper;

    typedef internal::TensorContractionInputMapper<RhsScalar, Index, internal::Rhs,
                                                   RightEvaluator, right_nocontract_t,
                                                   contract_t, rhs_packet_size,
                                                   rhs_inner_dim_contiguous,
                                                   rhs_inner_dim_reordered, rhs_alignment> RhsMapper;

    LhsMapper lhs(m_leftImpl, m_left_nocontract_strides, m_i_strides,
                  m_left_contracting_strides, m_k_strides);
    RhsMapper rhs(m_rightImpl, m_right_nocontract_strides, m_j_strides,
                  m_right_contracting_strides, m_k_strides);

    const Scalar alpha(1);
    const Index resIncr(1);

    // zero out the result buffer (which must be of size at least rows * sizeof(Scalar)
    m_device.memset(buffer, 0, rows * sizeof(Scalar));

    internal::general_matrix_vector_product<Index,LhsScalar,LhsMapper,ColMajor,false,RhsScalar,RhsMapper,false>::run(
        rows, cols, lhs, rhs,
        buffer, resIncr, alpha);
  }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  EIGEN_DEVICE_FUNC void evalGemm(Scalar* buffer) const {
    // columns in left side, rows in right side
    const Index k = this->m_k_size;

    // rows in left side
    const Index m = this->m_i_size;

    // columns in right side
    const Index n = this->m_j_size;

    // zero out the result buffer (which must be of size at least m * n * sizeof(Scalar)
    this->m_device.memset(buffer, 0, m * n * sizeof(Scalar));

    // define mr, nr, and all of my data mapper types
    typedef typename internal::remove_const<typename EvalLeftArgType::Scalar>::type LhsScalar;
    typedef typename internal::remove_const<typename EvalRightArgType::Scalar>::type RhsScalar;
    typedef typename internal::gebp_traits<LhsScalar, RhsScalar> Traits;

    const Index nr = Traits::nr;
    const Index mr = Traits::mr;

    typedef TensorEvaluator<EvalLeftArgType, Device> LeftEvaluator;
    typedef TensorEvaluator<EvalRightArgType, Device> RightEvaluator;

    const Index lhs_packet_size = internal::unpacket_traits<typename LeftEvaluator::PacketReturnType>::size;
    const Index rhs_packet_size = internal::unpacket_traits<typename RightEvaluator::PacketReturnType>::size;

    typedef internal::TensorContractionInputMapper<LhsScalar, Index, internal::Lhs,
                                                   LeftEvaluator, left_nocontract_t,
                                                   contract_t, lhs_packet_size,
                                                   lhs_inner_dim_contiguous,
                                                   false, Unaligned> LhsMapper;

    typedef internal::TensorContractionInputMapper<RhsScalar, Index, internal::Rhs,
                                                   RightEvaluator, right_nocontract_t,
                                                   contract_t, rhs_packet_size,
                                                   rhs_inner_dim_contiguous,
                                                   rhs_inner_dim_reordered, Unaligned> RhsMapper;

    typedef internal::blas_data_mapper<Scalar, Index, ColMajor> OutputMapper;

    // Declare GEBP packing and kernel structs
    internal::gemm_pack_lhs<LhsScalar, Index, typename LhsMapper::SubMapper, mr, Traits::LhsProgress, ColMajor> pack_lhs;
    internal::gemm_pack_rhs<RhsScalar, Index, typename RhsMapper::SubMapper, nr, ColMajor> pack_rhs;

    internal::gebp_kernel<LhsScalar, RhsScalar, Index, OutputMapper, mr, nr, false, false> gebp;

    // initialize data mappers
    LhsMapper lhs(this->m_leftImpl, this->m_left_nocontract_strides, this->m_i_strides,
                  this->m_left_contracting_strides, this->m_k_strides);

    RhsMapper rhs(this->m_rightImpl, this->m_right_nocontract_strides, this->m_j_strides,
                  this->m_right_contracting_strides, this->m_k_strides);

    OutputMapper output(buffer, m);

    // Sizes of the blocks to load in cache. See the Goto paper for details.
    internal::TensorContractionBlocking<LhsMapper, RhsMapper, Index, internal::ShardByCol> blocking(k, m, n, 1);
    const Index kc = blocking.kc();
    const Index mc = numext::mini(m, blocking.mc());
    const Index nc = numext::mini(n, blocking.nc());
    const Index sizeA = mc * kc;
    const Index sizeB = kc * nc;

    LhsScalar* blockA = static_cast<LhsScalar *>(this->m_device.allocate(sizeA * sizeof(LhsScalar)));
    RhsScalar* blockB = static_cast<RhsScalar *>(this->m_device.allocate(sizeB * sizeof(RhsScalar)));

    for(Index i2=0; i2<m; i2+=mc)
    {
      const Index actual_mc = numext::mini(i2+mc,m)-i2;
      for (Index k2 = 0; k2 < k; k2 += kc) {
        // make sure we don't overshoot right edge of left matrix, then pack vertical panel
        const Index actual_kc = numext::mini(k2 + kc, k) - k2;
        pack_lhs(blockA, lhs.getSubMapper(i2, k2), actual_kc, actual_mc, 0, 0);

        // series of horizontal blocks
        for (Index j2 = 0; j2 < n; j2 += nc) {
          // make sure we don't overshoot right edge of right matrix, then pack block
          const Index actual_nc = numext::mini(j2 + nc, n) - j2;
          pack_rhs(blockB, rhs.getSubMapper(k2, j2), actual_kc, actual_nc, 0, 0);

          // call gebp (matrix kernel)
          // The parameters here are copied from Eigen's GEMM implementation
          gebp(output.getSubMapper(i2, j2), blockA, blockB, actual_mc, actual_kc, actual_nc, Scalar(1), -1, -1, 0, 0);
        }
      }
    }

    this->m_device.deallocate(blockA);
    this->m_device.deallocate(blockB);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void cleanup() {
    m_leftImpl.cleanup();
    m_rightImpl.cleanup();

    if (m_result != NULL) {
      m_device.deallocate(m_result);
      m_result = NULL;
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CoeffReturnType coeff(Index index) const {
    return m_result[index];
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost costPerCoeff(bool) const {
    return TensorOpCost(sizeof(CoeffReturnType), 0, 0);
  }

  template<int LoadMode>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE PacketReturnType packet(Index index) const {
    return internal::ploadt<PacketReturnType, LoadMode>(m_result + index);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar* data() const { return m_result; }

  protected:
  // Prevent assignment
  TensorContractionEvaluatorBase& operator = (const TensorContractionEvaluatorBase&);
  Dimensions m_dimensions;

  contract_t m_k_strides;
  contract_t m_left_contracting_strides;
  contract_t m_right_contracting_strides;

  bool m_lhs_inner_dim_contiguous;
  bool m_rhs_inner_dim_contiguous;
  bool m_rhs_inner_dim_reordered;

  left_nocontract_t m_i_strides;
  right_nocontract_t m_j_strides;
  left_nocontract_t m_left_nocontract_strides;
  right_nocontract_t m_right_nocontract_strides;

  Index m_i_size;
  Index m_j_size;
  Index m_k_size;

  TensorEvaluator<EvalLeftArgType, Device> m_leftImpl;
  TensorEvaluator<EvalRightArgType, Device> m_rightImpl;
  const Device& m_device;
  Scalar* m_result;
};


// evaluator for default device
template<typename Indices, typename LeftArgType, typename RightArgType, typename Device>
struct TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> :
    public TensorContractionEvaluatorBase<
      TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> > {
  typedef TensorEvaluator<const TensorContractionOp<Indices, LeftArgType, RightArgType>, Device> Self;
  typedef TensorContractionEvaluatorBase<Self> Base;

  typedef TensorContractionOp<Indices, LeftArgType, RightArgType> XprType;
  typedef typename internal::remove_const<typename XprType::Scalar>::type Scalar;
  typedef typename XprType::Index Index;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename PacketType<CoeffReturnType, Device>::type PacketReturnType;

  enum {
    Layout = TensorEvaluator<LeftArgType, Device>::Layout
  };

  // Most of the code is assuming that both input tensors are ColMajor. If the
  // inputs are RowMajor, we will "cheat" by swapping the LHS and RHS:
  // If we want to compute A * B = C, where A is LHS and B is RHS, the code
  // will pretend B is LHS and A is RHS.
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), LeftArgType, RightArgType>::type EvalLeftArgType;
  typedef typename internal::conditional<
    static_cast<int>(Layout) == static_cast<int>(ColMajor), RightArgType, LeftArgType>::type EvalRightArgType;

  static const int LDims =
      internal::array_size<typename TensorEvaluator<EvalLeftArgType, Device>::Dimensions>::value;
  static const int RDims =
      internal::array_size<typename TensorEvaluator<EvalRightArgType, Device>::Dimensions>::value;
  static const int ContractDims = internal::array_size<Indices>::value;

  typedef array<Index, ContractDims> contract_t;
  typedef array<Index, LDims - ContractDims> left_nocontract_t;
  typedef array<Index, RDims - ContractDims> right_nocontract_t;

  static const int NumDims = LDims + RDims - 2 * ContractDims;

  // Could we use NumDimensions here?
  typedef DSizes<Index, NumDims> Dimensions;

  EIGEN_DEVICE_FUNC TensorEvaluator(const XprType& op, const Device& device) :
      Base(op, device) { }

  template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment>
  EIGEN_DEVICE_FUNC void evalProduct(Scalar* buffer) const {
    if (this->m_j_size == 1) {
      this->template evalGemv<lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Alignment>(buffer);
      return;
    }

    this->template evalGemm<lhs_inner_dim_contiguous, rhs_inner_dim_contiguous, rhs_inner_dim_reordered, Alignment>(buffer);
  }
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_CONTRACTION_H
