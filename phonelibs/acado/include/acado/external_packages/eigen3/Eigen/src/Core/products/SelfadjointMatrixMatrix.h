// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SELFADJOINT_MATRIX_MATRIX_H
#define EIGEN_SELFADJOINT_MATRIX_MATRIX_H

namespace Eigen { 

namespace internal {

// pack a selfadjoint block diagonal for use with the gebp_kernel
template<typename Scalar, typename Index, int Pack1, int Pack2, int StorageOrder>
struct symm_pack_lhs
{
  template<int BlockRows> inline
  void pack(Scalar* blockA, const const_blas_data_mapper<Scalar,Index,StorageOrder>& lhs, Index cols, Index i, Index& count)
  {
    // normal copy
    for(Index k=0; k<i; k++)
      for(Index w=0; w<BlockRows; w++)
        blockA[count++] = lhs(i+w,k);           // normal
    // symmetric copy
    Index h = 0;
    for(Index k=i; k<i+BlockRows; k++)
    {
      for(Index w=0; w<h; w++)
        blockA[count++] = numext::conj(lhs(k, i+w)); // transposed

      blockA[count++] = numext::real(lhs(k,k));   // real (diagonal)

      for(Index w=h+1; w<BlockRows; w++)
        blockA[count++] = lhs(i+w, k);          // normal
      ++h;
    }
    // transposed copy
    for(Index k=i+BlockRows; k<cols; k++)
      for(Index w=0; w<BlockRows; w++)
        blockA[count++] = numext::conj(lhs(k, i+w)); // transposed
  }
  void operator()(Scalar* blockA, const Scalar* _lhs, Index lhsStride, Index cols, Index rows)
  {
    const_blas_data_mapper<Scalar,Index,StorageOrder> lhs(_lhs,lhsStride);
    Index count = 0;
    Index peeled_mc = (rows/Pack1)*Pack1;
    for(Index i=0; i<peeled_mc; i+=Pack1)
    {
      pack<Pack1>(blockA, lhs, cols, i, count);
    }

    if(rows-peeled_mc>=Pack2)
    {
      pack<Pack2>(blockA, lhs, cols, peeled_mc, count);
      peeled_mc += Pack2;
    }

    // do the same with mr==1
    for(Index i=peeled_mc; i<rows; i++)
    {
      for(Index k=0; k<i; k++)
        blockA[count++] = lhs(i, k);              // normal

      blockA[count++] = numext::real(lhs(i, i));       // real (diagonal)

      for(Index k=i+1; k<cols; k++)
        blockA[count++] = numext::conj(lhs(k, i));     // transposed
    }
  }
};

template<typename Scalar, typename Index, int nr, int StorageOrder>
struct symm_pack_rhs
{
  enum { PacketSize = packet_traits<Scalar>::size };
  void operator()(Scalar* blockB, const Scalar* _rhs, Index rhsStride, Index rows, Index cols, Index k2)
  {
    Index end_k = k2 + rows;
    Index count = 0;
    const_blas_data_mapper<Scalar,Index,StorageOrder> rhs(_rhs,rhsStride);
    Index packet_cols = (cols/nr)*nr;

    // first part: normal case
    for(Index j2=0; j2<k2; j2+=nr)
    {
      for(Index k=k2; k<end_k; k++)
      {
        blockB[count+0] = rhs(k,j2+0);
        blockB[count+1] = rhs(k,j2+1);
        if (nr==4)
        {
          blockB[count+2] = rhs(k,j2+2);
          blockB[count+3] = rhs(k,j2+3);
        }
        count += nr;
      }
    }

    // second part: diagonal block
    for(Index j2=k2; j2<(std::min)(k2+rows,packet_cols); j2+=nr)
    {
      // again we can split vertically in three different parts (transpose, symmetric, normal)
      // transpose
      for(Index k=k2; k<j2; k++)
      {
        blockB[count+0] = numext::conj(rhs(j2+0,k));
        blockB[count+1] = numext::conj(rhs(j2+1,k));
        if (nr==4)
        {
          blockB[count+2] = numext::conj(rhs(j2+2,k));
          blockB[count+3] = numext::conj(rhs(j2+3,k));
        }
        count += nr;
      }
      // symmetric
      Index h = 0;
      for(Index k=j2; k<j2+nr; k++)
      {
        // normal
        for (Index w=0 ; w<h; ++w)
          blockB[count+w] = rhs(k,j2+w);

        blockB[count+h] = numext::real(rhs(k,k));

        // transpose
        for (Index w=h+1 ; w<nr; ++w)
          blockB[count+w] = numext::conj(rhs(j2+w,k));
        count += nr;
        ++h;
      }
      // normal
      for(Index k=j2+nr; k<end_k; k++)
      {
        blockB[count+0] = rhs(k,j2+0);
        blockB[count+1] = rhs(k,j2+1);
        if (nr==4)
        {
          blockB[count+2] = rhs(k,j2+2);
          blockB[count+3] = rhs(k,j2+3);
        }
        count += nr;
      }
    }

    // third part: transposed
    for(Index j2=k2+rows; j2<packet_cols; j2+=nr)
    {
      for(Index k=k2; k<end_k; k++)
      {
        blockB[count+0] = numext::conj(rhs(j2+0,k));
        blockB[count+1] = numext::conj(rhs(j2+1,k));
        if (nr==4)
        {
          blockB[count+2] = numext::conj(rhs(j2+2,k));
          blockB[count+3] = numext::conj(rhs(j2+3,k));
        }
        count += nr;
      }
    }

    // copy the remaining columns one at a time (=> the same with nr==1)
    for(Index j2=packet_cols; j2<cols; ++j2)
    {
      // transpose
      Index half = (std::min)(end_k,j2);
      for(Index k=k2; k<half; k++)
      {
        blockB[count] = numext::conj(rhs(j2,k));
        count += 1;
      }

      if(half==j2 && half<k2+rows)
      {
        blockB[count] = numext::real(rhs(j2,j2));
        count += 1;
      }
      else
        half--;

      // normal
      for(Index k=half+1; k<k2+rows; k++)
      {
        blockB[count] = rhs(k,j2);
        count += 1;
      }
    }
  }
};

/* Optimized selfadjoint matrix * matrix (_SYMM) product built on top of
 * the general matrix matrix product.
 */
template <typename Scalar, typename Index,
          int LhsStorageOrder, bool LhsSelfAdjoint, bool ConjugateLhs,
          int RhsStorageOrder, bool RhsSelfAdjoint, bool ConjugateRhs,
          int ResStorageOrder>
struct product_selfadjoint_matrix;

template <typename Scalar, typename Index,
          int LhsStorageOrder, bool LhsSelfAdjoint, bool ConjugateLhs,
          int RhsStorageOrder, bool RhsSelfAdjoint, bool ConjugateRhs>
struct product_selfadjoint_matrix<Scalar,Index,LhsStorageOrder,LhsSelfAdjoint,ConjugateLhs, RhsStorageOrder,RhsSelfAdjoint,ConjugateRhs,RowMajor>
{

  static EIGEN_STRONG_INLINE void run(
    Index rows, Index cols,
    const Scalar* lhs, Index lhsStride,
    const Scalar* rhs, Index rhsStride,
    Scalar* res,       Index resStride,
    const Scalar& alpha)
  {
    product_selfadjoint_matrix<Scalar, Index,
      EIGEN_LOGICAL_XOR(RhsSelfAdjoint,RhsStorageOrder==RowMajor) ? ColMajor : RowMajor,
      RhsSelfAdjoint, NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(RhsSelfAdjoint,ConjugateRhs),
      EIGEN_LOGICAL_XOR(LhsSelfAdjoint,LhsStorageOrder==RowMajor) ? ColMajor : RowMajor,
      LhsSelfAdjoint, NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(LhsSelfAdjoint,ConjugateLhs),
      ColMajor>
      ::run(cols, rows,  rhs, rhsStride,  lhs, lhsStride,  res, resStride,  alpha);
  }
};

template <typename Scalar, typename Index,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
struct product_selfadjoint_matrix<Scalar,Index,LhsStorageOrder,true,ConjugateLhs, RhsStorageOrder,false,ConjugateRhs,ColMajor>
{

  static EIGEN_DONT_INLINE void run(
    Index rows, Index cols,
    const Scalar* _lhs, Index lhsStride,
    const Scalar* _rhs, Index rhsStride,
    Scalar* res,        Index resStride,
    const Scalar& alpha);
};

template <typename Scalar, typename Index,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
EIGEN_DONT_INLINE void product_selfadjoint_matrix<Scalar,Index,LhsStorageOrder,true,ConjugateLhs, RhsStorageOrder,false,ConjugateRhs,ColMajor>::run(
    Index rows, Index cols,
    const Scalar* _lhs, Index lhsStride,
    const Scalar* _rhs, Index rhsStride,
    Scalar* res,        Index resStride,
    const Scalar& alpha)
  {
    Index size = rows;

    const_blas_data_mapper<Scalar, Index, LhsStorageOrder> lhs(_lhs,lhsStride);
    const_blas_data_mapper<Scalar, Index, RhsStorageOrder> rhs(_rhs,rhsStride);

    typedef gebp_traits<Scalar,Scalar> Traits;

    Index kc = size;  // cache block size along the K direction
    Index mc = rows;  // cache block size along the M direction
    Index nc = cols;  // cache block size along the N direction
    computeProductBlockingSizes<Scalar,Scalar>(kc, mc, nc);
    // kc must smaller than mc
    kc = (std::min)(kc,mc);

    std::size_t sizeW = kc*Traits::WorkSpaceFactor;
    std::size_t sizeB = sizeW + kc*cols;
    ei_declare_aligned_stack_constructed_variable(Scalar, blockA, kc*mc, 0);
    ei_declare_aligned_stack_constructed_variable(Scalar, allocatedBlockB, sizeB, 0);
    Scalar* blockB = allocatedBlockB + sizeW;

    gebp_kernel<Scalar, Scalar, Index, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp_kernel;
    symm_pack_lhs<Scalar, Index, Traits::mr, Traits::LhsProgress, LhsStorageOrder> pack_lhs;
    gemm_pack_rhs<Scalar, Index, Traits::nr,RhsStorageOrder> pack_rhs;
    gemm_pack_lhs<Scalar, Index, Traits::mr, Traits::LhsProgress, LhsStorageOrder==RowMajor?ColMajor:RowMajor, true> pack_lhs_transposed;

    for(Index k2=0; k2<size; k2+=kc)
    {
      const Index actual_kc = (std::min)(k2+kc,size)-k2;

      // we have selected one row panel of rhs and one column panel of lhs
      // pack rhs's panel into a sequential chunk of memory
      // and expand each coeff to a constant packet for further reuse
      pack_rhs(blockB, &rhs(k2,0), rhsStride, actual_kc, cols);

      // the select lhs's panel has to be split in three different parts:
      //  1 - the transposed panel above the diagonal block => transposed packed copy
      //  2 - the diagonal block => special packed copy
      //  3 - the panel below the diagonal block => generic packed copy
      for(Index i2=0; i2<k2; i2+=mc)
      {
        const Index actual_mc = (std::min)(i2+mc,k2)-i2;
        // transposed packed copy
        pack_lhs_transposed(blockA, &lhs(k2, i2), lhsStride, actual_kc, actual_mc);

        gebp_kernel(res+i2, resStride, blockA, blockB, actual_mc, actual_kc, cols, alpha);
      }
      // the block diagonal
      {
        const Index actual_mc = (std::min)(k2+kc,size)-k2;
        // symmetric packed copy
        pack_lhs(blockA, &lhs(k2,k2), lhsStride, actual_kc, actual_mc);

        gebp_kernel(res+k2, resStride, blockA, blockB, actual_mc, actual_kc, cols, alpha);
      }

      for(Index i2=k2+kc; i2<size; i2+=mc)
      {
        const Index actual_mc = (std::min)(i2+mc,size)-i2;
        gemm_pack_lhs<Scalar, Index, Traits::mr, Traits::LhsProgress, LhsStorageOrder,false>()
          (blockA, &lhs(i2, k2), lhsStride, actual_kc, actual_mc);

        gebp_kernel(res+i2, resStride, blockA, blockB, actual_mc, actual_kc, cols, alpha);
      }
    }
  }

// matrix * selfadjoint product
template <typename Scalar, typename Index,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
struct product_selfadjoint_matrix<Scalar,Index,LhsStorageOrder,false,ConjugateLhs, RhsStorageOrder,true,ConjugateRhs,ColMajor>
{

  static EIGEN_DONT_INLINE void run(
    Index rows, Index cols,
    const Scalar* _lhs, Index lhsStride,
    const Scalar* _rhs, Index rhsStride,
    Scalar* res,        Index resStride,
    const Scalar& alpha);
};

template <typename Scalar, typename Index,
          int LhsStorageOrder, bool ConjugateLhs,
          int RhsStorageOrder, bool ConjugateRhs>
EIGEN_DONT_INLINE void product_selfadjoint_matrix<Scalar,Index,LhsStorageOrder,false,ConjugateLhs, RhsStorageOrder,true,ConjugateRhs,ColMajor>::run(
    Index rows, Index cols,
    const Scalar* _lhs, Index lhsStride,
    const Scalar* _rhs, Index rhsStride,
    Scalar* res,        Index resStride,
    const Scalar& alpha)
  {
    Index size = cols;

    const_blas_data_mapper<Scalar, Index, LhsStorageOrder> lhs(_lhs,lhsStride);

    typedef gebp_traits<Scalar,Scalar> Traits;

    Index kc = size; // cache block size along the K direction
    Index mc = rows;  // cache block size along the M direction
    Index nc = cols;  // cache block size along the N direction
    computeProductBlockingSizes<Scalar,Scalar>(kc, mc, nc);
    std::size_t sizeW = kc*Traits::WorkSpaceFactor;
    std::size_t sizeB = sizeW + kc*cols;
    ei_declare_aligned_stack_constructed_variable(Scalar, blockA, kc*mc, 0);
    ei_declare_aligned_stack_constructed_variable(Scalar, allocatedBlockB, sizeB, 0);
    Scalar* blockB = allocatedBlockB + sizeW;

    gebp_kernel<Scalar, Scalar, Index, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp_kernel;
    gemm_pack_lhs<Scalar, Index, Traits::mr, Traits::LhsProgress, LhsStorageOrder> pack_lhs;
    symm_pack_rhs<Scalar, Index, Traits::nr,RhsStorageOrder> pack_rhs;

    for(Index k2=0; k2<size; k2+=kc)
    {
      const Index actual_kc = (std::min)(k2+kc,size)-k2;

      pack_rhs(blockB, _rhs, rhsStride, actual_kc, cols, k2);

      // => GEPP
      for(Index i2=0; i2<rows; i2+=mc)
      {
        const Index actual_mc = (std::min)(i2+mc,rows)-i2;
        pack_lhs(blockA, &lhs(i2, k2), lhsStride, actual_kc, actual_mc);

        gebp_kernel(res+i2, resStride, blockA, blockB, actual_mc, actual_kc, cols, alpha);
      }
    }
  }

} // end namespace internal

/***************************************************************************
* Wrapper to product_selfadjoint_matrix
***************************************************************************/

namespace internal {
template<typename Lhs, int LhsMode, typename Rhs, int RhsMode>
struct traits<SelfadjointProductMatrix<Lhs,LhsMode,false,Rhs,RhsMode,false> >
  : traits<ProductBase<SelfadjointProductMatrix<Lhs,LhsMode,false,Rhs,RhsMode,false>, Lhs, Rhs> >
{};
}

template<typename Lhs, int LhsMode, typename Rhs, int RhsMode>
struct SelfadjointProductMatrix<Lhs,LhsMode,false,Rhs,RhsMode,false>
  : public ProductBase<SelfadjointProductMatrix<Lhs,LhsMode,false,Rhs,RhsMode,false>, Lhs, Rhs >
{
  EIGEN_PRODUCT_PUBLIC_INTERFACE(SelfadjointProductMatrix)

  SelfadjointProductMatrix(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs) {}

  enum {
    LhsIsUpper = (LhsMode&(Upper|Lower))==Upper,
    LhsIsSelfAdjoint = (LhsMode&SelfAdjoint)==SelfAdjoint,
    RhsIsUpper = (RhsMode&(Upper|Lower))==Upper,
    RhsIsSelfAdjoint = (RhsMode&SelfAdjoint)==SelfAdjoint
  };

  template<typename Dest> void scaleAndAddTo(Dest& dst, const Scalar& alpha) const
  {
    eigen_assert(dst.rows()==m_lhs.rows() && dst.cols()==m_rhs.cols());

    typename internal::add_const_on_value_type<ActualLhsType>::type lhs = LhsBlasTraits::extract(m_lhs);
    typename internal::add_const_on_value_type<ActualRhsType>::type rhs = RhsBlasTraits::extract(m_rhs);

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                               * RhsBlasTraits::extractScalarFactor(m_rhs);

    internal::product_selfadjoint_matrix<Scalar, Index,
      EIGEN_LOGICAL_XOR(LhsIsUpper,
                        internal::traits<Lhs>::Flags &RowMajorBit) ? RowMajor : ColMajor, LhsIsSelfAdjoint,
      NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(LhsIsUpper,bool(LhsBlasTraits::NeedToConjugate)),
      EIGEN_LOGICAL_XOR(RhsIsUpper,
                        internal::traits<Rhs>::Flags &RowMajorBit) ? RowMajor : ColMajor, RhsIsSelfAdjoint,
      NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(RhsIsUpper,bool(RhsBlasTraits::NeedToConjugate)),
      internal::traits<Dest>::Flags&RowMajorBit  ? RowMajor : ColMajor>
      ::run(
        lhs.rows(), rhs.cols(),                 // sizes
        &lhs.coeffRef(0,0),    lhs.outerStride(),  // lhs info
        &rhs.coeffRef(0,0),    rhs.outerStride(),  // rhs info
        &dst.coeffRef(0,0), dst.outerStride(),  // result info
        actualAlpha                             // alpha
      );
  }
};

} // end namespace Eigen

#endif // EIGEN_SELFADJOINT_MATRIX_MATRIX_H
