// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERAL_MATRIX_MATRIX_H
#define EIGEN_GENERAL_MATRIX_MATRIX_H

namespace Eigen { 

namespace internal {

template<typename _LhsScalar, typename _RhsScalar> class level3_blocking;

/* Specialization for a row-major destination matrix => simple transposition of the product */
template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs>
struct general_matrix_matrix_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,RhsStorageOrder,ConjugateRhs,RowMajor>
{
  typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;
  static EIGEN_STRONG_INLINE void run(
    Index rows, Index cols, Index depth,
    const LhsScalar* lhs, Index lhsStride,
    const RhsScalar* rhs, Index rhsStride,
    ResScalar* res, Index resStride,
    ResScalar alpha,
    level3_blocking<RhsScalar,LhsScalar>& blocking,
    GemmParallelInfo<Index>* info = 0)
  {
    // transpose the product such that the result is column major
    general_matrix_matrix_product<Index,
      RhsScalar, RhsStorageOrder==RowMajor ? ColMajor : RowMajor, ConjugateRhs,
      LhsScalar, LhsStorageOrder==RowMajor ? ColMajor : RowMajor, ConjugateLhs,
      ColMajor>
    ::run(cols,rows,depth,rhs,rhsStride,lhs,lhsStride,res,resStride,alpha,blocking,info);
  }
};

/*  Specialization for a col-major destination matrix
 *    => Blocking algorithm following Goto's paper */
template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs>
struct general_matrix_matrix_product<Index,LhsScalar,LhsStorageOrder,ConjugateLhs,RhsScalar,RhsStorageOrder,ConjugateRhs,ColMajor>
{

typedef typename scalar_product_traits<LhsScalar, RhsScalar>::ReturnType ResScalar;
static void run(Index rows, Index cols, Index depth,
  const LhsScalar* _lhs, Index lhsStride,
  const RhsScalar* _rhs, Index rhsStride,
  ResScalar* res, Index resStride,
  ResScalar alpha,
  level3_blocking<LhsScalar,RhsScalar>& blocking,
  GemmParallelInfo<Index>* info = 0)
{
  const_blas_data_mapper<LhsScalar, Index, LhsStorageOrder> lhs(_lhs,lhsStride);
  const_blas_data_mapper<RhsScalar, Index, RhsStorageOrder> rhs(_rhs,rhsStride);

  typedef gebp_traits<LhsScalar,RhsScalar> Traits;

  Index kc = blocking.kc();                   // cache block size along the K direction
  Index mc = (std::min)(rows,blocking.mc());  // cache block size along the M direction
  //Index nc = blocking.nc(); // cache block size along the N direction

  gemm_pack_lhs<LhsScalar, Index, Traits::mr, Traits::LhsProgress, LhsStorageOrder> pack_lhs;
  gemm_pack_rhs<RhsScalar, Index, Traits::nr, RhsStorageOrder> pack_rhs;
  gebp_kernel<LhsScalar, RhsScalar, Index, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;

#ifdef EIGEN_HAS_OPENMP
  if(info)
  {
    // this is the parallel version!
    Index tid = omp_get_thread_num();
    Index threads = omp_get_num_threads();
    
    std::size_t sizeA = kc*mc;
    std::size_t sizeW = kc*Traits::WorkSpaceFactor;
    ei_declare_aligned_stack_constructed_variable(LhsScalar, blockA, sizeA, 0);
    ei_declare_aligned_stack_constructed_variable(RhsScalar, w, sizeW, 0);
    
    RhsScalar* blockB = blocking.blockB();
    eigen_internal_assert(blockB!=0);

    // For each horizontal panel of the rhs, and corresponding vertical panel of the lhs...
    for(Index k=0; k<depth; k+=kc)
    {
      const Index actual_kc = (std::min)(k+kc,depth)-k; // => rows of B', and cols of the A'

      // In order to reduce the chance that a thread has to wait for the other,
      // let's start by packing A'.
      pack_lhs(blockA, &lhs(0,k), lhsStride, actual_kc, mc);

      // Pack B_k to B' in a parallel fashion:
      // each thread packs the sub block B_k,j to B'_j where j is the thread id.

      // However, before copying to B'_j, we have to make sure that no other thread is still using it,
      // i.e., we test that info[tid].users equals 0.
      // Then, we set info[tid].users to the number of threads to mark that all other threads are going to use it.
      while(info[tid].users!=0) {}
      info[tid].users += threads;

      pack_rhs(blockB+info[tid].rhs_start*actual_kc, &rhs(k,info[tid].rhs_start), rhsStride, actual_kc, info[tid].rhs_length);

      // Notify the other threads that the part B'_j is ready to go.
      info[tid].sync = k;

      // Computes C_i += A' * B' per B'_j
      for(Index shift=0; shift<threads; ++shift)
      {
        Index j = (tid+shift)%threads;

        // At this point we have to make sure that B'_j has been updated by the thread j,
        // we use testAndSetOrdered to mimic a volatile access.
        // However, no need to wait for the B' part which has been updated by the current thread!
        if(shift>0)
          while(info[j].sync!=k) {}

        gebp(res+info[j].rhs_start*resStride, resStride, blockA, blockB+info[j].rhs_start*actual_kc, mc, actual_kc, info[j].rhs_length, alpha, -1,-1,0,0, w);
      }

      // Then keep going as usual with the remaining A'
      for(Index i=mc; i<rows; i+=mc)
      {
        const Index actual_mc = (std::min)(i+mc,rows)-i;

        // pack A_i,k to A'
        pack_lhs(blockA, &lhs(i,k), lhsStride, actual_kc, actual_mc);

        // C_i += A' * B'
        gebp(res+i, resStride, blockA, blockB, actual_mc, actual_kc, cols, alpha, -1,-1,0,0, w);
      }

      // Release all the sub blocks B'_j of B' for the current thread,
      // i.e., we simply decrement the number of users by 1
      for(Index j=0; j<threads; ++j)
        #pragma omp atomic
        --(info[j].users);
    }
  }
  else
#endif // EIGEN_HAS_OPENMP
  {
    EIGEN_UNUSED_VARIABLE(info);

    // this is the sequential version!
    std::size_t sizeA = kc*mc;
    std::size_t sizeB = kc*cols;
    std::size_t sizeW = kc*Traits::WorkSpaceFactor;

    ei_declare_aligned_stack_constructed_variable(LhsScalar, blockA, sizeA, blocking.blockA());
    ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, blocking.blockB());
    ei_declare_aligned_stack_constructed_variable(RhsScalar, blockW, sizeW, blocking.blockW());

    // For each horizontal panel of the rhs, and corresponding panel of the lhs...
    // (==GEMM_VAR1)
    for(Index k2=0; k2<depth; k2+=kc)
    {
      const Index actual_kc = (std::min)(k2+kc,depth)-k2;

      // OK, here we have selected one horizontal panel of rhs and one vertical panel of lhs.
      // => Pack rhs's panel into a sequential chunk of memory (L2 caching)
      // Note that this panel will be read as many times as the number of blocks in the lhs's
      // vertical panel which is, in practice, a very low number.
      pack_rhs(blockB, &rhs(k2,0), rhsStride, actual_kc, cols);

      // For each mc x kc block of the lhs's vertical panel...
      // (==GEPP_VAR1)
      for(Index i2=0; i2<rows; i2+=mc)
      {
        const Index actual_mc = (std::min)(i2+mc,rows)-i2;

        // We pack the lhs's block into a sequential chunk of memory (L1 caching)
        // Note that this block will be read a very high number of times, which is equal to the number of
        // micro vertical panel of the large rhs's panel (e.g., cols/4 times).
        pack_lhs(blockA, &lhs(i2,k2), lhsStride, actual_kc, actual_mc);

        // Everything is packed, we can now call the block * panel kernel:
        gebp(res+i2, resStride, blockA, blockB, actual_mc, actual_kc, cols, alpha, -1, -1, 0, 0, blockW);
      }
    }
  }
}

};

/*********************************************************************************
*  Specialization of GeneralProduct<> for "large" GEMM, i.e.,
*  implementation of the high level wrapper to general_matrix_matrix_product
**********************************************************************************/

template<typename Lhs, typename Rhs>
struct traits<GeneralProduct<Lhs,Rhs,GemmProduct> >
 : traits<ProductBase<GeneralProduct<Lhs,Rhs,GemmProduct>, Lhs, Rhs> >
{};

template<typename Scalar, typename Index, typename Gemm, typename Lhs, typename Rhs, typename Dest, typename BlockingType>
struct gemm_functor
{
  gemm_functor(const Lhs& lhs, const Rhs& rhs, Dest& dest, const Scalar& actualAlpha,
                  BlockingType& blocking)
    : m_lhs(lhs), m_rhs(rhs), m_dest(dest), m_actualAlpha(actualAlpha), m_blocking(blocking)
  {}

  void initParallelSession() const
  {
    m_blocking.allocateB();
  }

  void operator() (Index row, Index rows, Index col=0, Index cols=-1, GemmParallelInfo<Index>* info=0) const
  {
    if(cols==-1)
      cols = m_rhs.cols();

    Gemm::run(rows, cols, m_lhs.cols(),
              /*(const Scalar*)*/&m_lhs.coeffRef(row,0), m_lhs.outerStride(),
              /*(const Scalar*)*/&m_rhs.coeffRef(0,col), m_rhs.outerStride(),
              (Scalar*)&(m_dest.coeffRef(row,col)), m_dest.outerStride(),
              m_actualAlpha, m_blocking, info);
  }

  protected:
    const Lhs& m_lhs;
    const Rhs& m_rhs;
    Dest& m_dest;
    Scalar m_actualAlpha;
    BlockingType& m_blocking;
};

template<int StorageOrder, typename LhsScalar, typename RhsScalar, int MaxRows, int MaxCols, int MaxDepth, int KcFactor=1,
bool FiniteAtCompileTime = MaxRows!=Dynamic && MaxCols!=Dynamic && MaxDepth != Dynamic> class gemm_blocking_space;

template<typename _LhsScalar, typename _RhsScalar>
class level3_blocking
{
    typedef _LhsScalar LhsScalar;
    typedef _RhsScalar RhsScalar;

  protected:
    LhsScalar* m_blockA;
    RhsScalar* m_blockB;
    RhsScalar* m_blockW;

    DenseIndex m_mc;
    DenseIndex m_nc;
    DenseIndex m_kc;

  public:

    level3_blocking()
      : m_blockA(0), m_blockB(0), m_blockW(0), m_mc(0), m_nc(0), m_kc(0)
    {}

    inline DenseIndex mc() const { return m_mc; }
    inline DenseIndex nc() const { return m_nc; }
    inline DenseIndex kc() const { return m_kc; }

    inline LhsScalar* blockA() { return m_blockA; }
    inline RhsScalar* blockB() { return m_blockB; }
    inline RhsScalar* blockW() { return m_blockW; }
};

template<int StorageOrder, typename _LhsScalar, typename _RhsScalar, int MaxRows, int MaxCols, int MaxDepth, int KcFactor>
class gemm_blocking_space<StorageOrder,_LhsScalar,_RhsScalar,MaxRows, MaxCols, MaxDepth, KcFactor, true>
  : public level3_blocking<
      typename conditional<StorageOrder==RowMajor,_RhsScalar,_LhsScalar>::type,
      typename conditional<StorageOrder==RowMajor,_LhsScalar,_RhsScalar>::type>
{
    enum {
      Transpose = StorageOrder==RowMajor,
      ActualRows = Transpose ? MaxCols : MaxRows,
      ActualCols = Transpose ? MaxRows : MaxCols
    };
    typedef typename conditional<Transpose,_RhsScalar,_LhsScalar>::type LhsScalar;
    typedef typename conditional<Transpose,_LhsScalar,_RhsScalar>::type RhsScalar;
    typedef gebp_traits<LhsScalar,RhsScalar> Traits;
    enum {
      SizeA = ActualRows * MaxDepth,
      SizeB = ActualCols * MaxDepth,
      SizeW = MaxDepth * Traits::WorkSpaceFactor
    };

    EIGEN_ALIGN16 LhsScalar m_staticA[SizeA];
    EIGEN_ALIGN16 RhsScalar m_staticB[SizeB];
    EIGEN_ALIGN16 RhsScalar m_staticW[SizeW];

  public:

    gemm_blocking_space(DenseIndex /*rows*/, DenseIndex /*cols*/, DenseIndex /*depth*/)
    {
      this->m_mc = ActualRows;
      this->m_nc = ActualCols;
      this->m_kc = MaxDepth;
      this->m_blockA = m_staticA;
      this->m_blockB = m_staticB;
      this->m_blockW = m_staticW;
    }

    inline void allocateA() {}
    inline void allocateB() {}
    inline void allocateW() {}
    inline void allocateAll() {}
};

template<int StorageOrder, typename _LhsScalar, typename _RhsScalar, int MaxRows, int MaxCols, int MaxDepth, int KcFactor>
class gemm_blocking_space<StorageOrder,_LhsScalar,_RhsScalar,MaxRows, MaxCols, MaxDepth, KcFactor, false>
  : public level3_blocking<
      typename conditional<StorageOrder==RowMajor,_RhsScalar,_LhsScalar>::type,
      typename conditional<StorageOrder==RowMajor,_LhsScalar,_RhsScalar>::type>
{
    enum {
      Transpose = StorageOrder==RowMajor
    };
    typedef typename conditional<Transpose,_RhsScalar,_LhsScalar>::type LhsScalar;
    typedef typename conditional<Transpose,_LhsScalar,_RhsScalar>::type RhsScalar;
    typedef gebp_traits<LhsScalar,RhsScalar> Traits;

    DenseIndex m_sizeA;
    DenseIndex m_sizeB;
    DenseIndex m_sizeW;

  public:

    gemm_blocking_space(DenseIndex rows, DenseIndex cols, DenseIndex depth)
    {
      this->m_mc = Transpose ? cols : rows;
      this->m_nc = Transpose ? rows : cols;
      this->m_kc = depth;

      computeProductBlockingSizes<LhsScalar,RhsScalar,KcFactor>(this->m_kc, this->m_mc, this->m_nc);
      m_sizeA = this->m_mc * this->m_kc;
      m_sizeB = this->m_kc * this->m_nc;
      m_sizeW = this->m_kc*Traits::WorkSpaceFactor;
    }

    void allocateA()
    {
      if(this->m_blockA==0)
        this->m_blockA = aligned_new<LhsScalar>(m_sizeA);
    }

    void allocateB()
    {
      if(this->m_blockB==0)
        this->m_blockB = aligned_new<RhsScalar>(m_sizeB);
    }

    void allocateW()
    {
      if(this->m_blockW==0)
        this->m_blockW = aligned_new<RhsScalar>(m_sizeW);
    }

    void allocateAll()
    {
      allocateA();
      allocateB();
      allocateW();
    }

    ~gemm_blocking_space()
    {
      aligned_delete(this->m_blockA, m_sizeA);
      aligned_delete(this->m_blockB, m_sizeB);
      aligned_delete(this->m_blockW, m_sizeW);
    }
};

} // end namespace internal

template<typename Lhs, typename Rhs>
class GeneralProduct<Lhs, Rhs, GemmProduct>
  : public ProductBase<GeneralProduct<Lhs,Rhs,GemmProduct>, Lhs, Rhs>
{
    enum {
      MaxDepthAtCompileTime = EIGEN_SIZE_MIN_PREFER_FIXED(Lhs::MaxColsAtCompileTime,Rhs::MaxRowsAtCompileTime)
    };
  public:
    EIGEN_PRODUCT_PUBLIC_INTERFACE(GeneralProduct)
    
    typedef typename  Lhs::Scalar LhsScalar;
    typedef typename  Rhs::Scalar RhsScalar;
    typedef           Scalar      ResScalar;

    GeneralProduct(const Lhs& lhs, const Rhs& rhs) : Base(lhs,rhs)
    {
      typedef internal::scalar_product_op<LhsScalar,RhsScalar> BinOp;
      EIGEN_CHECK_BINARY_COMPATIBILIY(BinOp,LhsScalar,RhsScalar);
    }

    template<typename Dest> void scaleAndAddTo(Dest& dst, const Scalar& alpha) const
    {
      eigen_assert(dst.rows()==m_lhs.rows() && dst.cols()==m_rhs.cols());

      typename internal::add_const_on_value_type<ActualLhsType>::type lhs = LhsBlasTraits::extract(m_lhs);
      typename internal::add_const_on_value_type<ActualRhsType>::type rhs = RhsBlasTraits::extract(m_rhs);

      Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                                 * RhsBlasTraits::extractScalarFactor(m_rhs);

      typedef internal::gemm_blocking_space<(Dest::Flags&RowMajorBit) ? RowMajor : ColMajor,LhsScalar,RhsScalar,
              Dest::MaxRowsAtCompileTime,Dest::MaxColsAtCompileTime,MaxDepthAtCompileTime> BlockingType;

      typedef internal::gemm_functor<
        Scalar, Index,
        internal::general_matrix_matrix_product<
          Index,
          LhsScalar, (_ActualLhsType::Flags&RowMajorBit) ? RowMajor : ColMajor, bool(LhsBlasTraits::NeedToConjugate),
          RhsScalar, (_ActualRhsType::Flags&RowMajorBit) ? RowMajor : ColMajor, bool(RhsBlasTraits::NeedToConjugate),
          (Dest::Flags&RowMajorBit) ? RowMajor : ColMajor>,
        _ActualLhsType, _ActualRhsType, Dest, BlockingType> GemmFunctor;

      BlockingType blocking(dst.rows(), dst.cols(), lhs.cols());

      internal::parallelize_gemm<(Dest::MaxRowsAtCompileTime>32 || Dest::MaxRowsAtCompileTime==Dynamic)>(GemmFunctor(lhs, rhs, dst, actualAlpha, blocking), this->rows(), this->cols(), Dest::Flags&RowMajorBit);
    }
};

} // end namespace Eigen

#endif // EIGEN_GENERAL_MATRIX_MATRIX_H
