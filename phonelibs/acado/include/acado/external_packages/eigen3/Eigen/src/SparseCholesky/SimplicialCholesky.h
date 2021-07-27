// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2012 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SIMPLICIAL_CHOLESKY_H
#define EIGEN_SIMPLICIAL_CHOLESKY_H

namespace Eigen { 

enum SimplicialCholeskyMode {
  SimplicialCholeskyLLT,
  SimplicialCholeskyLDLT
};

/** \ingroup SparseCholesky_Module
  * \brief A direct sparse Cholesky factorizations
  *
  * These classes provide LL^T and LDL^T Cholesky factorizations of sparse matrices that are
  * selfadjoint and positive definite. The factorization allows for solving A.X = B where
  * X and B can be either dense or sparse.
  * 
  * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
  * such that the factorized matrix is P A P^-1.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
  *               or Upper. Default is Lower.
  *
  */
template<typename Derived>
class SimplicialCholeskyBase : internal::noncopyable
{
  public:
    typedef typename internal::traits<Derived>::MatrixType MatrixType;
    enum { UpLo = internal::traits<Derived>::UpLo };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;
    typedef SparseMatrix<Scalar,ColMajor,Index> CholMatrixType;
    typedef Matrix<Scalar,Dynamic,1> VectorType;

  public:

    /** Default constructor */
    SimplicialCholeskyBase()
      : m_info(Success), m_isInitialized(false), m_shiftOffset(0), m_shiftScale(1)
    {}

    SimplicialCholeskyBase(const MatrixType& matrix)
      : m_info(Success), m_isInitialized(false), m_shiftOffset(0), m_shiftScale(1)
    {
      derived().compute(matrix);
    }

    ~SimplicialCholeskyBase()
    {
    }

    Derived& derived() { return *static_cast<Derived*>(this); }
    const Derived& derived() const { return *static_cast<const Derived*>(this); }
    
    inline Index cols() const { return m_matrix.cols(); }
    inline Index rows() const { return m_matrix.rows(); }
    
    /** \brief Reports whether previous computation was successful.
      *
      * \returns \c Success if computation was succesful,
      *          \c NumericalIssue if the matrix.appears to be negative.
      */
    ComputationInfo info() const
    {
      eigen_assert(m_isInitialized && "Decomposition is not initialized.");
      return m_info;
    }
    
    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::solve_retval<SimplicialCholeskyBase, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "Simplicial LLT or LDLT is not initialized.");
      eigen_assert(rows()==b.rows()
                && "SimplicialCholeskyBase::solve(): invalid number of rows of the right hand side matrix b");
      return internal::solve_retval<SimplicialCholeskyBase, Rhs>(*this, b.derived());
    }
    
    /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A.
      *
      * \sa compute()
      */
    template<typename Rhs>
    inline const internal::sparse_solve_retval<SimplicialCholeskyBase, Rhs>
    solve(const SparseMatrixBase<Rhs>& b) const
    {
      eigen_assert(m_isInitialized && "Simplicial LLT or LDLT is not initialized.");
      eigen_assert(rows()==b.rows()
                && "SimplicialCholesky::solve(): invalid number of rows of the right hand side matrix b");
      return internal::sparse_solve_retval<SimplicialCholeskyBase, Rhs>(*this, b.derived());
    }
    
    /** \returns the permutation P
      * \sa permutationPinv() */
    const PermutationMatrix<Dynamic,Dynamic,Index>& permutationP() const
    { return m_P; }
    
    /** \returns the inverse P^-1 of the permutation P
      * \sa permutationP() */
    const PermutationMatrix<Dynamic,Dynamic,Index>& permutationPinv() const
    { return m_Pinv; }

    /** Sets the shift parameters that will be used to adjust the diagonal coefficients during the numerical factorization.
      *
      * During the numerical factorization, the diagonal coefficients are transformed by the following linear model:\n
      * \c d_ii = \a offset + \a scale * \c d_ii
      *
      * The default is the identity transformation with \a offset=0, and \a scale=1.
      *
      * \returns a reference to \c *this.
      */
    Derived& setShift(const RealScalar& offset, const RealScalar& scale = 1)
    {
      m_shiftOffset = offset;
      m_shiftScale = scale;
      return derived();
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal */
    template<typename Stream>
    void dumpMemory(Stream& s)
    {
      int total = 0;
      s << "  L:        " << ((total+=(m_matrix.cols()+1) * sizeof(int) + m_matrix.nonZeros()*(sizeof(int)+sizeof(Scalar))) >> 20) << "Mb" << "\n";
      s << "  diag:     " << ((total+=m_diag.size() * sizeof(Scalar)) >> 20) << "Mb" << "\n";
      s << "  tree:     " << ((total+=m_parent.size() * sizeof(int)) >> 20) << "Mb" << "\n";
      s << "  nonzeros: " << ((total+=m_nonZerosPerCol.size() * sizeof(int)) >> 20) << "Mb" << "\n";
      s << "  perm:     " << ((total+=m_P.size() * sizeof(int)) >> 20) << "Mb" << "\n";
      s << "  perm^-1:  " << ((total+=m_Pinv.size() * sizeof(int)) >> 20) << "Mb" << "\n";
      s << "  TOTAL:    " << (total>> 20) << "Mb" << "\n";
    }

    /** \internal */
    template<typename Rhs,typename Dest>
    void _solve(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const
    {
      eigen_assert(m_factorizationIsOk && "The decomposition is not in a valid state for solving, you must first call either compute() or symbolic()/numeric()");
      eigen_assert(m_matrix.rows()==b.rows());

      if(m_info!=Success)
        return;

      if(m_P.size()>0)
        dest = m_P * b;
      else
        dest = b;

      if(m_matrix.nonZeros()>0) // otherwise L==I
        derived().matrixL().solveInPlace(dest);

      if(m_diag.size()>0)
        dest = m_diag.asDiagonal().inverse() * dest;

      if (m_matrix.nonZeros()>0) // otherwise U==I
        derived().matrixU().solveInPlace(dest);

      if(m_P.size()>0)
        dest = m_Pinv * dest;
    }

#endif // EIGEN_PARSED_BY_DOXYGEN

  protected:
    
    /** Computes the sparse Cholesky decomposition of \a matrix */
    template<bool DoLDLT>
    void compute(const MatrixType& matrix)
    {
      eigen_assert(matrix.rows()==matrix.cols());
      Index size = matrix.cols();
      CholMatrixType ap(size,size);
      ordering(matrix, ap);
      analyzePattern_preordered(ap, DoLDLT);
      factorize_preordered<DoLDLT>(ap);
    }
    
    template<bool DoLDLT>
    void factorize(const MatrixType& a)
    {
      eigen_assert(a.rows()==a.cols());
      int size = a.cols();
      CholMatrixType ap(size,size);
      ap.template selfadjointView<Upper>() = a.template selfadjointView<UpLo>().twistedBy(m_P);
      factorize_preordered<DoLDLT>(ap);
    }

    template<bool DoLDLT>
    void factorize_preordered(const CholMatrixType& a);

    void analyzePattern(const MatrixType& a, bool doLDLT)
    {
      eigen_assert(a.rows()==a.cols());
      int size = a.cols();
      CholMatrixType ap(size,size);
      ordering(a, ap);
      analyzePattern_preordered(ap,doLDLT);
    }
    void analyzePattern_preordered(const CholMatrixType& a, bool doLDLT);
    
    void ordering(const MatrixType& a, CholMatrixType& ap);

    /** keeps off-diagonal entries; drops diagonal entries */
    struct keep_diag {
      inline bool operator() (const Index& row, const Index& col, const Scalar&) const
      {
        return row!=col;
      }
    };

    mutable ComputationInfo m_info;
    bool m_isInitialized;
    bool m_factorizationIsOk;
    bool m_analysisIsOk;
    
    CholMatrixType m_matrix;
    VectorType m_diag;                                // the diagonal coefficients (LDLT mode)
    VectorXi m_parent;                                // elimination tree
    VectorXi m_nonZerosPerCol;
    PermutationMatrix<Dynamic,Dynamic,Index> m_P;     // the permutation
    PermutationMatrix<Dynamic,Dynamic,Index> m_Pinv;  // the inverse permutation

    RealScalar m_shiftOffset;
    RealScalar m_shiftScale;
};

template<typename _MatrixType, int _UpLo = Lower> class SimplicialLLT;
template<typename _MatrixType, int _UpLo = Lower> class SimplicialLDLT;
template<typename _MatrixType, int _UpLo = Lower> class SimplicialCholesky;

namespace internal {

template<typename _MatrixType, int _UpLo> struct traits<SimplicialLLT<_MatrixType,_UpLo> >
{
  typedef _MatrixType MatrixType;
  enum { UpLo = _UpLo };
  typedef typename MatrixType::Scalar                         Scalar;
  typedef typename MatrixType::Index                          Index;
  typedef SparseMatrix<Scalar, ColMajor, Index>               CholMatrixType;
  typedef SparseTriangularView<CholMatrixType, Eigen::Lower>  MatrixL;
  typedef SparseTriangularView<typename CholMatrixType::AdjointReturnType, Eigen::Upper>   MatrixU;
  static inline MatrixL getL(const MatrixType& m) { return m; }
  static inline MatrixU getU(const MatrixType& m) { return m.adjoint(); }
};

template<typename _MatrixType,int _UpLo> struct traits<SimplicialLDLT<_MatrixType,_UpLo> >
{
  typedef _MatrixType MatrixType;
  enum { UpLo = _UpLo };
  typedef typename MatrixType::Scalar                             Scalar;
  typedef typename MatrixType::Index                              Index;
  typedef SparseMatrix<Scalar, ColMajor, Index>                   CholMatrixType;
  typedef SparseTriangularView<CholMatrixType, Eigen::UnitLower>  MatrixL;
  typedef SparseTriangularView<typename CholMatrixType::AdjointReturnType, Eigen::UnitUpper> MatrixU;
  static inline MatrixL getL(const MatrixType& m) { return m; }
  static inline MatrixU getU(const MatrixType& m) { return m.adjoint(); }
};

template<typename _MatrixType, int _UpLo> struct traits<SimplicialCholesky<_MatrixType,_UpLo> >
{
  typedef _MatrixType MatrixType;
  enum { UpLo = _UpLo };
};

}

/** \ingroup SparseCholesky_Module
  * \class SimplicialLLT
  * \brief A direct sparse LLT Cholesky factorizations
  *
  * This class provides a LL^T Cholesky factorizations of sparse matrices that are
  * selfadjoint and positive definite. The factorization allows for solving A.X = B where
  * X and B can be either dense or sparse.
  * 
  * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
  * such that the factorized matrix is P A P^-1.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
  *               or Upper. Default is Lower.
  *
  * \sa class SimplicialLDLT
  */
template<typename _MatrixType, int _UpLo>
    class SimplicialLLT : public SimplicialCholeskyBase<SimplicialLLT<_MatrixType,_UpLo> >
{
public:
    typedef _MatrixType MatrixType;
    enum { UpLo = _UpLo };
    typedef SimplicialCholeskyBase<SimplicialLLT> Base;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;
    typedef SparseMatrix<Scalar,ColMajor,Index> CholMatrixType;
    typedef Matrix<Scalar,Dynamic,1> VectorType;
    typedef internal::traits<SimplicialLLT> Traits;
    typedef typename Traits::MatrixL  MatrixL;
    typedef typename Traits::MatrixU  MatrixU;
public:
    /** Default constructor */
    SimplicialLLT() : Base() {}
    /** Constructs and performs the LLT factorization of \a matrix */
    SimplicialLLT(const MatrixType& matrix)
        : Base(matrix) {}

    /** \returns an expression of the factor L */
    inline const MatrixL matrixL() const {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LLT not factorized");
        return Traits::getL(Base::m_matrix);
    }

    /** \returns an expression of the factor U (= L^*) */
    inline const MatrixU matrixU() const {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LLT not factorized");
        return Traits::getU(Base::m_matrix);
    }
    
    /** Computes the sparse Cholesky decomposition of \a matrix */
    SimplicialLLT& compute(const MatrixType& matrix)
    {
      Base::template compute<false>(matrix);
      return *this;
    }

    /** Performs a symbolic decomposition on the sparcity of \a matrix.
      *
      * This function is particularly useful when solving for several problems having the same structure.
      *
      * \sa factorize()
      */
    void analyzePattern(const MatrixType& a)
    {
      Base::analyzePattern(a, false);
    }

    /** Performs a numeric decomposition of \a matrix
      *
      * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been performed.
      *
      * \sa analyzePattern()
      */
    void factorize(const MatrixType& a)
    {
      Base::template factorize<false>(a);
    }

    /** \returns the determinant of the underlying matrix from the current factorization */
    Scalar determinant() const
    {
      Scalar detL = Base::m_matrix.diagonal().prod();
      return numext::abs2(detL);
    }
};

/** \ingroup SparseCholesky_Module
  * \class SimplicialLDLT
  * \brief A direct sparse LDLT Cholesky factorizations without square root.
  *
  * This class provides a LDL^T Cholesky factorizations without square root of sparse matrices that are
  * selfadjoint and positive definite. The factorization allows for solving A.X = B where
  * X and B can be either dense or sparse.
  * 
  * In order to reduce the fill-in, a symmetric permutation P is applied prior to the factorization
  * such that the factorized matrix is P A P^-1.
  *
  * \tparam _MatrixType the type of the sparse matrix A, it must be a SparseMatrix<>
  * \tparam _UpLo the triangular part that will be used for the computations. It can be Lower
  *               or Upper. Default is Lower.
  *
  * \sa class SimplicialLLT
  */
template<typename _MatrixType, int _UpLo>
    class SimplicialLDLT : public SimplicialCholeskyBase<SimplicialLDLT<_MatrixType,_UpLo> >
{
public:
    typedef _MatrixType MatrixType;
    enum { UpLo = _UpLo };
    typedef SimplicialCholeskyBase<SimplicialLDLT> Base;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;
    typedef SparseMatrix<Scalar,ColMajor,Index> CholMatrixType;
    typedef Matrix<Scalar,Dynamic,1> VectorType;
    typedef internal::traits<SimplicialLDLT> Traits;
    typedef typename Traits::MatrixL  MatrixL;
    typedef typename Traits::MatrixU  MatrixU;
public:
    /** Default constructor */
    SimplicialLDLT() : Base() {}

    /** Constructs and performs the LLT factorization of \a matrix */
    SimplicialLDLT(const MatrixType& matrix)
        : Base(matrix) {}

    /** \returns a vector expression of the diagonal D */
    inline const VectorType vectorD() const {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
        return Base::m_diag;
    }
    /** \returns an expression of the factor L */
    inline const MatrixL matrixL() const {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
        return Traits::getL(Base::m_matrix);
    }

    /** \returns an expression of the factor U (= L^*) */
    inline const MatrixU matrixU() const {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LDLT not factorized");
        return Traits::getU(Base::m_matrix);
    }

    /** Computes the sparse Cholesky decomposition of \a matrix */
    SimplicialLDLT& compute(const MatrixType& matrix)
    {
      Base::template compute<true>(matrix);
      return *this;
    }
    
    /** Performs a symbolic decomposition on the sparcity of \a matrix.
      *
      * This function is particularly useful when solving for several problems having the same structure.
      *
      * \sa factorize()
      */
    void analyzePattern(const MatrixType& a)
    {
      Base::analyzePattern(a, true);
    }

    /** Performs a numeric decomposition of \a matrix
      *
      * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been performed.
      *
      * \sa analyzePattern()
      */
    void factorize(const MatrixType& a)
    {
      Base::template factorize<true>(a);
    }

    /** \returns the determinant of the underlying matrix from the current factorization */
    Scalar determinant() const
    {
      return Base::m_diag.prod();
    }
};

/** \deprecated use SimplicialLDLT or class SimplicialLLT
  * \ingroup SparseCholesky_Module
  * \class SimplicialCholesky
  *
  * \sa class SimplicialLDLT, class SimplicialLLT
  */
template<typename _MatrixType, int _UpLo>
    class SimplicialCholesky : public SimplicialCholeskyBase<SimplicialCholesky<_MatrixType,_UpLo> >
{
public:
    typedef _MatrixType MatrixType;
    enum { UpLo = _UpLo };
    typedef SimplicialCholeskyBase<SimplicialCholesky> Base;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;
    typedef SparseMatrix<Scalar,ColMajor,Index> CholMatrixType;
    typedef Matrix<Scalar,Dynamic,1> VectorType;
    typedef internal::traits<SimplicialCholesky> Traits;
    typedef internal::traits<SimplicialLDLT<MatrixType,UpLo> > LDLTTraits;
    typedef internal::traits<SimplicialLLT<MatrixType,UpLo>  > LLTTraits;
  public:
    SimplicialCholesky() : Base(), m_LDLT(true) {}

    SimplicialCholesky(const MatrixType& matrix)
      : Base(), m_LDLT(true)
    {
      compute(matrix);
    }

    SimplicialCholesky& setMode(SimplicialCholeskyMode mode)
    {
      switch(mode)
      {
      case SimplicialCholeskyLLT:
        m_LDLT = false;
        break;
      case SimplicialCholeskyLDLT:
        m_LDLT = true;
        break;
      default:
        break;
      }

      return *this;
    }

    inline const VectorType vectorD() const {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial Cholesky not factorized");
        return Base::m_diag;
    }
    inline const CholMatrixType rawMatrix() const {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial Cholesky not factorized");
        return Base::m_matrix;
    }
    
    /** Computes the sparse Cholesky decomposition of \a matrix */
    SimplicialCholesky& compute(const MatrixType& matrix)
    {
      if(m_LDLT)
        Base::template compute<true>(matrix);
      else
        Base::template compute<false>(matrix);
      return *this;
    }

    /** Performs a symbolic decomposition on the sparcity of \a matrix.
      *
      * This function is particularly useful when solving for several problems having the same structure.
      *
      * \sa factorize()
      */
    void analyzePattern(const MatrixType& a)
    {
      Base::analyzePattern(a, m_LDLT);
    }

    /** Performs a numeric decomposition of \a matrix
      *
      * The given matrix must has the same sparcity than the matrix on which the symbolic decomposition has been performed.
      *
      * \sa analyzePattern()
      */
    void factorize(const MatrixType& a)
    {
      if(m_LDLT)
        Base::template factorize<true>(a);
      else
        Base::template factorize<false>(a);
    }

    /** \internal */
    template<typename Rhs,typename Dest>
    void _solve(const MatrixBase<Rhs> &b, MatrixBase<Dest> &dest) const
    {
      eigen_assert(Base::m_factorizationIsOk && "The decomposition is not in a valid state for solving, you must first call either compute() or symbolic()/numeric()");
      eigen_assert(Base::m_matrix.rows()==b.rows());

      if(Base::m_info!=Success)
        return;

      if(Base::m_P.size()>0)
        dest = Base::m_P * b;
      else
        dest = b;

      if(Base::m_matrix.nonZeros()>0) // otherwise L==I
      {
        if(m_LDLT)
          LDLTTraits::getL(Base::m_matrix).solveInPlace(dest);
        else
          LLTTraits::getL(Base::m_matrix).solveInPlace(dest);
      }

      if(Base::m_diag.size()>0)
        dest = Base::m_diag.asDiagonal().inverse() * dest;

      if (Base::m_matrix.nonZeros()>0) // otherwise I==I
      {
        if(m_LDLT)
          LDLTTraits::getU(Base::m_matrix).solveInPlace(dest);
        else
          LLTTraits::getU(Base::m_matrix).solveInPlace(dest);
      }

      if(Base::m_P.size()>0)
        dest = Base::m_Pinv * dest;
    }
    
    Scalar determinant() const
    {
      if(m_LDLT)
      {
        return Base::m_diag.prod();
      }
      else
      {
        Scalar detL = Diagonal<const CholMatrixType>(Base::m_matrix).prod();
        return numext::abs2(detL);
      }
    }
    
  protected:
    bool m_LDLT;
};

template<typename Derived>
void SimplicialCholeskyBase<Derived>::ordering(const MatrixType& a, CholMatrixType& ap)
{
  eigen_assert(a.rows()==a.cols());
  const Index size = a.rows();
  // TODO allows to configure the permutation
  // Note that amd compute the inverse permutation
  {
    CholMatrixType C;
    C = a.template selfadjointView<UpLo>();
    // remove diagonal entries:
    // seems not to be needed
    // C.prune(keep_diag());
    internal::minimum_degree_ordering(C, m_Pinv);
  }

  if(m_Pinv.size()>0)
    m_P = m_Pinv.inverse();
  else
    m_P.resize(0);

  ap.resize(size,size);
  ap.template selfadjointView<Upper>() = a.template selfadjointView<UpLo>().twistedBy(m_P);
}

namespace internal {
  
template<typename Derived, typename Rhs>
struct solve_retval<SimplicialCholeskyBase<Derived>, Rhs>
  : solve_retval_base<SimplicialCholeskyBase<Derived>, Rhs>
{
  typedef SimplicialCholeskyBase<Derived> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec().derived()._solve(rhs(),dst);
  }
};

template<typename Derived, typename Rhs>
struct sparse_solve_retval<SimplicialCholeskyBase<Derived>, Rhs>
  : sparse_solve_retval_base<SimplicialCholeskyBase<Derived>, Rhs>
{
  typedef SimplicialCholeskyBase<Derived> Dec;
  EIGEN_MAKE_SPARSE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    this->defaultEvalTo(dst);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_SIMPLICIAL_CHOLESKY_H
