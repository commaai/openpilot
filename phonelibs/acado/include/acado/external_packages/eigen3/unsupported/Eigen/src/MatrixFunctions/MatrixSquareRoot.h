// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_SQUARE_ROOT
#define EIGEN_MATRIX_SQUARE_ROOT

namespace Eigen { 

/** \ingroup MatrixFunctions_Module
  * \brief Class for computing matrix square roots of upper quasi-triangular matrices.
  * \tparam  MatrixType  type of the argument of the matrix square root,
  *                      expected to be an instantiation of the Matrix class template.
  *
  * This class computes the square root of the upper quasi-triangular
  * matrix stored in the upper Hessenberg part of the matrix passed to
  * the constructor.
  *
  * \sa MatrixSquareRoot, MatrixSquareRootTriangular
  */
template <typename MatrixType>
class MatrixSquareRootQuasiTriangular
{
  public:

    /** \brief Constructor. 
      *
      * \param[in]  A  upper quasi-triangular matrix whose square root 
      *                is to be computed.
      *
      * The class stores a reference to \p A, so it should not be
      * changed (or destroyed) before compute() is called.
      */
    MatrixSquareRootQuasiTriangular(const MatrixType& A) 
      : m_A(A) 
    {
      eigen_assert(A.rows() == A.cols());
    }
    
    /** \brief Compute the matrix square root
      *
      * \param[out] result  square root of \p A, as specified in the constructor.
      *
      * Only the upper Hessenberg part of \p result is updated, the
      * rest is not touched.  See MatrixBase::sqrt() for details on
      * how this computation is implemented.
      */
    template <typename ResultType> void compute(ResultType &result);    
    
  private:
    typedef typename MatrixType::Index Index;
    typedef typename MatrixType::Scalar Scalar;
    
    void computeDiagonalPartOfSqrt(MatrixType& sqrtT, const MatrixType& T);
    void computeOffDiagonalPartOfSqrt(MatrixType& sqrtT, const MatrixType& T);
    void compute2x2diagonalBlock(MatrixType& sqrtT, const MatrixType& T, typename MatrixType::Index i);
    void compute1x1offDiagonalBlock(MatrixType& sqrtT, const MatrixType& T, 
				  typename MatrixType::Index i, typename MatrixType::Index j);
    void compute1x2offDiagonalBlock(MatrixType& sqrtT, const MatrixType& T, 
				  typename MatrixType::Index i, typename MatrixType::Index j);
    void compute2x1offDiagonalBlock(MatrixType& sqrtT, const MatrixType& T, 
				  typename MatrixType::Index i, typename MatrixType::Index j);
    void compute2x2offDiagonalBlock(MatrixType& sqrtT, const MatrixType& T, 
				  typename MatrixType::Index i, typename MatrixType::Index j);
  
    template <typename SmallMatrixType>
    static void solveAuxiliaryEquation(SmallMatrixType& X, const SmallMatrixType& A, 
				     const SmallMatrixType& B, const SmallMatrixType& C);
  
    const MatrixType& m_A;
};

template <typename MatrixType>
template <typename ResultType> 
void MatrixSquareRootQuasiTriangular<MatrixType>::compute(ResultType &result)
{
  result.resize(m_A.rows(), m_A.cols());
  computeDiagonalPartOfSqrt(result, m_A);
  computeOffDiagonalPartOfSqrt(result, m_A);
}

// pre:  T is quasi-upper-triangular and sqrtT is a zero matrix of the same size
// post: the diagonal blocks of sqrtT are the square roots of the diagonal blocks of T
template <typename MatrixType>
void MatrixSquareRootQuasiTriangular<MatrixType>::computeDiagonalPartOfSqrt(MatrixType& sqrtT, 
									  const MatrixType& T)
{
  using std::sqrt;
  const Index size = m_A.rows();
  for (Index i = 0; i < size; i++) {
    if (i == size - 1 || T.coeff(i+1, i) == 0) {
      eigen_assert(T(i,i) >= 0);
      sqrtT.coeffRef(i,i) = sqrt(T.coeff(i,i));
    }
    else {
      compute2x2diagonalBlock(sqrtT, T, i);
      ++i;
    }
  }
}

// pre:  T is quasi-upper-triangular and diagonal blocks of sqrtT are square root of diagonal blocks of T.
// post: sqrtT is the square root of T.
template <typename MatrixType>
void MatrixSquareRootQuasiTriangular<MatrixType>::computeOffDiagonalPartOfSqrt(MatrixType& sqrtT, 
									     const MatrixType& T)
{
  const Index size = m_A.rows();
  for (Index j = 1; j < size; j++) {
      if (T.coeff(j, j-1) != 0)  // if T(j-1:j, j-1:j) is a 2-by-2 block
	continue;
    for (Index i = j-1; i >= 0; i--) {
      if (i > 0 && T.coeff(i, i-1) != 0)  // if T(i-1:i, i-1:i) is a 2-by-2 block
	continue;
      bool iBlockIs2x2 = (i < size - 1) && (T.coeff(i+1, i) != 0);
      bool jBlockIs2x2 = (j < size - 1) && (T.coeff(j+1, j) != 0);
      if (iBlockIs2x2 && jBlockIs2x2) 
	compute2x2offDiagonalBlock(sqrtT, T, i, j);
      else if (iBlockIs2x2 && !jBlockIs2x2) 
	compute2x1offDiagonalBlock(sqrtT, T, i, j);
      else if (!iBlockIs2x2 && jBlockIs2x2) 
	compute1x2offDiagonalBlock(sqrtT, T, i, j);
      else if (!iBlockIs2x2 && !jBlockIs2x2) 
	compute1x1offDiagonalBlock(sqrtT, T, i, j);
    }
  }
}

// pre:  T.block(i,i,2,2) has complex conjugate eigenvalues
// post: sqrtT.block(i,i,2,2) is square root of T.block(i,i,2,2)
template <typename MatrixType>
void MatrixSquareRootQuasiTriangular<MatrixType>
     ::compute2x2diagonalBlock(MatrixType& sqrtT, const MatrixType& T, typename MatrixType::Index i)
{
  // TODO: This case (2-by-2 blocks with complex conjugate eigenvalues) is probably hidden somewhere
  //       in EigenSolver. If we expose it, we could call it directly from here.
  Matrix<Scalar,2,2> block = T.template block<2,2>(i,i);
  EigenSolver<Matrix<Scalar,2,2> > es(block);
  sqrtT.template block<2,2>(i,i)
    = (es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal() * es.eigenvectors().inverse()).real();
}

// pre:  block structure of T is such that (i,j) is a 1x1 block,
//       all blocks of sqrtT to left of and below (i,j) are correct
// post: sqrtT(i,j) has the correct value
template <typename MatrixType>
void MatrixSquareRootQuasiTriangular<MatrixType>
     ::compute1x1offDiagonalBlock(MatrixType& sqrtT, const MatrixType& T, 
				  typename MatrixType::Index i, typename MatrixType::Index j)
{
  Scalar tmp = (sqrtT.row(i).segment(i+1,j-i-1) * sqrtT.col(j).segment(i+1,j-i-1)).value();
  sqrtT.coeffRef(i,j) = (T.coeff(i,j) - tmp) / (sqrtT.coeff(i,i) + sqrtT.coeff(j,j));
}

// similar to compute1x1offDiagonalBlock()
template <typename MatrixType>
void MatrixSquareRootQuasiTriangular<MatrixType>
     ::compute1x2offDiagonalBlock(MatrixType& sqrtT, const MatrixType& T, 
				  typename MatrixType::Index i, typename MatrixType::Index j)
{
  Matrix<Scalar,1,2> rhs = T.template block<1,2>(i,j);
  if (j-i > 1)
    rhs -= sqrtT.block(i, i+1, 1, j-i-1) * sqrtT.block(i+1, j, j-i-1, 2);
  Matrix<Scalar,2,2> A = sqrtT.coeff(i,i) * Matrix<Scalar,2,2>::Identity();
  A += sqrtT.template block<2,2>(j,j).transpose();
  sqrtT.template block<1,2>(i,j).transpose() = A.fullPivLu().solve(rhs.transpose());
}

// similar to compute1x1offDiagonalBlock()
template <typename MatrixType>
void MatrixSquareRootQuasiTriangular<MatrixType>
     ::compute2x1offDiagonalBlock(MatrixType& sqrtT, const MatrixType& T, 
				  typename MatrixType::Index i, typename MatrixType::Index j)
{
  Matrix<Scalar,2,1> rhs = T.template block<2,1>(i,j);
  if (j-i > 2)
    rhs -= sqrtT.block(i, i+2, 2, j-i-2) * sqrtT.block(i+2, j, j-i-2, 1);
  Matrix<Scalar,2,2> A = sqrtT.coeff(j,j) * Matrix<Scalar,2,2>::Identity();
  A += sqrtT.template block<2,2>(i,i);
  sqrtT.template block<2,1>(i,j) = A.fullPivLu().solve(rhs);
}

// similar to compute1x1offDiagonalBlock()
template <typename MatrixType>
void MatrixSquareRootQuasiTriangular<MatrixType>
     ::compute2x2offDiagonalBlock(MatrixType& sqrtT, const MatrixType& T, 
				  typename MatrixType::Index i, typename MatrixType::Index j)
{
  Matrix<Scalar,2,2> A = sqrtT.template block<2,2>(i,i);
  Matrix<Scalar,2,2> B = sqrtT.template block<2,2>(j,j);
  Matrix<Scalar,2,2> C = T.template block<2,2>(i,j);
  if (j-i > 2)
    C -= sqrtT.block(i, i+2, 2, j-i-2) * sqrtT.block(i+2, j, j-i-2, 2);
  Matrix<Scalar,2,2> X;
  solveAuxiliaryEquation(X, A, B, C);
  sqrtT.template block<2,2>(i,j) = X;
}

// solves the equation A X + X B = C where all matrices are 2-by-2
template <typename MatrixType>
template <typename SmallMatrixType>
void MatrixSquareRootQuasiTriangular<MatrixType>
     ::solveAuxiliaryEquation(SmallMatrixType& X, const SmallMatrixType& A,
			      const SmallMatrixType& B, const SmallMatrixType& C)
{
  EIGEN_STATIC_ASSERT((internal::is_same<SmallMatrixType, Matrix<Scalar,2,2> >::value),
		      EIGEN_INTERNAL_ERROR_PLEASE_FILE_A_BUG_REPORT);

  Matrix<Scalar,4,4> coeffMatrix = Matrix<Scalar,4,4>::Zero();
  coeffMatrix.coeffRef(0,0) = A.coeff(0,0) + B.coeff(0,0);
  coeffMatrix.coeffRef(1,1) = A.coeff(0,0) + B.coeff(1,1);
  coeffMatrix.coeffRef(2,2) = A.coeff(1,1) + B.coeff(0,0);
  coeffMatrix.coeffRef(3,3) = A.coeff(1,1) + B.coeff(1,1);
  coeffMatrix.coeffRef(0,1) = B.coeff(1,0);
  coeffMatrix.coeffRef(0,2) = A.coeff(0,1);
  coeffMatrix.coeffRef(1,0) = B.coeff(0,1);
  coeffMatrix.coeffRef(1,3) = A.coeff(0,1);
  coeffMatrix.coeffRef(2,0) = A.coeff(1,0);
  coeffMatrix.coeffRef(2,3) = B.coeff(1,0);
  coeffMatrix.coeffRef(3,1) = A.coeff(1,0);
  coeffMatrix.coeffRef(3,2) = B.coeff(0,1);
  
  Matrix<Scalar,4,1> rhs;
  rhs.coeffRef(0) = C.coeff(0,0);
  rhs.coeffRef(1) = C.coeff(0,1);
  rhs.coeffRef(2) = C.coeff(1,0);
  rhs.coeffRef(3) = C.coeff(1,1);
  
  Matrix<Scalar,4,1> result;
  result = coeffMatrix.fullPivLu().solve(rhs);

  X.coeffRef(0,0) = result.coeff(0);
  X.coeffRef(0,1) = result.coeff(1);
  X.coeffRef(1,0) = result.coeff(2);
  X.coeffRef(1,1) = result.coeff(3);
}


/** \ingroup MatrixFunctions_Module
  * \brief Class for computing matrix square roots of upper triangular matrices.
  * \tparam  MatrixType  type of the argument of the matrix square root,
  *                      expected to be an instantiation of the Matrix class template.
  *
  * This class computes the square root of the upper triangular matrix
  * stored in the upper triangular part (including the diagonal) of
  * the matrix passed to the constructor.
  *
  * \sa MatrixSquareRoot, MatrixSquareRootQuasiTriangular
  */
template <typename MatrixType>
class MatrixSquareRootTriangular
{
  public:
    MatrixSquareRootTriangular(const MatrixType& A) 
      : m_A(A) 
    {
      eigen_assert(A.rows() == A.cols());
    }

    /** \brief Compute the matrix square root
      *
      * \param[out] result  square root of \p A, as specified in the constructor.
      *
      * Only the upper triangular part (including the diagonal) of 
      * \p result is updated, the rest is not touched.  See
      * MatrixBase::sqrt() for details on how this computation is
      * implemented.
      */
    template <typename ResultType> void compute(ResultType &result);    

 private:
    const MatrixType& m_A;
};

template <typename MatrixType>
template <typename ResultType> 
void MatrixSquareRootTriangular<MatrixType>::compute(ResultType &result)
{
  using std::sqrt;

  // Compute square root of m_A and store it in upper triangular part of result
  // This uses that the square root of triangular matrices can be computed directly.
  result.resize(m_A.rows(), m_A.cols());
  typedef typename MatrixType::Index Index;
  for (Index i = 0; i < m_A.rows(); i++) {
    result.coeffRef(i,i) = sqrt(m_A.coeff(i,i));
  }
  for (Index j = 1; j < m_A.cols(); j++) {
    for (Index i = j-1; i >= 0; i--) {
      typedef typename MatrixType::Scalar Scalar;
      // if i = j-1, then segment has length 0 so tmp = 0
      Scalar tmp = (result.row(i).segment(i+1,j-i-1) * result.col(j).segment(i+1,j-i-1)).value();
      // denominator may be zero if original matrix is singular
      result.coeffRef(i,j) = (m_A.coeff(i,j) - tmp) / (result.coeff(i,i) + result.coeff(j,j));
    }
  }
}


/** \ingroup MatrixFunctions_Module
  * \brief Class for computing matrix square roots of general matrices.
  * \tparam  MatrixType  type of the argument of the matrix square root,
  *                      expected to be an instantiation of the Matrix class template.
  *
  * \sa MatrixSquareRootTriangular, MatrixSquareRootQuasiTriangular, MatrixBase::sqrt()
  */
template <typename MatrixType, int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex>
class MatrixSquareRoot
{
  public:

    /** \brief Constructor. 
      *
      * \param[in]  A  matrix whose square root is to be computed.
      *
      * The class stores a reference to \p A, so it should not be
      * changed (or destroyed) before compute() is called.
      */
    MatrixSquareRoot(const MatrixType& A); 
    
    /** \brief Compute the matrix square root
      *
      * \param[out] result  square root of \p A, as specified in the constructor.
      *
      * See MatrixBase::sqrt() for details on how this computation is
      * implemented.
      */
    template <typename ResultType> void compute(ResultType &result);    
};


// ********** Partial specialization for real matrices **********

template <typename MatrixType>
class MatrixSquareRoot<MatrixType, 0>
{
  public:

    MatrixSquareRoot(const MatrixType& A) 
      : m_A(A) 
    {  
      eigen_assert(A.rows() == A.cols());
    }
  
    template <typename ResultType> void compute(ResultType &result)
    {
      // Compute Schur decomposition of m_A
      const RealSchur<MatrixType> schurOfA(m_A);  
      const MatrixType& T = schurOfA.matrixT();
      const MatrixType& U = schurOfA.matrixU();
    
      // Compute square root of T
      MatrixType sqrtT = MatrixType::Zero(m_A.rows(), m_A.cols());
      MatrixSquareRootQuasiTriangular<MatrixType>(T).compute(sqrtT);
    
      // Compute square root of m_A
      result = U * sqrtT * U.adjoint();
    }
    
  private:
    const MatrixType& m_A;
};


// ********** Partial specialization for complex matrices **********

template <typename MatrixType>
class MatrixSquareRoot<MatrixType, 1>
{
  public:

    MatrixSquareRoot(const MatrixType& A) 
      : m_A(A) 
    {  
      eigen_assert(A.rows() == A.cols());
    }
  
    template <typename ResultType> void compute(ResultType &result)
    {
      // Compute Schur decomposition of m_A
      const ComplexSchur<MatrixType> schurOfA(m_A);  
      const MatrixType& T = schurOfA.matrixT();
      const MatrixType& U = schurOfA.matrixU();
    
      // Compute square root of T
      MatrixType sqrtT;
      MatrixSquareRootTriangular<MatrixType>(T).compute(sqrtT);
    
      // Compute square root of m_A
      result = U * (sqrtT.template triangularView<Upper>() * U.adjoint());
    }
    
  private:
    const MatrixType& m_A;
};


/** \ingroup MatrixFunctions_Module
  *
  * \brief Proxy for the matrix square root of some matrix (expression).
  *
  * \tparam Derived  Type of the argument to the matrix square root.
  *
  * This class holds the argument to the matrix square root until it
  * is assigned or evaluated for some other reason (so the argument
  * should not be changed in the meantime). It is the return type of
  * MatrixBase::sqrt() and most of the time this is the only way it is
  * used.
  */
template<typename Derived> class MatrixSquareRootReturnValue
: public ReturnByValue<MatrixSquareRootReturnValue<Derived> >
{
    typedef typename Derived::Index Index;
  public:
    /** \brief Constructor.
      *
      * \param[in]  src  %Matrix (expression) forming the argument of the
      * matrix square root.
      */
    MatrixSquareRootReturnValue(const Derived& src) : m_src(src) { }

    /** \brief Compute the matrix square root.
      *
      * \param[out]  result  the matrix square root of \p src in the
      * constructor.
      */
    template <typename ResultType>
    inline void evalTo(ResultType& result) const
    {
      const typename Derived::PlainObject srcEvaluated = m_src.eval();
      MatrixSquareRoot<typename Derived::PlainObject> me(srcEvaluated);
      me.compute(result);
    }

    Index rows() const { return m_src.rows(); }
    Index cols() const { return m_src.cols(); }

  protected:
    const Derived& m_src;
  private:
    MatrixSquareRootReturnValue& operator=(const MatrixSquareRootReturnValue&);
};

namespace internal {
template<typename Derived>
struct traits<MatrixSquareRootReturnValue<Derived> >
{
  typedef typename Derived::PlainObject ReturnType;
};
}

template <typename Derived>
const MatrixSquareRootReturnValue<Derived> MatrixBase<Derived>::sqrt() const
{
  eigen_assert(rows() == cols());
  return MatrixSquareRootReturnValue<Derived>(derived());
}

} // end namespace Eigen

#endif // EIGEN_MATRIX_FUNCTION
