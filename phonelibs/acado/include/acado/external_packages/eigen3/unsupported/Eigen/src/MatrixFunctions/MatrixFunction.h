// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_FUNCTION
#define EIGEN_MATRIX_FUNCTION

#include "StemFunction.h"
#include "MatrixFunctionAtomic.h"


namespace Eigen { 

/** \ingroup MatrixFunctions_Module
  * \brief Class for computing matrix functions.
  * \tparam  MatrixType  type of the argument of the matrix function,
  *                      expected to be an instantiation of the Matrix class template.
  * \tparam  AtomicType  type for computing matrix function of atomic blocks.
  * \tparam  IsComplex   used internally to select correct specialization.
  *
  * This class implements the Schur-Parlett algorithm for computing matrix functions. The spectrum of the
  * matrix is divided in clustered of eigenvalues that lies close together. This class delegates the
  * computation of the matrix function on every block corresponding to these clusters to an object of type
  * \p AtomicType and uses these results to compute the matrix function of the whole matrix. The class
  * \p AtomicType should have a \p compute() member function for computing the matrix function of a block.
  *
  * \sa class MatrixFunctionAtomic, class MatrixLogarithmAtomic
  */
template <typename MatrixType, 
	  typename AtomicType,  
          int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex>
class MatrixFunction
{  
  public:

    /** \brief Constructor. 
      *
      * \param[in]  A       argument of matrix function, should be a square matrix.
      * \param[in]  atomic  class for computing matrix function of atomic blocks.
      *
      * The class stores references to \p A and \p atomic, so they should not be
      * changed (or destroyed) before compute() is called.
      */
    MatrixFunction(const MatrixType& A, AtomicType& atomic);

    /** \brief Compute the matrix function.
      *
      * \param[out] result  the function \p f applied to \p A, as
      * specified in the constructor.
      *
      * See MatrixBase::matrixFunction() for details on how this computation
      * is implemented.
      */
    template <typename ResultType> 
    void compute(ResultType &result);    
};


/** \internal \ingroup MatrixFunctions_Module 
  * \brief Partial specialization of MatrixFunction for real matrices
  */
template <typename MatrixType, typename AtomicType>
class MatrixFunction<MatrixType, AtomicType, 0>
{  
  private:

    typedef internal::traits<MatrixType> Traits;
    typedef typename Traits::Scalar Scalar;
    static const int Rows = Traits::RowsAtCompileTime;
    static const int Cols = Traits::ColsAtCompileTime;
    static const int Options = MatrixType::Options;
    static const int MaxRows = Traits::MaxRowsAtCompileTime;
    static const int MaxCols = Traits::MaxColsAtCompileTime;

    typedef std::complex<Scalar> ComplexScalar;
    typedef Matrix<ComplexScalar, Rows, Cols, Options, MaxRows, MaxCols> ComplexMatrix;

  public:

    /** \brief Constructor. 
      *
      * \param[in]  A       argument of matrix function, should be a square matrix.
      * \param[in]  atomic  class for computing matrix function of atomic blocks.
      */
    MatrixFunction(const MatrixType& A, AtomicType& atomic) : m_A(A), m_atomic(atomic) { }

    /** \brief Compute the matrix function.
      *
      * \param[out] result  the function \p f applied to \p A, as
      * specified in the constructor.
      *
      * This function converts the real matrix \c A to a complex matrix,
      * uses MatrixFunction<MatrixType,1> and then converts the result back to
      * a real matrix.
      */
    template <typename ResultType>
    void compute(ResultType& result) 
    {
      ComplexMatrix CA = m_A.template cast<ComplexScalar>();
      ComplexMatrix Cresult;
      MatrixFunction<ComplexMatrix, AtomicType> mf(CA, m_atomic);
      mf.compute(Cresult);
      result = Cresult.real();
    }

  private:
    typename internal::nested<MatrixType>::type m_A; /**< \brief Reference to argument of matrix function. */
    AtomicType& m_atomic; /**< \brief Class for computing matrix function of atomic blocks. */

    MatrixFunction& operator=(const MatrixFunction&);
};

      
/** \internal \ingroup MatrixFunctions_Module 
  * \brief Partial specialization of MatrixFunction for complex matrices
  */
template <typename MatrixType, typename AtomicType>
class MatrixFunction<MatrixType, AtomicType, 1>
{
  private:

    typedef internal::traits<MatrixType> Traits;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    static const int RowsAtCompileTime = Traits::RowsAtCompileTime;
    static const int ColsAtCompileTime = Traits::ColsAtCompileTime;
    static const int Options = MatrixType::Options;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef Matrix<Scalar, Traits::RowsAtCompileTime, 1> VectorType;
    typedef Matrix<Index, Traits::RowsAtCompileTime, 1> IntVectorType;
    typedef Matrix<Index, Dynamic, 1> DynamicIntVectorType;
    typedef std::list<Scalar> Cluster;
    typedef std::list<Cluster> ListOfClusters;
    typedef Matrix<Scalar, Dynamic, Dynamic, Options, RowsAtCompileTime, ColsAtCompileTime> DynMatrixType;

  public:

    MatrixFunction(const MatrixType& A, AtomicType& atomic);
    template <typename ResultType> void compute(ResultType& result);

  private:

    void computeSchurDecomposition();
    void partitionEigenvalues();
    typename ListOfClusters::iterator findCluster(Scalar key);
    void computeClusterSize();
    void computeBlockStart();
    void constructPermutation();
    void permuteSchur();
    void swapEntriesInSchur(Index index);
    void computeBlockAtomic();
    Block<MatrixType> block(MatrixType& A, Index i, Index j);
    void computeOffDiagonal();
    DynMatrixType solveTriangularSylvester(const DynMatrixType& A, const DynMatrixType& B, const DynMatrixType& C);

    typename internal::nested<MatrixType>::type m_A; /**< \brief Reference to argument of matrix function. */
    AtomicType& m_atomic; /**< \brief Class for computing matrix function of atomic blocks. */
    MatrixType m_T; /**< \brief Triangular part of Schur decomposition */
    MatrixType m_U; /**< \brief Unitary part of Schur decomposition */
    MatrixType m_fT; /**< \brief %Matrix function applied to #m_T */
    ListOfClusters m_clusters; /**< \brief Partition of eigenvalues into clusters of ei'vals "close" to each other */
    DynamicIntVectorType m_eivalToCluster; /**< \brief m_eivalToCluster[i] = j means i-th ei'val is in j-th cluster */
    DynamicIntVectorType m_clusterSize; /**< \brief Number of eigenvalues in each clusters  */
    DynamicIntVectorType m_blockStart; /**< \brief Row index at which block corresponding to i-th cluster starts */
    IntVectorType m_permutation; /**< \brief Permutation which groups ei'vals in the same cluster together */

    /** \brief Maximum distance allowed between eigenvalues to be considered "close".
      *
      * This is morally a \c static \c const \c Scalar, but only
      * integers can be static constant class members in C++. The
      * separation constant is set to 0.1, a value taken from the
      * paper by Davies and Higham. */
    static const RealScalar separation() { return static_cast<RealScalar>(0.1); }

    MatrixFunction& operator=(const MatrixFunction&);
};

/** \brief Constructor. 
 *
 * \param[in]  A       argument of matrix function, should be a square matrix.
 * \param[in]  atomic  class for computing matrix function of atomic blocks.
 */
template <typename MatrixType, typename AtomicType>
MatrixFunction<MatrixType,AtomicType,1>::MatrixFunction(const MatrixType& A, AtomicType& atomic)
  : m_A(A), m_atomic(atomic)
{
  /* empty body */
}

/** \brief Compute the matrix function.
  *
  * \param[out] result  the function \p f applied to \p A, as
  * specified in the constructor.
  */
template <typename MatrixType, typename AtomicType>
template <typename ResultType>
void MatrixFunction<MatrixType,AtomicType,1>::compute(ResultType& result) 
{
  computeSchurDecomposition();
  partitionEigenvalues();
  computeClusterSize();
  computeBlockStart();
  constructPermutation();
  permuteSchur();
  computeBlockAtomic();
  computeOffDiagonal();
  result = m_U * (m_fT.template triangularView<Upper>() * m_U.adjoint());
}

/** \brief Store the Schur decomposition of #m_A in #m_T and #m_U */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::computeSchurDecomposition()
{
  const ComplexSchur<MatrixType> schurOfA(m_A);  
  m_T = schurOfA.matrixT();
  m_U = schurOfA.matrixU();
}

/** \brief Partition eigenvalues in clusters of ei'vals close to each other
  * 
  * This function computes #m_clusters. This is a partition of the
  * eigenvalues of #m_T in clusters, such that
  * # Any eigenvalue in a certain cluster is at most separation() away
  *   from another eigenvalue in the same cluster.
  * # The distance between two eigenvalues in different clusters is
  *   more than separation().
  * The implementation follows Algorithm 4.1 in the paper of Davies
  * and Higham. 
  */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::partitionEigenvalues()
{
  using std::abs;
  const Index rows = m_T.rows();
  VectorType diag = m_T.diagonal(); // contains eigenvalues of A

  for (Index i=0; i<rows; ++i) {
    // Find set containing diag(i), adding a new set if necessary
    typename ListOfClusters::iterator qi = findCluster(diag(i));
    if (qi == m_clusters.end()) {
      Cluster l;
      l.push_back(diag(i));
      m_clusters.push_back(l);
      qi = m_clusters.end();
      --qi;
    }

    // Look for other element to add to the set
    for (Index j=i+1; j<rows; ++j) {
      if (abs(diag(j) - diag(i)) <= separation() && std::find(qi->begin(), qi->end(), diag(j)) == qi->end()) {
        typename ListOfClusters::iterator qj = findCluster(diag(j));
        if (qj == m_clusters.end()) {
          qi->push_back(diag(j));
        } else {
          qi->insert(qi->end(), qj->begin(), qj->end());
          m_clusters.erase(qj);
        }
      }
    }
  }
}

/** \brief Find cluster in #m_clusters containing some value 
  * \param[in] key Value to find
  * \returns Iterator to cluster containing \c key, or
  * \c m_clusters.end() if no cluster in m_clusters contains \c key.
  */
template <typename MatrixType, typename AtomicType>
typename MatrixFunction<MatrixType,AtomicType,1>::ListOfClusters::iterator MatrixFunction<MatrixType,AtomicType,1>::findCluster(Scalar key)
{
  typename Cluster::iterator j;
  for (typename ListOfClusters::iterator i = m_clusters.begin(); i != m_clusters.end(); ++i) {
    j = std::find(i->begin(), i->end(), key);
    if (j != i->end())
      return i;
  }
  return m_clusters.end();
}

/** \brief Compute #m_clusterSize and #m_eivalToCluster using #m_clusters */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::computeClusterSize()
{
  const Index rows = m_T.rows();
  VectorType diag = m_T.diagonal(); 
  const Index numClusters = static_cast<Index>(m_clusters.size());

  m_clusterSize.setZero(numClusters);
  m_eivalToCluster.resize(rows);
  Index clusterIndex = 0;
  for (typename ListOfClusters::const_iterator cluster = m_clusters.begin(); cluster != m_clusters.end(); ++cluster) {
    for (Index i = 0; i < diag.rows(); ++i) {
      if (std::find(cluster->begin(), cluster->end(), diag(i)) != cluster->end()) {
        ++m_clusterSize[clusterIndex];
        m_eivalToCluster[i] = clusterIndex;
      }
    }
    ++clusterIndex;
  }
}

/** \brief Compute #m_blockStart using #m_clusterSize */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::computeBlockStart()
{
  m_blockStart.resize(m_clusterSize.rows());
  m_blockStart(0) = 0;
  for (Index i = 1; i < m_clusterSize.rows(); i++) {
    m_blockStart(i) = m_blockStart(i-1) + m_clusterSize(i-1);
  }
}

/** \brief Compute #m_permutation using #m_eivalToCluster and #m_blockStart */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::constructPermutation()
{
  DynamicIntVectorType indexNextEntry = m_blockStart;
  m_permutation.resize(m_T.rows());
  for (Index i = 0; i < m_T.rows(); i++) {
    Index cluster = m_eivalToCluster[i];
    m_permutation[i] = indexNextEntry[cluster];
    ++indexNextEntry[cluster];
  }
}  

/** \brief Permute Schur decomposition in #m_U and #m_T according to #m_permutation */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::permuteSchur()
{
  IntVectorType p = m_permutation;
  for (Index i = 0; i < p.rows() - 1; i++) {
    Index j;
    for (j = i; j < p.rows(); j++) {
      if (p(j) == i) break;
    }
    eigen_assert(p(j) == i);
    for (Index k = j-1; k >= i; k--) {
      swapEntriesInSchur(k);
      std::swap(p.coeffRef(k), p.coeffRef(k+1));
    }
  }
}

/** \brief Swap rows \a index and \a index+1 in Schur decomposition in #m_U and #m_T */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::swapEntriesInSchur(Index index)
{
  JacobiRotation<Scalar> rotation;
  rotation.makeGivens(m_T(index, index+1), m_T(index+1, index+1) - m_T(index, index));
  m_T.applyOnTheLeft(index, index+1, rotation.adjoint());
  m_T.applyOnTheRight(index, index+1, rotation);
  m_U.applyOnTheRight(index, index+1, rotation);
}  

/** \brief Compute block diagonal part of #m_fT.
  *
  * This routine computes the matrix function applied to the block diagonal part of #m_T, with the blocking
  * given by #m_blockStart. The matrix function of each diagonal block is computed by #m_atomic. The
  * off-diagonal parts of #m_fT are set to zero.
  */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::computeBlockAtomic()
{ 
  m_fT.resize(m_T.rows(), m_T.cols());
  m_fT.setZero();
  for (Index i = 0; i < m_clusterSize.rows(); ++i) {
    block(m_fT, i, i) = m_atomic.compute(block(m_T, i, i));
  }
}

/** \brief Return block of matrix according to blocking given by #m_blockStart */
template <typename MatrixType, typename AtomicType>
Block<MatrixType> MatrixFunction<MatrixType,AtomicType,1>::block(MatrixType& A, Index i, Index j)
{
  return A.block(m_blockStart(i), m_blockStart(j), m_clusterSize(i), m_clusterSize(j));
}

/** \brief Compute part of #m_fT above block diagonal.
  *
  * This routine assumes that the block diagonal part of #m_fT (which
  * equals the matrix function applied to #m_T) has already been computed and computes
  * the part above the block diagonal. The part below the diagonal is
  * zero, because #m_T is upper triangular.
  */
template <typename MatrixType, typename AtomicType>
void MatrixFunction<MatrixType,AtomicType,1>::computeOffDiagonal()
{ 
  for (Index diagIndex = 1; diagIndex < m_clusterSize.rows(); diagIndex++) {
    for (Index blockIndex = 0; blockIndex < m_clusterSize.rows() - diagIndex; blockIndex++) {
      // compute (blockIndex, blockIndex+diagIndex) block
      DynMatrixType A = block(m_T, blockIndex, blockIndex);
      DynMatrixType B = -block(m_T, blockIndex+diagIndex, blockIndex+diagIndex);
      DynMatrixType C = block(m_fT, blockIndex, blockIndex) * block(m_T, blockIndex, blockIndex+diagIndex);
      C -= block(m_T, blockIndex, blockIndex+diagIndex) * block(m_fT, blockIndex+diagIndex, blockIndex+diagIndex);
      for (Index k = blockIndex + 1; k < blockIndex + diagIndex; k++) {
	C += block(m_fT, blockIndex, k) * block(m_T, k, blockIndex+diagIndex);
	C -= block(m_T, blockIndex, k) * block(m_fT, k, blockIndex+diagIndex);
      }
      block(m_fT, blockIndex, blockIndex+diagIndex) = solveTriangularSylvester(A, B, C);
    }
  }
}

/** \brief Solve a triangular Sylvester equation AX + XB = C 
  *
  * \param[in]  A  the matrix A; should be square and upper triangular
  * \param[in]  B  the matrix B; should be square and upper triangular
  * \param[in]  C  the matrix C; should have correct size.
  *
  * \returns the solution X.
  *
  * If A is m-by-m and B is n-by-n, then both C and X are m-by-n. 
  * The (i,j)-th component of the Sylvester equation is
  * \f[ 
  *     \sum_{k=i}^m A_{ik} X_{kj} + \sum_{k=1}^j X_{ik} B_{kj} = C_{ij}. 
  * \f]
  * This can be re-arranged to yield:
  * \f[ 
  *     X_{ij} = \frac{1}{A_{ii} + B_{jj}} \Bigl( C_{ij}
  *     - \sum_{k=i+1}^m A_{ik} X_{kj} - \sum_{k=1}^{j-1} X_{ik} B_{kj} \Bigr).
  * \f]
  * It is assumed that A and B are such that the numerator is never
  * zero (otherwise the Sylvester equation does not have a unique
  * solution). In that case, these equations can be evaluated in the
  * order \f$ i=m,\ldots,1 \f$ and \f$ j=1,\ldots,n \f$.
  */
template <typename MatrixType, typename AtomicType>
typename MatrixFunction<MatrixType,AtomicType,1>::DynMatrixType MatrixFunction<MatrixType,AtomicType,1>::solveTriangularSylvester(
  const DynMatrixType& A, 
  const DynMatrixType& B, 
  const DynMatrixType& C)
{
  eigen_assert(A.rows() == A.cols());
  eigen_assert(A.isUpperTriangular());
  eigen_assert(B.rows() == B.cols());
  eigen_assert(B.isUpperTriangular());
  eigen_assert(C.rows() == A.rows());
  eigen_assert(C.cols() == B.rows());

  Index m = A.rows();
  Index n = B.rows();
  DynMatrixType X(m, n);

  for (Index i = m - 1; i >= 0; --i) {
    for (Index j = 0; j < n; ++j) {

      // Compute AX = \sum_{k=i+1}^m A_{ik} X_{kj}
      Scalar AX;
      if (i == m - 1) {
	AX = 0; 
      } else {
	Matrix<Scalar,1,1> AXmatrix = A.row(i).tail(m-1-i) * X.col(j).tail(m-1-i);
	AX = AXmatrix(0,0);
      }

      // Compute XB = \sum_{k=1}^{j-1} X_{ik} B_{kj}
      Scalar XB;
      if (j == 0) {
	XB = 0; 
      } else {
	Matrix<Scalar,1,1> XBmatrix = X.row(i).head(j) * B.col(j).head(j);
	XB = XBmatrix(0,0);
      }

      X(i,j) = (C(i,j) - AX - XB) / (A(i,i) + B(j,j));
    }
  }
  return X;
}

/** \ingroup MatrixFunctions_Module
  *
  * \brief Proxy for the matrix function of some matrix (expression).
  *
  * \tparam Derived  Type of the argument to the matrix function.
  *
  * This class holds the argument to the matrix function until it is
  * assigned or evaluated for some other reason (so the argument
  * should not be changed in the meantime). It is the return type of
  * matrixBase::matrixFunction() and related functions and most of the
  * time this is the only way it is used.
  */
template<typename Derived> class MatrixFunctionReturnValue
: public ReturnByValue<MatrixFunctionReturnValue<Derived> >
{
  public:

    typedef typename Derived::Scalar Scalar;
    typedef typename Derived::Index Index;
    typedef typename internal::stem_function<Scalar>::type StemFunction;

   /** \brief Constructor.
      *
      * \param[in] A  %Matrix (expression) forming the argument of the
      * matrix function.
      * \param[in] f  Stem function for matrix function under consideration.
      */
    MatrixFunctionReturnValue(const Derived& A, StemFunction f) : m_A(A), m_f(f) { }

    /** \brief Compute the matrix function.
      *
      * \param[out] result \p f applied to \p A, where \p f and \p A
      * are as in the constructor.
      */
    template <typename ResultType>
    inline void evalTo(ResultType& result) const
    {
      typedef typename Derived::PlainObject PlainObject;
      typedef internal::traits<PlainObject> Traits;
      static const int RowsAtCompileTime = Traits::RowsAtCompileTime;
      static const int ColsAtCompileTime = Traits::ColsAtCompileTime;
      static const int Options = PlainObject::Options;
      typedef std::complex<typename NumTraits<Scalar>::Real> ComplexScalar;
      typedef Matrix<ComplexScalar, Dynamic, Dynamic, Options, RowsAtCompileTime, ColsAtCompileTime> DynMatrixType;
      typedef MatrixFunctionAtomic<DynMatrixType> AtomicType;
      AtomicType atomic(m_f);

      const PlainObject Aevaluated = m_A.eval();
      MatrixFunction<PlainObject, AtomicType> mf(Aevaluated, atomic);
      mf.compute(result);
    }

    Index rows() const { return m_A.rows(); }
    Index cols() const { return m_A.cols(); }

  private:
    typename internal::nested<Derived>::type m_A;
    StemFunction *m_f;

    MatrixFunctionReturnValue& operator=(const MatrixFunctionReturnValue&);
};

namespace internal {
template<typename Derived>
struct traits<MatrixFunctionReturnValue<Derived> >
{
  typedef typename Derived::PlainObject ReturnType;
};
}


/********** MatrixBase methods **********/


template <typename Derived>
const MatrixFunctionReturnValue<Derived> MatrixBase<Derived>::matrixFunction(typename internal::stem_function<typename internal::traits<Derived>::Scalar>::type f) const
{
  eigen_assert(rows() == cols());
  return MatrixFunctionReturnValue<Derived>(derived(), f);
}

template <typename Derived>
const MatrixFunctionReturnValue<Derived> MatrixBase<Derived>::sin() const
{
  eigen_assert(rows() == cols());
  typedef typename internal::stem_function<Scalar>::ComplexScalar ComplexScalar;
  return MatrixFunctionReturnValue<Derived>(derived(), StdStemFunctions<ComplexScalar>::sin);
}

template <typename Derived>
const MatrixFunctionReturnValue<Derived> MatrixBase<Derived>::cos() const
{
  eigen_assert(rows() == cols());
  typedef typename internal::stem_function<Scalar>::ComplexScalar ComplexScalar;
  return MatrixFunctionReturnValue<Derived>(derived(), StdStemFunctions<ComplexScalar>::cos);
}

template <typename Derived>
const MatrixFunctionReturnValue<Derived> MatrixBase<Derived>::sinh() const
{
  eigen_assert(rows() == cols());
  typedef typename internal::stem_function<Scalar>::ComplexScalar ComplexScalar;
  return MatrixFunctionReturnValue<Derived>(derived(), StdStemFunctions<ComplexScalar>::sinh);
}

template <typename Derived>
const MatrixFunctionReturnValue<Derived> MatrixBase<Derived>::cosh() const
{
  eigen_assert(rows() == cols());
  typedef typename internal::stem_function<Scalar>::ComplexScalar ComplexScalar;
  return MatrixFunctionReturnValue<Derived>(derived(), StdStemFunctions<ComplexScalar>::cosh);
}

} // end namespace Eigen

#endif // EIGEN_MATRIX_FUNCTION
