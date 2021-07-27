// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_FUNCTION_ATOMIC
#define EIGEN_MATRIX_FUNCTION_ATOMIC

namespace Eigen { 

/** \ingroup MatrixFunctions_Module
  * \class MatrixFunctionAtomic
  * \brief Helper class for computing matrix functions of atomic matrices.
  *
  * \internal
  * Here, an atomic matrix is a triangular matrix whose diagonal
  * entries are close to each other.
  */
template <typename MatrixType>
class MatrixFunctionAtomic
{
  public:

    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename internal::stem_function<Scalar>::type StemFunction;
    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

    /** \brief Constructor
      * \param[in]  f  matrix function to compute.
      */
    MatrixFunctionAtomic(StemFunction f) : m_f(f) { }

    /** \brief Compute matrix function of atomic matrix
      * \param[in]  A  argument of matrix function, should be upper triangular and atomic
      * \returns  f(A), the matrix function evaluated at the given matrix
      */
    MatrixType compute(const MatrixType& A);

  private:

    // Prevent copying
    MatrixFunctionAtomic(const MatrixFunctionAtomic&);
    MatrixFunctionAtomic& operator=(const MatrixFunctionAtomic&);

    void computeMu();
    bool taylorConverged(Index s, const MatrixType& F, const MatrixType& Fincr, const MatrixType& P);

    /** \brief Pointer to scalar function */
    StemFunction* m_f;

    /** \brief Size of matrix function */
    Index m_Arows;

    /** \brief Mean of eigenvalues */
    Scalar m_avgEival;

    /** \brief Argument shifted by mean of eigenvalues */
    MatrixType m_Ashifted;

    /** \brief Constant used to determine whether Taylor series has converged */
    RealScalar m_mu;
};

template <typename MatrixType>
MatrixType MatrixFunctionAtomic<MatrixType>::compute(const MatrixType& A)
{
  // TODO: Use that A is upper triangular
  m_Arows = A.rows();
  m_avgEival = A.trace() / Scalar(RealScalar(m_Arows));
  m_Ashifted = A - m_avgEival * MatrixType::Identity(m_Arows, m_Arows);
  computeMu();
  MatrixType F = m_f(m_avgEival, 0) * MatrixType::Identity(m_Arows, m_Arows);
  MatrixType P = m_Ashifted;
  MatrixType Fincr;
  for (Index s = 1; s < 1.1 * m_Arows + 10; s++) { // upper limit is fairly arbitrary
    Fincr = m_f(m_avgEival, static_cast<int>(s)) * P;
    F += Fincr;
    P = Scalar(RealScalar(1.0/(s + 1))) * P * m_Ashifted;
    if (taylorConverged(s, F, Fincr, P)) {
      return F;
    }
  }
  eigen_assert("Taylor series does not converge" && 0);
  return F;
}

/** \brief Compute \c m_mu. */
template <typename MatrixType>
void MatrixFunctionAtomic<MatrixType>::computeMu()
{
  const MatrixType N = MatrixType::Identity(m_Arows, m_Arows) - m_Ashifted;
  VectorType e = VectorType::Ones(m_Arows);
  N.template triangularView<Upper>().solveInPlace(e);
  m_mu = e.cwiseAbs().maxCoeff();
}

/** \brief Determine whether Taylor series has converged */
template <typename MatrixType>
bool MatrixFunctionAtomic<MatrixType>::taylorConverged(Index s, const MatrixType& F,
						       const MatrixType& Fincr, const MatrixType& P)
{
  const Index n = F.rows();
  const RealScalar F_norm = F.cwiseAbs().rowwise().sum().maxCoeff();
  const RealScalar Fincr_norm = Fincr.cwiseAbs().rowwise().sum().maxCoeff();
  if (Fincr_norm < NumTraits<Scalar>::epsilon() * F_norm) {
    RealScalar delta = 0;
    RealScalar rfactorial = 1;
    for (Index r = 0; r < n; r++) {
      RealScalar mx = 0;
      for (Index i = 0; i < n; i++)
        mx = (std::max)(mx, std::abs(m_f(m_Ashifted(i, i) + m_avgEival, static_cast<int>(s+r))));
      if (r != 0)
        rfactorial *= RealScalar(r);
      delta = (std::max)(delta, mx / rfactorial);
    }
    const RealScalar P_norm = P.cwiseAbs().rowwise().sum().maxCoeff();
    if (m_mu * delta * P_norm < NumTraits<Scalar>::epsilon() * F_norm)
      return true;
  }
  return false;
}

} // end namespace Eigen

#endif // EIGEN_MATRIX_FUNCTION_ATOMIC
