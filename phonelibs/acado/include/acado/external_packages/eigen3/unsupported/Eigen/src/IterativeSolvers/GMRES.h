// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Kolja Brix <brix@igpm.rwth-aaachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GMRES_H
#define EIGEN_GMRES_H

namespace Eigen { 

namespace internal {

/**
 * Generalized Minimal Residual Algorithm based on the
 * Arnoldi algorithm implemented with Householder reflections.
 *
 * Parameters:
 *  \param mat       matrix of linear system of equations
 *  \param Rhs       right hand side vector of linear system of equations
 *  \param x         on input: initial guess, on output: solution
 *  \param precond   preconditioner used
 *  \param iters     on input: maximum number of iterations to perform
 *                   on output: number of iterations performed
 *  \param restart   number of iterations for a restart
 *  \param tol_error on input: residual tolerance
 *                   on output: residuum achieved
 *
 * \sa IterativeMethods::bicgstab() 
 *  
 *
 * For references, please see:
 *
 * Saad, Y. and Schultz, M. H.
 * GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems.
 * SIAM J.Sci.Stat.Comp. 7, 1986, pp. 856 - 869.
 *
 * Saad, Y.
 * Iterative Methods for Sparse Linear Systems.
 * Society for Industrial and Applied Mathematics, Philadelphia, 2003.
 *
 * Walker, H. F.
 * Implementations of the GMRES method.
 * Comput.Phys.Comm. 53, 1989, pp. 311 - 320.
 *
 * Walker, H. F.
 * Implementation of the GMRES Method using Householder Transformations.
 * SIAM J.Sci.Stat.Comp. 9, 1988, pp. 152 - 163.
 *
 */
template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
bool gmres(const MatrixType & mat, const Rhs & rhs, Dest & x, const Preconditioner & precond,
		int &iters, const int &restart, typename Dest::RealScalar & tol_error) {

	using std::sqrt;
	using std::abs;

	typedef typename Dest::RealScalar RealScalar;
	typedef typename Dest::Scalar Scalar;
	typedef Matrix < Scalar, Dynamic, 1 > VectorType;
	typedef Matrix < Scalar, Dynamic, Dynamic > FMatrixType;

	RealScalar tol = tol_error;
	const int maxIters = iters;
	iters = 0;

	const int m = mat.rows();

	VectorType p0 = rhs - mat*x;
	VectorType r0 = precond.solve(p0);
// 	RealScalar r0_sqnorm = r0.squaredNorm();

	VectorType w = VectorType::Zero(restart + 1);

	FMatrixType H = FMatrixType::Zero(m, restart + 1);
	VectorType tau = VectorType::Zero(restart + 1);
	std::vector < JacobiRotation < Scalar > > G(restart);

	// generate first Householder vector
	VectorType e;
	RealScalar beta;
	r0.makeHouseholder(e, tau.coeffRef(0), beta);
	w(0)=(Scalar) beta;
	H.bottomLeftCorner(m - 1, 1) = e;

	for (int k = 1; k <= restart; ++k) {

		++iters;

		VectorType v = VectorType::Unit(m, k - 1), workspace(m);

		// apply Householder reflections H_{1} ... H_{k-1} to v
		for (int i = k - 1; i >= 0; --i) {
			v.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
		}

		// apply matrix M to v:  v = mat * v;
		VectorType t=mat*v;
		v=precond.solve(t);

		// apply Householder reflections H_{k-1} ... H_{1} to v
		for (int i = 0; i < k; ++i) {
			v.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
		}

		if (v.tail(m - k).norm() != 0.0) {

			if (k <= restart) {

				// generate new Householder vector
                                  VectorType e(m - k - 1);
				RealScalar beta;
				v.tail(m - k).makeHouseholder(e, tau.coeffRef(k), beta);
				H.col(k).tail(m - k - 1) = e;

				// apply Householder reflection H_{k} to v
				v.tail(m - k).applyHouseholderOnTheLeft(H.col(k).tail(m - k - 1), tau.coeffRef(k), workspace.data());

			}
                }

                if (k > 1) {
                        for (int i = 0; i < k - 1; ++i) {
                                // apply old Givens rotations to v
                                v.applyOnTheLeft(i, i + 1, G[i].adjoint());
                        }
                }

                if (k<m && v(k) != (Scalar) 0) {
                        // determine next Givens rotation
                        G[k - 1].makeGivens(v(k - 1), v(k));

                        // apply Givens rotation to v and w
                        v.applyOnTheLeft(k - 1, k, G[k - 1].adjoint());
                        w.applyOnTheLeft(k - 1, k, G[k - 1].adjoint());

                }

                // insert coefficients into upper matrix triangle
                H.col(k - 1).head(k) = v.head(k);

                bool stop=(k==m || abs(w(k)) < tol || iters == maxIters);

                if (stop || k == restart) {

                        // solve upper triangular system
                        VectorType y = w.head(k);
                        H.topLeftCorner(k, k).template triangularView < Eigen::Upper > ().solveInPlace(y);

                        // use Horner-like scheme to calculate solution vector
                        VectorType x_new = y(k - 1) * VectorType::Unit(m, k - 1);

                        // apply Householder reflection H_{k} to x_new
                        x_new.tail(m - k + 1).applyHouseholderOnTheLeft(H.col(k - 1).tail(m - k), tau.coeffRef(k - 1), workspace.data());

                        for (int i = k - 2; i >= 0; --i) {
                                x_new += y(i) * VectorType::Unit(m, i);
                                // apply Householder reflection H_{i} to x_new
                                x_new.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
                        }

                        x += x_new;

                        if (stop) {
                                return true;
                        } else {
                                k=0;

                                // reset data for a restart  r0 = rhs - mat * x;
                                VectorType p0=mat*x;
                                VectorType p1=precond.solve(p0);
                                r0 = rhs - p1;
//                                 r0_sqnorm = r0.squaredNorm();
                                w = VectorType::Zero(restart + 1);
                                H = FMatrixType::Zero(m, restart + 1);
                                tau = VectorType::Zero(restart + 1);

                                // generate first Householder vector
                                RealScalar beta;
                                r0.makeHouseholder(e, tau.coeffRef(0), beta);
                                w(0)=(Scalar) beta;
                                H.bottomLeftCorner(m - 1, 1) = e;

                        }

                }



	}
	
	return false;

}

}

template< typename _MatrixType,
          typename _Preconditioner = DiagonalPreconditioner<typename _MatrixType::Scalar> >
class GMRES;

namespace internal {

template< typename _MatrixType, typename _Preconditioner>
struct traits<GMRES<_MatrixType,_Preconditioner> >
{
  typedef _MatrixType MatrixType;
  typedef _Preconditioner Preconditioner;
};

}

/** \ingroup IterativeLinearSolvers_Module
  * \brief A GMRES solver for sparse square problems
  *
  * This class allows to solve for A.x = b sparse linear problems using a generalized minimal
  * residual method. The vectors x and b can be either dense or sparse.
  *
  * \tparam _MatrixType the type of the sparse matrix A, can be a dense or a sparse matrix.
  * \tparam _Preconditioner the type of the preconditioner. Default is DiagonalPreconditioner
  *
  * The maximal number of iterations and tolerance value can be controlled via the setMaxIterations()
  * and setTolerance() methods. The defaults are the size of the problem for the maximal number of iterations
  * and NumTraits<Scalar>::epsilon() for the tolerance.
  * 
  * This class can be used as the direct solver classes. Here is a typical usage example:
  * \code
  * int n = 10000;
  * VectorXd x(n), b(n);
  * SparseMatrix<double> A(n,n);
  * // fill A and b
  * GMRES<SparseMatrix<double> > solver(A);
  * x = solver.solve(b);
  * std::cout << "#iterations:     " << solver.iterations() << std::endl;
  * std::cout << "estimated error: " << solver.error()      << std::endl;
  * // update b, and solve again
  * x = solver.solve(b);
  * \endcode
  * 
  * By default the iterations start with x=0 as an initial guess of the solution.
  * One can control the start using the solveWithGuess() method. Here is a step by
  * step execution example starting with a random guess and printing the evolution
  * of the estimated error:
  * * \code
  * x = VectorXd::Random(n);
  * solver.setMaxIterations(1);
  * int i = 0;
  * do {
  *   x = solver.solveWithGuess(b,x);
  *   std::cout << i << " : " << solver.error() << std::endl;
  *   ++i;
  * } while (solver.info()!=Success && i<100);
  * \endcode
  * Note that such a step by step excution is slightly slower.
  * 
  * \sa class SimplicialCholesky, DiagonalPreconditioner, IdentityPreconditioner
  */
template< typename _MatrixType, typename _Preconditioner>
class GMRES : public IterativeSolverBase<GMRES<_MatrixType,_Preconditioner> >
{
  typedef IterativeSolverBase<GMRES> Base;
  using Base::mp_matrix;
  using Base::m_error;
  using Base::m_iterations;
  using Base::m_info;
  using Base::m_isInitialized;
 
private:
  int m_restart;
  
public:
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef _Preconditioner Preconditioner;

public:

  /** Default constructor. */
  GMRES() : Base(), m_restart(30) {}

  /** Initialize the solver with matrix \a A for further \c Ax=b solving.
    * 
    * This constructor is a shortcut for the default constructor followed
    * by a call to compute().
    * 
    * \warning this class stores a reference to the matrix A as well as some
    * precomputed values that depend on it. Therefore, if \a A is changed
    * this class becomes invalid. Call compute() to update it with the new
    * matrix A, or modify a copy of A.
    */
  GMRES(const MatrixType& A) : Base(A), m_restart(30) {}

  ~GMRES() {}
  
  /** Get the number of iterations after that a restart is performed.
    */
  int get_restart() { return m_restart; }
  
  /** Set the number of iterations after that a restart is performed.
    *  \param restart   number of iterations for a restarti, default is 30.
    */
  void set_restart(const int restart) { m_restart=restart; }
  
  /** \returns the solution x of \f$ A x = b \f$ using the current decomposition of A
    * \a x0 as an initial solution.
    *
    * \sa compute()
    */
  template<typename Rhs,typename Guess>
  inline const internal::solve_retval_with_guess<GMRES, Rhs, Guess>
  solveWithGuess(const MatrixBase<Rhs>& b, const Guess& x0) const
  {
    eigen_assert(m_isInitialized && "GMRES is not initialized.");
    eigen_assert(Base::rows()==b.rows()
              && "GMRES::solve(): invalid number of rows of the right hand side matrix b");
    return internal::solve_retval_with_guess
            <GMRES, Rhs, Guess>(*this, b.derived(), x0);
  }
  
  /** \internal */
  template<typename Rhs,typename Dest>
  void _solveWithGuess(const Rhs& b, Dest& x) const
  {    
    bool failed = false;
    for(int j=0; j<b.cols(); ++j)
    {
      m_iterations = Base::maxIterations();
      m_error = Base::m_tolerance;
      
      typename Dest::ColXpr xj(x,j);
      if(!internal::gmres(*mp_matrix, b.col(j), xj, Base::m_preconditioner, m_iterations, m_restart, m_error))
        failed = true;
    }
    m_info = failed ? NumericalIssue
           : m_error <= Base::m_tolerance ? Success
           : NoConvergence;
    m_isInitialized = true;
  }

  /** \internal */
  template<typename Rhs,typename Dest>
  void _solve(const Rhs& b, Dest& x) const
  {
    x = b;
    if(x.squaredNorm() == 0) return; // Check Zero right hand side
    _solveWithGuess(b,x);
  }

protected:

};


namespace internal {

  template<typename _MatrixType, typename _Preconditioner, typename Rhs>
struct solve_retval<GMRES<_MatrixType, _Preconditioner>, Rhs>
  : solve_retval_base<GMRES<_MatrixType, _Preconditioner>, Rhs>
{
  typedef GMRES<_MatrixType, _Preconditioner> Dec;
  EIGEN_MAKE_SOLVE_HELPERS(Dec,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dec()._solve(rhs(),dst);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_GMRES_H
