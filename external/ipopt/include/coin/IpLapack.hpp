// Copyright (C) 2005, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpLapack.hpp 2449 2013-12-16 00:25:42Z ghackebeil $
//
// Authors:  Andreas Waechter              IBM    2005-12-25

#ifndef __IPLAPACK_HPP__
#define __IPLAPACK_HPP__

#include "IpUtils.hpp"
#include "IpException.hpp"

namespace Ipopt
{
  DECLARE_STD_EXCEPTION(LAPACK_NOT_INCLUDED);

  /** Wrapper for LAPACK subroutine DPOTRS.  Solving a linear system
   *  given a Cholesky factorization.  We assume that the Cholesky
   *  factor is lower traiangular. */
  void IpLapackDpotrs(Index ndim, Index nrhs, const Number *a, Index lda,
                      Number *b, Index ldb);

  /** Wrapper for LAPACK subroutine DPOTRF.  Compute Cholesky
   *  factorization (lower triangular factor).  info is the return
   *  value from the LAPACK routine. */
  void IpLapackDpotrf(Index ndim, Number *a, Index lda, Index& info);

  /** Wrapper for LAPACK subroutine DSYEV.  Compute the Eigenvalue
   *  decomposition for a given matrix.  If compute_eigenvectors is
   *  true, a will contain the eigenvectors in its columns on
   *  return.  */
  void IpLapackDsyev(bool compute_eigenvectors, Index ndim, Number *a,
                     Index lda, Number *w, Index& info);

  /** Wrapper for LAPACK subroutine DGETRF.  Compute LU factorization.
   *  info is the return value from the LAPACK routine. */
  void IpLapackDgetrf(Index ndim, Number *a, Index* pivot, Index lda,
                      Index& info);

  /** Wrapper for LAPACK subroutine DGETRS.  Solving a linear system
   *  given a LU factorization. */
  void IpLapackDgetrs(Index ndim, Index nrhs, const Number *a, Index lda,
                      Index* ipiv, Number *b, Index ldb);

  /** Wrapper for LAPACK subroutine DPPSV.  Solves a symmetric positive
   *  definite linear system in packed storage format (upper triangular).
   *  info is the return value from the LAPACK routine. */
  void IpLapackDppsv(Index ndim, Index nrhs, const Number *a,
                     Number *b, Index ldb, Index& info);

} // namespace Ipopt

#endif
