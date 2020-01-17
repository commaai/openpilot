// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpSymLinearSolver.hpp 2322 2013-06-12 17:45:57Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPSYMLINEARSOLVER_HPP__
#define __IPSYMLINEARSOLVER_HPP__

#include "IpUtils.hpp"
#include "IpSymMatrix.hpp"
#include "IpAlgStrategy.hpp"
#include <vector>

namespace Ipopt
{

  /** Enum to report outcome of a linear solve */
  enum ESymSolverStatus {
    /** Successful solve */
    SYMSOLVER_SUCCESS,
    /** Matrix seems to be singular; solve was aborted */
    SYMSOLVER_SINGULAR,
    /** The number of negative eigenvalues is not correct */
    SYMSOLVER_WRONG_INERTIA,
    /** Call the solver interface again after the matrix values have
     *  been restored */
    SYMSOLVER_CALL_AGAIN,
    /** Unrecoverable error in linear solver occurred.  The
     *  optimization will be aborted. */
    SYMSOLVER_FATAL_ERROR
  };

  /** Base class for all derived symmetric linear
   *  solvers.  In the full space version of Ipopt a large linear
   *  system has to be solved for the augmented system.  This case is
   *  meant to be the base class for all derived linear solvers for
   *  symmetric matrices (of type SymMatrix).
   *
   *  A linear solver can be used repeatedly for matrices with
   *  identical structure of nonzero elements.  The nonzero structure
   *  of those matrices must not be changed between calls.
   *
   *  The called might ask the solver to only solve the linear system
   *  if the system is nonsingular, and if the number of negative
   *  eigenvalues matches a given number.
   */
  class SymLinearSolver: public AlgorithmStrategyObject
  {
  public:
    /** @name Constructor/Destructor */
    //@{
    SymLinearSolver()
    {}

    virtual ~SymLinearSolver()
    {}
    //@}

    /** overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** @name Methods for requesting solution of the linear system. */
    //@{
    /** Solve operation for multiple right hand sides.  Solves the
     *  linear system A * Sol = Rhs with multiple right hand sides.  If
     *  necessary, A is factorized.  Correct solutions are only
     *  guaranteed if the return values is SYMSOLVER_SUCCESS.  The
     *  solver will return SYMSOLVER_SINGULAR if the linear system is
     *  singular, and it will return SYMSOLVER_WRONG_INERTIA if
     *  check_NegEVals is true and the number of negative eigenvalues
     *  in the matrix does not match numberOfNegEVals.
     *
     *  check_NegEVals cannot be chosen true, if ProvidesInertia()
     *  returns false.
     */
    virtual ESymSolverStatus MultiSolve(const SymMatrix &A,
                                        std::vector<SmartPtr<const Vector> >& rhsV,
                                        std::vector<SmartPtr<Vector> >& solV,
                                        bool check_NegEVals,
                                        Index numberOfNegEVals)=0;

    /** Solve operation for a single right hand side. Solves the
     *  linear system A * Sol = Rhs.  See MultiSolve for more
     *  details. */
    ESymSolverStatus Solve(const SymMatrix &A,
                           const Vector& rhs, Vector& sol,
                           bool check_NegEVals,
                           Index numberOfNegEVals)
    {
      std::vector<SmartPtr<const Vector> > rhsV(1);
      rhsV[0] = &rhs;
      std::vector<SmartPtr<Vector> > solV(1);
      solV[0] = &sol;
      return MultiSolve(A, rhsV, solV, check_NegEVals,
                        numberOfNegEVals);
    }

    /** Number of negative eigenvalues detected during last
     *  factorization.  Returns the number of negative eigenvalues of
     *  the most recent factorized matrix.  This must not be called if
     *  the linear solver does not compute this quantities (see
     *  ProvidesInertia).
     */
    virtual Index NumberOfNegEVals() const =0;
    //@}

    //* @name Options of Linear solver */
    //@{
    /** Request to increase quality of solution for next solve.
     * Ask linear solver to increase quality of solution for the next
     * solve (e.g. increase pivot tolerance).  Returns false, if this
     * is not possible (e.g. maximal pivot tolerance already used.)
     */
    virtual bool IncreaseQuality() =0;

    /** Query whether inertia is computed by linear solver.
     * Returns true, if linear solver provides inertia.
     */
    virtual bool ProvidesInertia() const =0;
    //@}
  };


} // namespace Ipopt

#endif
