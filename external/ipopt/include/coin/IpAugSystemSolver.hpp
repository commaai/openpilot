// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpAugSystemSolver.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IP_AUGSYSTEMSOLVER_HPP__
#define __IP_AUGSYSTEMSOLVER_HPP__

#include "IpSymMatrix.hpp"
#include "IpSymLinearSolver.hpp"
#include "IpAlgStrategy.hpp"

namespace Ipopt
{
  DECLARE_STD_EXCEPTION(FATAL_ERROR_IN_LINEAR_SOLVER);

  /** Base class for Solver for the augmented system.  This is the
   *  base class for linear solvers that solve the augmented system,
   *  which is defined as
   *
   *  \f$\left[\begin{array}{cccc}
   *  W + D_x + \delta_xI & 0 & J_c^T & J_d^T\\
   *  0 & D_s + \delta_sI & 0 & -I \\
   *  J_c & 0 & D_c - \delta_cI & 0\\
   *  J_d & -I & 0 & D_d - \delta_dI
   *  \end{array}\right]
   *  \left(\begin{array}{c}sol_x\\sol_s\\sol_c\\sol_d\end{array}\right)=
   *  \left(\begin{array}{c}rhs_x\\rhs_s\\rhs_c\\rhs_d\end{array}\right)\f$
   *
   *  Since this system might be solved repeatedly for different right
   *  hand sides, it is desirable to step the factorization of a
   *  direct linear solver if possible.
   */
  class AugSystemSolver: public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default constructor. */
    AugSystemSolver()
    {}
    /** Default destructor */
    virtual ~AugSystemSolver()
    {}
    //@}

    /** overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** Set up the augmented system and solve it for a given right hand
     *  side.  If desired (i.e. if check_NegEVals is true), then the
     *  solution is only computed if the number of negative eigenvalues
     *  matches numberOfNegEVals.
     *
     *  The return value is the return value of the linear solver object.
     */
    virtual ESymSolverStatus Solve(
      const SymMatrix* W,
      double W_factor,
      const Vector* D_x,
      double delta_x,
      const Vector* D_s,
      double delta_s,
      const Matrix* J_c,
      const Vector* D_c,
      double delta_c,
      const Matrix* J_d,
      const Vector* D_d,
      double delta_d,
      const Vector& rhs_x,
      const Vector& rhs_s,
      const Vector& rhs_c,
      const Vector& rhs_d,
      Vector& sol_x,
      Vector& sol_s,
      Vector& sol_c,
      Vector& sol_d,
      bool check_NegEVals,
      Index numberOfNegEVals)
    {
      std::vector<SmartPtr<const Vector> > rhs_xV(1);
      rhs_xV[0] = &rhs_x;
      std::vector<SmartPtr<const Vector> > rhs_sV(1);
      rhs_sV[0] = &rhs_s;
      std::vector<SmartPtr<const Vector> > rhs_cV(1);
      rhs_cV[0] = &rhs_c;
      std::vector<SmartPtr<const Vector> > rhs_dV(1);
      rhs_dV[0] = &rhs_d;
      std::vector<SmartPtr<Vector> > sol_xV(1);
      sol_xV[0] = &sol_x;
      std::vector<SmartPtr<Vector> > sol_sV(1);
      sol_sV[0] = &sol_s;
      std::vector<SmartPtr<Vector> > sol_cV(1);
      sol_cV[0] = &sol_c;
      std::vector<SmartPtr<Vector> > sol_dV(1);
      sol_dV[0] = &sol_d;
      return MultiSolve(W, W_factor, D_x, delta_x, D_s, delta_s, J_c, D_c, delta_c,
                        J_d, D_d, delta_d, rhs_xV, rhs_sV, rhs_cV, rhs_dV,
                        sol_xV, sol_sV, sol_cV, sol_dV, check_NegEVals,
                        numberOfNegEVals);
    }

    /** Like Solve, but for multiple right hand sides.  The inheriting
     *  class has to be overload at least one of Solve and
     *  MultiSolve. */
    virtual ESymSolverStatus MultiSolve(
      const SymMatrix* W,
      double W_factor,
      const Vector* D_x,
      double delta_x,
      const Vector* D_s,
      double delta_s,
      const Matrix* J_c,
      const Vector* D_c,
      double delta_c,
      const Matrix* J_d,
      const Vector* D_d,
      double delta_d,
      std::vector<SmartPtr<const Vector> >& rhs_xV,
      std::vector<SmartPtr<const Vector> >& rhs_sV,
      std::vector<SmartPtr<const Vector> >& rhs_cV,
      std::vector<SmartPtr<const Vector> >& rhs_dV,
      std::vector<SmartPtr<Vector> >& sol_xV,
      std::vector<SmartPtr<Vector> >& sol_sV,
      std::vector<SmartPtr<Vector> >& sol_cV,
      std::vector<SmartPtr<Vector> >& sol_dV,
      bool check_NegEVals,
      Index numberOfNegEVals)
    {
      // Solve for one right hand side after the other
      Index nrhs = (Index)rhs_xV.size();
      DBG_ASSERT(nrhs>0);
      DBG_ASSERT(nrhs==(Index)rhs_sV.size());
      DBG_ASSERT(nrhs==(Index)rhs_cV.size());
      DBG_ASSERT(nrhs==(Index)rhs_dV.size());
      DBG_ASSERT(nrhs==(Index)sol_xV.size());
      DBG_ASSERT(nrhs==(Index)sol_sV.size());
      DBG_ASSERT(nrhs==(Index)sol_cV.size());
      DBG_ASSERT(nrhs==(Index)sol_dV.size());

      ESymSolverStatus retval=SYMSOLVER_SUCCESS;
      for (Index i=0; i<nrhs; i++) {
        retval = Solve(W, W_factor, D_x, delta_x, D_s, delta_s, J_c, D_c, delta_c,
                       J_d, D_d, delta_d,
                       *rhs_xV[i], *rhs_sV[i], *rhs_cV[i], *rhs_dV[i],
                       *sol_xV[i], *sol_sV[i], *sol_cV[i], *sol_dV[i],
                       check_NegEVals, numberOfNegEVals);
        if (retval!=SYMSOLVER_SUCCESS) {
          break;
        }
      }
      return retval;
    }

    /** Number of negative eigenvalues detected during last
     * solve.  Returns the number of negative eigenvalues of
     * the most recent factorized matrix.  This must not be called if
     * the linear solver does not compute this quantities (see
     * ProvidesInertia).
     */
    virtual Index NumberOfNegEVals() const =0;

    /** Query whether inertia is computed by linear solver.
     *  Returns true, if linear solver provides inertia.
     */
    virtual bool ProvidesInertia() const =0;

    /** Request to increase quality of solution for next solve.  Ask
     *  underlying linear solver to increase quality of solution for
     *  the next solve (e.g. increase pivot tolerance).  Returns
     *  false, if this is not possible (e.g. maximal pivot tolerance
     *  already used.)
     */
    virtual bool IncreaseQuality() =0;

  private:
    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Copy Constructor */
    AugSystemSolver(const AugSystemSolver&);

    /** Overloaded Equals Operator */
    void operator=(const AugSystemSolver&);
    //@}

  };

} // namespace Ipopt

#endif
