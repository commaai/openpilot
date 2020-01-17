// Copyright (C) 2005, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpSolveStatistics.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Carl Laird, Andreas Waechter          IBM    2005-08-15

#ifndef __IPSOLVESTATISTICS_HPP__
#define __IPSOLVESTATISTICS_HPP__

#include "IpReferenced.hpp"
#include "IpSmartPtr.hpp"

namespace Ipopt
{
  // forward declaration (to avoid inclusion of too many header files)
  class IpoptNLP;
  class IpoptData;
  class IpoptCalculatedQuantities;

  /** This class collects statistics about an optimziation run, such
   *  as iteration count, final infeasibilities etc.  It is meant to
   *  provide such information to a user of Ipopt during the
   *  finalize_solution call.
   */
  class SolveStatistics : public ReferencedObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default constructor.  It takes in those collecting Ipopt
     *  objects that can provide the statistics information.  Those
     *  statistics are retrieved at the time of the constructor
     *  call. */
    SolveStatistics(const SmartPtr<IpoptNLP>& ip_nlp,
                    const SmartPtr<IpoptData>& ip_data,
                    const SmartPtr<IpoptCalculatedQuantities>& ip_cq);

    /** Default destructor */
    virtual ~SolveStatistics()
    {}
    //@}

    /** @name Accessor methods for retrieving different kind of solver
     *  statistics information */
    //@{
    /** Iteration counts. */
    virtual Index IterationCount() const;
    /** Total CPU time, including function evaluations. */
    virtual Number TotalCpuTime() const;
    /** Total CPU time, including function evaluations. Included for
     *  backward compatibility. */
    Number TotalCPUTime() const
    {
      return TotalCpuTime();
    }
    /** Total System time, including function evaluations. */
    virtual Number TotalSysTime() const;
    /** Total wall clock time, including function evaluations. */
    virtual Number TotalWallclockTime() const;
    /** Number of NLP function evaluations. */
    virtual void NumberOfEvaluations(Index& num_obj_evals,
                                     Index& num_constr_evals,
                                     Index& num_obj_grad_evals,
                                     Index& num_constr_jac_evals,
                                     Index& num_hess_evals) const;
    /** Unscaled solution infeasibilities */
    virtual void Infeasibilities(Number& dual_inf,
                                 Number& constr_viol,
                                 Number& complementarity,
                                 Number& kkt_error) const;
    /** Scaled solution infeasibilities */
    virtual void ScaledInfeasibilities(Number& scaled_dual_inf,
                                       Number& scaled_constr_viol,
                                       Number& scaled_complementarity,
                                       Number& scaled_kkt_error) const;
    /** Final value of objective function */
    virtual Number FinalObjective() const;
    /** Final scaled value of objective function */
    virtual Number FinalScaledObjective() const;
    //@}

  private:
    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Default Constructor */
    SolveStatistics();

    /** Copy Constructor */
    SolveStatistics(const SolveStatistics&);

    /** Overloaded Equals Operator */
    void operator=(const SolveStatistics&);
    //@}

    /** @name Fields for storing the statistics data */
    //@{
    /** Number of iterations. */
    Index num_iters_;
    /* Total CPU time */
    Number total_cpu_time_;
    /* Total system time */
    Number total_sys_time_;
    /* Total wall clock time */
    Number total_wallclock_time_;
    /** Number of objective function evaluations. */
    Index num_obj_evals_;
    /** Number of constraints evaluations (max of equality and
     *  inequality) */
    Index num_constr_evals_;
    /** Number of objective gradient evaluations. */
    Index num_obj_grad_evals_;
    /** Number of constraint Jacobian evaluations. */
    Index num_constr_jac_evals_;
    /** Number of Lagrangian Hessian evaluations. */
    Index num_hess_evals_;

    /** Final scaled value of objective function */
    Number scaled_obj_val_;
    /** Final unscaled value of objective function */
    Number obj_val_;
    /** Final scaled dual infeasibility (max-norm) */
    Number scaled_dual_inf_;
    /** Final unscaled dual infeasibility (max-norm) */
    Number dual_inf_;
    /** Final scaled constraint violation (max-norm) */
    Number scaled_constr_viol_;
    /** Final unscaled constraint violation (max-norm) */
    Number constr_viol_;
    /** Final scaled complementarity error (max-norm) */
    Number scaled_compl_;
    /** Final unscaled complementarity error (max-norm) */
    Number compl_;
    /** Final overall scaled KKT error (max-norm) */
    Number scaled_kkt_error_;
    /** Final overall unscaled KKT error (max-norm) */
    Number kkt_error_;
    //@}
  };

} // namespace Ipopt

#endif
