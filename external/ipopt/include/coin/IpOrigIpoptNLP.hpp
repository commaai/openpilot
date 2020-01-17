// Copyright (C) 2004, 2010 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpOrigIpoptNLP.hpp 2594 2015-08-09 14:31:05Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPORIGIPOPTNLP_HPP__
#define __IPORIGIPOPTNLP_HPP__

#include "IpIpoptNLP.hpp"
#include "IpException.hpp"
#include "IpTimingStatistics.hpp"

namespace Ipopt
{

  /** enumeration for the Hessian information type. */
  enum HessianApproximationType {
    EXACT=0,
    LIMITED_MEMORY
  };

  /** enumeration for the Hessian approximation space. */
  enum HessianApproximationSpace {
    NONLINEAR_VARS=0,
    ALL_VARS
  };

  /** This class maps the traditional NLP into
   *  something that is more useful by Ipopt.
   *  This class takes care of storing the
   *  calculated model results, handles caching,
   *  and (some day) takes care of addition of slacks.
   */
  class OrigIpoptNLP : public IpoptNLP
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    OrigIpoptNLP(const SmartPtr<const Journalist>& jnlst,
                 const SmartPtr<NLP>& nlp,
                 const SmartPtr<NLPScalingObject>& nlp_scaling);

    /** Default destructor */
    virtual ~OrigIpoptNLP();
    //@}

    /** Initialize - overloaded from IpoptNLP */
    virtual bool Initialize(const Journalist& jnlst,
                            const OptionsList& options,
                            const std::string& prefix);

    /** Initialize (create) structures for
     *  the iteration data */
    virtual bool InitializeStructures(SmartPtr<Vector>& x,
                                      bool init_x,
                                      SmartPtr<Vector>& y_c,
                                      bool init_y_c,
                                      SmartPtr<Vector>& y_d,
                                      bool init_y_d,
                                      SmartPtr<Vector>& z_L,
                                      bool init_z_L,
                                      SmartPtr<Vector>& z_U,
                                      bool init_z_U,
                                      SmartPtr<Vector>& v_L,
                                      SmartPtr<Vector>& v_U
                                     );

    /** Method accessing the GetWarmStartIterate of the NLP */
    virtual bool GetWarmStartIterate(IteratesVector& warm_start_iterate)
    {
      return nlp_->GetWarmStartIterate(warm_start_iterate);
    }
    /** Accessor methods for model data */
    //@{
    /** Objective value */
    virtual Number f(const Vector& x);

    /** Objective value (depending in mu) - incorrect version for
     *  OrigIpoptNLP */
    virtual Number f(const Vector& x, Number mu);

    /** Gradient of the objective */
    virtual SmartPtr<const Vector> grad_f(const Vector& x);

    /** Gradient of the objective (depending in mu) - incorrect
     *  version for OrigIpoptNLP */
    virtual SmartPtr<const Vector> grad_f(const Vector& x, Number mu);

    /** Equality constraint residual */
    virtual SmartPtr<const Vector> c(const Vector& x);

    /** Jacobian Matrix for equality constraints */
    virtual SmartPtr<const Matrix> jac_c(const Vector& x);

    /** Inequality constraint residual (reformulated
     *  as equalities with slacks */
    virtual SmartPtr<const Vector> d(const Vector& x);

    /** Jacobian Matrix for inequality constraints*/
    virtual SmartPtr<const Matrix> jac_d(const Vector& x);

    /** Hessian of the Lagrangian */
    virtual SmartPtr<const SymMatrix> h(const Vector& x,
                                        Number obj_factor,
                                        const Vector& yc,
                                        const Vector& yd
                                       );

    /** Hessian of the Lagrangian (depending in mu) - incorrect
     *  version for OrigIpoptNLP */
    virtual SmartPtr<const SymMatrix> h(const Vector& x,
                                        Number obj_factor,
                                        const Vector& yc,
                                        const Vector& yd,
                                        Number mu);

    /** Provides a Hessian matrix from the correct matrix space with
     *  uninitialized values.  This can be used in LeastSquareMults to
     *  obtain a "zero Hessian". */
    virtual SmartPtr<const SymMatrix> uninitialized_h();

    /** Lower bounds on x */
    virtual SmartPtr<const Vector> x_L() const
    {
      return x_L_;
    }

    /** Permutation matrix (x_L_ -> x) */
    virtual SmartPtr<const Matrix> Px_L() const
    {
      return Px_L_;
    }

    /** Upper bounds on x */
    virtual SmartPtr<const Vector> x_U() const
    {
      return x_U_;
    }

    /** Permutation matrix (x_U_ -> x */
    virtual SmartPtr<const Matrix> Px_U() const
    {
      return Px_U_;
    }

    /** Lower bounds on d */
    virtual SmartPtr<const Vector> d_L() const
    {
      return d_L_;
    }

    /** Permutation matrix (d_L_ -> d) */
    virtual SmartPtr<const Matrix> Pd_L() const
    {
      return Pd_L_;
    }

    /** Upper bounds on d */
    virtual SmartPtr<const Vector> d_U() const
    {
      return d_U_;
    }

    /** Permutation matrix (d_U_ -> d */
    virtual SmartPtr<const Matrix> Pd_U() const
    {
      return Pd_U_;
    }

    virtual SmartPtr<const SymMatrixSpace> HessianMatrixSpace() const
    {
      return h_space_;
    }

    virtual SmartPtr<const VectorSpace> x_space() const
    {
      return x_space_;
    }
    //@}

    /** Accessor method for vector/matrix spaces pointers */
    virtual void GetSpaces(SmartPtr<const VectorSpace>& x_space,
                           SmartPtr<const VectorSpace>& c_space,
                           SmartPtr<const VectorSpace>& d_space,
                           SmartPtr<const VectorSpace>& x_l_space,
                           SmartPtr<const MatrixSpace>& px_l_space,
                           SmartPtr<const VectorSpace>& x_u_space,
                           SmartPtr<const MatrixSpace>& px_u_space,
                           SmartPtr<const VectorSpace>& d_l_space,
                           SmartPtr<const MatrixSpace>& pd_l_space,
                           SmartPtr<const VectorSpace>& d_u_space,
                           SmartPtr<const MatrixSpace>& pd_u_space,
                           SmartPtr<const MatrixSpace>& Jac_c_space,
                           SmartPtr<const MatrixSpace>& Jac_d_space,
                           SmartPtr<const SymMatrixSpace>& Hess_lagrangian_space);

    /** Method for adapting the variable bounds.  This is called if
     *  slacks are becoming too small */
    virtual void AdjustVariableBounds(const Vector& new_x_L,
                                      const Vector& new_x_U,
                                      const Vector& new_d_L,
                                      const Vector& new_d_U);

    /** @name Counters for the number of function evaluations. */
    //@{
    virtual Index f_evals() const
    {
      return f_evals_;
    }
    virtual Index grad_f_evals() const
    {
      return grad_f_evals_;
    }
    virtual Index c_evals() const
    {
      return c_evals_;
    }
    virtual Index jac_c_evals() const
    {
      return jac_c_evals_;
    }
    virtual Index d_evals() const
    {
      return d_evals_;
    }
    virtual Index jac_d_evals() const
    {
      return jac_d_evals_;
    }
    virtual Index h_evals() const
    {
      return h_evals_;
    }
    //@}

    /** Solution Routines - overloaded from IpoptNLP*/
    //@{
    void FinalizeSolution(SolverReturn status,
                          const Vector& x, const Vector& z_L, const Vector& z_U,
                          const Vector& c, const Vector& d,
                          const Vector& y_c, const Vector& y_d,
                          Number obj_value,
                          const IpoptData* ip_data,
                          IpoptCalculatedQuantities* ip_cq);
    bool IntermediateCallBack(AlgorithmMode mode,
                              Index iter, Number obj_value,
                              Number inf_pr, Number inf_du,
                              Number mu, Number d_norm,
                              Number regularization_size,
                              Number alpha_du, Number alpha_pr,
                              Index ls_trials,
                              SmartPtr<const IpoptData> ip_data,
                              SmartPtr<IpoptCalculatedQuantities> ip_cq);
    //@}

    /** @name Methods for IpoptType */
    //@{
    /** Called by IpoptType to register the options */
    static void RegisterOptions(SmartPtr<RegisteredOptions> roptions);
    //@}

    /** Accessor method to the underlying NLP */
    SmartPtr<NLP> nlp()
    {
      return nlp_;
    }

    /**@name Methods related to function evaluation timing. */
    //@{

    /** Reset the timing statistics */
    void ResetTimes();

    void PrintTimingStatistics(Journalist& jnlst,
                               EJournalLevel level,
                               EJournalCategory category) const;

    const TimedTask& f_eval_time() const
    {
      return f_eval_time_;
    }
    const TimedTask& grad_f_eval_time() const
    {
      return grad_f_eval_time_;
    }
    const TimedTask& c_eval_time() const
    {
      return c_eval_time_;
    }
    const TimedTask& jac_c_eval_time() const
    {
      return jac_c_eval_time_;
    }
    const TimedTask& d_eval_time() const
    {
      return d_eval_time_;
    }
    const TimedTask& jac_d_eval_time() const
    {
      return jac_d_eval_time_;
    }
    const TimedTask& h_eval_time() const
    {
      return h_eval_time_;
    }

    Number TotalFunctionEvaluationCpuTime() const;
    Number TotalFunctionEvaluationSysTime() const;
    Number TotalFunctionEvaluationWallclockTime() const;
    //@}

  private:
    /** journalist */
    SmartPtr<const Journalist> jnlst_;

    /** Pointer to the NLP */
    SmartPtr<NLP> nlp_;

    /** Necessary Vector/Matrix spaces */
    //@{
    SmartPtr<const VectorSpace> x_space_;
    SmartPtr<const VectorSpace> c_space_;
    SmartPtr<const VectorSpace> d_space_;
    SmartPtr<const VectorSpace> x_l_space_;
    SmartPtr<const MatrixSpace> px_l_space_;
    SmartPtr<const VectorSpace> x_u_space_;
    SmartPtr<const MatrixSpace> px_u_space_;
    SmartPtr<const VectorSpace> d_l_space_;
    SmartPtr<const MatrixSpace> pd_l_space_;
    SmartPtr<const VectorSpace> d_u_space_;
    SmartPtr<const MatrixSpace> pd_u_space_;
    SmartPtr<const MatrixSpace> jac_c_space_;
    SmartPtr<const MatrixSpace> jac_d_space_;
    SmartPtr<const SymMatrixSpace> h_space_;

    SmartPtr<const MatrixSpace> scaled_jac_c_space_;
    SmartPtr<const MatrixSpace> scaled_jac_d_space_;
    SmartPtr<const SymMatrixSpace> scaled_h_space_;
    //@}
    /**@name Storage for Model Quantities */
    //@{
    /** Objective function */
    CachedResults<Number> f_cache_;

    /** Gradient of the objective function */
    CachedResults<SmartPtr<const Vector> > grad_f_cache_;

    /** Equality constraint residuals */
    CachedResults<SmartPtr<const Vector> > c_cache_;

    /** Jacobian Matrix for equality constraints
     *  (current iteration) */
    CachedResults<SmartPtr<const Matrix> > jac_c_cache_;

    /** Inequality constraint residual (reformulated
     *  as equalities with slacks */
    CachedResults<SmartPtr<const Vector> > d_cache_;

    /** Jacobian Matrix for inequality constraints
     *  (current iteration) */
    CachedResults<SmartPtr<const Matrix> > jac_d_cache_;

    /** Hessian of the lagrangian
     *  (current iteration) */
    CachedResults<SmartPtr<const SymMatrix> > h_cache_;

    /** Unscaled version of x vector */
    CachedResults<SmartPtr<const Vector> > unscaled_x_cache_;

    /** Lower bounds on x */
    SmartPtr<const Vector> x_L_;

    /** Permutation matrix (x_L_ -> x) */
    SmartPtr<const Matrix> Px_L_;

    /** Upper bounds on x */
    SmartPtr<const Vector> x_U_;

    /** Permutation matrix (x_U_ -> x */
    SmartPtr<const Matrix> Px_U_;

    /** Lower bounds on d */
    SmartPtr<const Vector> d_L_;

    /** Permutation matrix (d_L_ -> d) */
    SmartPtr<const Matrix> Pd_L_;

    /** Upper bounds on d */
    SmartPtr<const Vector> d_U_;

    /** Permutation matrix (d_U_ -> d */
    SmartPtr<const Matrix> Pd_U_;

    /** Original unmodified lower bounds on x */
    SmartPtr<const Vector> orig_x_L_;

    /** Original unmodified upper bounds on x */
    SmartPtr<const Vector> orig_x_U_;
    //@}

    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Default Constructor */
    OrigIpoptNLP();

    /** Copy Constructor */
    OrigIpoptNLP(const OrigIpoptNLP&);

    /** Overloaded Equals Operator */
    void operator=(const OrigIpoptNLP&);
    //@}

    /** @name auxilliary functions */
    //@{
    /** relax the bounds by a relative move of relax_bound_factor.
     *  Here, relax_bound_factor should be negative (or zero) for
     *  lower bounds, and positive (or zero) for upper bounds.
     */
    void relax_bounds(Number bound_relax_factor, Vector& bounds);
    /** Method for getting the unscaled version of the x vector */
    SmartPtr<const Vector> get_unscaled_x(const Vector& x);
    //@}

    /** @name Algorithmic parameters */
    //@{
    /** relaxation factor for the bounds */
    Number bound_relax_factor_;
    /** Flag indicating whether the primal variables should be
     *  projected back into original bounds are optimization. */
    bool honor_original_bounds_;
    /** Flag indicating whether the TNLP with identical structure has
     *  already been solved before. */
    bool warm_start_same_structure_;
    /** Flag indicating what Hessian information is to be used. */
    HessianApproximationType hessian_approximation_;
    /** Flag indicating in which space Hessian is to be approximated. */
    HessianApproximationSpace hessian_approximation_space_;
    /** Flag indicating whether it is desired to check if there are
     *  Nan or Inf entries in first and second derivative matrices. */
    bool check_derivatives_for_naninf_;
    /** Flag indicating if we need to ask for equality constraint
     *  Jacobians only once */
    bool jac_c_constant_;
    /** Flag indicating if we need to ask for inequality constraint
     *  Jacobians only once */
    bool jac_d_constant_;
    /** Flag indicating if we need to ask for Hessian only once */
    bool hessian_constant_;
    //@}

    /** @name Counters for the function evaluations */
    //@{
    Index f_evals_;
    Index grad_f_evals_;
    Index c_evals_;
    Index jac_c_evals_;
    Index d_evals_;
    Index jac_d_evals_;
    Index h_evals_;
    //@}

    /** Flag indicating if initialization method has been called */
    bool initialized_;

    /**@name Timing statistics for the function evaluations. */
    //@{
    TimedTask f_eval_time_;
    TimedTask grad_f_eval_time_;
    TimedTask c_eval_time_;
    TimedTask jac_c_eval_time_;
    TimedTask d_eval_time_;
    TimedTask jac_d_eval_time_;
    TimedTask h_eval_time_;
    //@}
  };

} // namespace Ipopt

#endif
