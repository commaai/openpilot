// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpIpoptNLP.hpp 2594 2015-08-09 14:31:05Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPIPOPTNLP_HPP__
#define __IPIPOPTNLP_HPP__

#include "IpNLP.hpp"
#include "IpJournalist.hpp"
#include "IpNLPScaling.hpp"

namespace Ipopt
{
  // forward declarations
  class IteratesVector;

  /** This is the abstract base class for classes that map
   *  the traditional NLP into
   *  something that is more useful by Ipopt.
   *  This class takes care of storing the
   *  calculated model results, handles cacheing,
   *  and (some day) takes care of addition of slacks.
   */
  class IpoptNLP : public ReferencedObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    IpoptNLP(const SmartPtr<NLPScalingObject> nlp_scaling)
        :
        nlp_scaling_(nlp_scaling)
    {}

    /** Default destructor */
    virtual ~IpoptNLP()
    {}
    //@}

    /** Initialization method.  Set the internal options and
     *  initialize internal data structures. */
    virtual bool Initialize(const Journalist& jnlst,
                            const OptionsList& options,
                            const std::string& prefix)
    {
      bool ret = true;
      if (IsValid(nlp_scaling_)) {
        ret = nlp_scaling_->Initialize(jnlst, options, prefix);
      }
      return ret;
    }

    /**@name Possible Exceptions */
    //@{
    /** thrown if there is any error evaluating values from the nlp */
    DECLARE_STD_EXCEPTION(Eval_Error);
    //@}
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
                                     ) = 0;

    /** Method accessing the GetWarmStartIterate of the NLP */
    virtual bool GetWarmStartIterate(IteratesVector& warm_start_iterate)=0;

    /** Accessor methods for model data */
    //@{
    /** Objective value */
    virtual Number f(const Vector& x) = 0;

    /** Gradient of the objective */
    virtual SmartPtr<const Vector> grad_f(const Vector& x) = 0;

    /** Equality constraint residual */
    virtual SmartPtr<const Vector> c(const Vector& x) = 0;

    /** Jacobian Matrix for equality constraints */
    virtual SmartPtr<const Matrix> jac_c(const Vector& x) = 0;

    /** Inequality constraint residual (reformulated
     *  as equalities with slacks */
    virtual SmartPtr<const Vector> d(const Vector& x) = 0;

    /** Jacobian Matrix for inequality constraints */
    virtual SmartPtr<const Matrix> jac_d(const Vector& x) = 0;

    /** Hessian of the Lagrangian */
    virtual SmartPtr<const SymMatrix> h(const Vector& x,
                                        Number obj_factor,
                                        const Vector& yc,
                                        const Vector& yd
                                       ) = 0;

    /** Lower bounds on x */
    virtual SmartPtr<const Vector> x_L() const = 0;

    /** Permutation matrix (x_L_ -> x) */
    virtual SmartPtr<const Matrix> Px_L() const = 0;

    /** Upper bounds on x */
    virtual SmartPtr<const Vector> x_U() const = 0;

    /** Permutation matrix (x_U_ -> x */
    virtual SmartPtr<const Matrix> Px_U() const = 0;

    /** Lower bounds on d */
    virtual SmartPtr<const Vector> d_L() const = 0;

    /** Permutation matrix (d_L_ -> d) */
    virtual SmartPtr<const Matrix> Pd_L() const = 0;

    /** Upper bounds on d */
    virtual SmartPtr<const Vector> d_U() const = 0;

    /** Permutation matrix (d_U_ -> d */
    virtual SmartPtr<const Matrix> Pd_U() const = 0;

    /** x_space */
    virtual SmartPtr<const VectorSpace> x_space() const = 0;

    /** Accessor method to obtain the MatrixSpace for the Hessian
     *  matrix (or it's approximation) */
    virtual SmartPtr<const SymMatrixSpace> HessianMatrixSpace() const = 0;
    //@}

    /** Accessor method for vector/matrix spaces pointers. */
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
                           SmartPtr<const SymMatrixSpace>& Hess_lagrangian_space) = 0;

    /** Method for adapting the variable bounds.  This is called if
     *  slacks are becoming too small */
    virtual void AdjustVariableBounds(const Vector& new_x_L,
                                      const Vector& new_x_U,
                                      const Vector& new_d_L,
                                      const Vector& new_d_U)=0;

    /** @name Counters for the number of function evaluations. */
    //@{
    virtual Index f_evals() const = 0;
    virtual Index grad_f_evals() const = 0;
    virtual Index c_evals() const = 0;
    virtual Index jac_c_evals() const = 0;
    virtual Index d_evals() const = 0;
    virtual Index jac_d_evals() const = 0;
    virtual Index h_evals() const = 0;
    //@}

    /** @name Special method for dealing with the fact that the
     *  restoration phase objective function depends on the barrier
     *  parameter */
    //@{
    /** Method for telling the IpoptCalculatedQuantities class whether
     *  the objective function depends on the barrier function.  This
     *  is only used for the restoration phase NLP
     *  formulation. Probably only RestoIpoptNLP should overwrite
     *  this. */
    virtual bool objective_depends_on_mu() const
    {
      return false;
    }

    /** Replacement for the default objective function method which
     *  knows about the barrier parameter */
    virtual Number f(const Vector& x, Number mu) = 0;

    /** Replacement for the default objective gradient method which
     *  knows about the barrier parameter  */
    virtual SmartPtr<const Vector> grad_f(const Vector& x, Number mu) = 0;

    /** Replacement for the default Lagrangian Hessian method which
     *  knows about the barrier parameter */
    virtual SmartPtr<const SymMatrix> h(const Vector& x,
                                        Number obj_factor,
                                        const Vector& yc,
                                        const Vector& yd,
                                        Number mu) = 0;

    /** Provides a Hessian matrix from the correct matrix space with
     *  uninitialized values.  This can be used in LeastSquareMults to
     *  obtain a "zero Hessian". */
    virtual SmartPtr<const SymMatrix> uninitialized_h() = 0;
    //@}

    /**@name solution routines */
    //@{
    virtual void FinalizeSolution(SolverReturn status,
                                  const Vector& x, const Vector& z_L, const Vector& z_U,
                                  const Vector& c, const Vector& d,
                                  const Vector& y_c, const Vector& y_d,
                                  Number obj_value,
                                  const IpoptData* ip_data,
                                  IpoptCalculatedQuantities* ip_cq)=0;

    virtual bool IntermediateCallBack(AlgorithmMode mode,
                                      Index iter, Number obj_value,
                                      Number inf_pr, Number inf_du,
                                      Number mu, Number d_norm,
                                      Number regularization_size,
                                      Number alpha_du, Number alpha_pr,
                                      Index ls_trials,
                                      SmartPtr<const IpoptData> ip_data,
                                      SmartPtr<IpoptCalculatedQuantities> ip_cq)=0;
    //@}

    /** Returns the scaling strategy object */
    SmartPtr<NLPScalingObject> NLP_scaling() const
    {
      DBG_ASSERT(IsValid(nlp_scaling_));
      return nlp_scaling_;
    }

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
    IpoptNLP(const IpoptNLP&);

    /** Overloaded Equals Operator */
    void operator=(const IpoptNLP&);
    //@}

    SmartPtr<NLPScalingObject> nlp_scaling_;
  };

} // namespace Ipopt

#endif
