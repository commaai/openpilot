// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpNLP.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPNLP_HPP__
#define __IPNLP_HPP__

#include "IpUtils.hpp"
#include "IpVector.hpp"
#include "IpSmartPtr.hpp"
#include "IpMatrix.hpp"
#include "IpSymMatrix.hpp"
#include "IpOptionsList.hpp"
#include "IpAlgTypes.hpp"
#include "IpReturnCodes.hpp"

namespace Ipopt
{
  // forward declarations
  class IpoptData;
  class IpoptCalculatedQuantities;
  class IteratesVector;

  /** Brief Class Description.
   *  Detailed Class Description.
   */
  class NLP : public ReferencedObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Default constructor */
    NLP()
    {}

    /** Default destructor */
    virtual ~NLP()
    {}
    //@}

    /** Exceptions */
    //@{
    DECLARE_STD_EXCEPTION(USER_SCALING_NOT_IMPLEMENTED);
    DECLARE_STD_EXCEPTION(INVALID_NLP);
    //@}

    /** @name NLP Initialization (overload in
     *  derived classes).*/
    //@{
    /** Overload if you want the chance to process options or parameters that
     *  may be specific to the NLP */
    virtual bool ProcessOptions(const OptionsList& options,
                                const std::string& prefix)
    {
      return true;
    }

    /** Method for creating the derived vector / matrix types.  The
     *  Hess_lagrangian_space pointer can be NULL if a quasi-Newton
     *  options is chosen. */
    virtual bool GetSpaces(SmartPtr<const VectorSpace>& x_space,
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
                           SmartPtr<const SymMatrixSpace>& Hess_lagrangian_space)=0;

    /** Method for obtaining the bounds information */
    virtual bool GetBoundsInformation(const Matrix& Px_L,
                                      Vector& x_L,
                                      const Matrix& Px_U,
                                      Vector& x_U,
                                      const Matrix& Pd_L,
                                      Vector& d_L,
                                      const Matrix& Pd_U,
                                      Vector& d_U)=0;

    /** Method for obtaining the starting point for all the
     *  iterates. ToDo it might not make sense to ask for initial
     *  values for v_L and v_U? */
    virtual bool GetStartingPoint(
      SmartPtr<Vector> x,
      bool need_x,
      SmartPtr<Vector> y_c,
      bool need_y_c,
      SmartPtr<Vector> y_d,
      bool need_y_d,
      SmartPtr<Vector> z_L,
      bool need_z_L,
      SmartPtr<Vector> z_U,
      bool need_z_U
    )=0;

    /** Method for obtaining an entire iterate as a warmstart point.
     *  The incoming IteratesVector has to be filled.  The default
     *  dummy implementation returns false. */
    virtual bool GetWarmStartIterate(IteratesVector& warm_start_iterate)
    {
      return false;
    }
    //@}

    /** @name NLP evaluation routines (overload
     *  in derived classes. */
    //@{
    virtual bool Eval_f(const Vector& x, Number& f) = 0;

    virtual bool Eval_grad_f(const Vector& x, Vector& g_f) = 0;

    virtual bool Eval_c(const Vector& x, Vector& c) = 0;

    virtual bool Eval_jac_c(const Vector& x, Matrix& jac_c) = 0;

    virtual bool Eval_d(const Vector& x, Vector& d) = 0;

    virtual bool Eval_jac_d(const Vector& x, Matrix& jac_d) = 0;

    virtual bool Eval_h(const Vector& x,
                        Number obj_factor,
                        const Vector& yc,
                        const Vector& yd,
                        SymMatrix& h) = 0;
    //@}

    /** @name NLP solution routines. Have default dummy
     *  implementations that can be overloaded. */
    //@{
    /** This method is called at the very end of the optimization.  It
     *  provides the final iterate to the user, so that it can be
     *  stored as the solution.  The status flag indicates the outcome
     *  of the optimization, where SolverReturn is defined in
     *  IpAlgTypes.hpp.  */
    virtual void FinalizeSolution(SolverReturn status,
                                  const Vector& x, const Vector& z_L,
                                  const Vector& z_U,
                                  const Vector& c, const Vector& d,
                                  const Vector& y_c, const Vector& y_d,
                                  Number obj_value,
                                  const IpoptData* ip_data,
                                  IpoptCalculatedQuantities* ip_cq)
    {}

    /** This method is called once per iteration, after the iteration
     *  summary output has been printed.  It provides the current
     *  information to the user to do with it anything she wants.  It
     *  also allows the user to ask for a premature termination of the
     *  optimization by returning false, in which case Ipopt will
     *  terminate with a corresponding return status.  The basic
     *  information provided in the argument list has the quantities
     *  values printed in the iteration summary line.  If more
     *  information is required, a user can obtain it from the IpData
     *  and IpCalculatedQuantities objects.  However, note that the
     *  provided quantities are all for the problem that Ipopt sees,
     *  i.e., the quantities might be scaled, fixed variables might be
     *  sorted out, etc.  The status indicates things like whether the
     *  algorithm is in the restoration phase...  In the restoration
     *  phase, the dual variables are probably not not changing. */
    virtual bool IntermediateCallBack(AlgorithmMode mode,
                                      Index iter, Number obj_value,
                                      Number inf_pr, Number inf_du,
                                      Number mu, Number d_norm,
                                      Number regularization_size,
                                      Number alpha_du, Number alpha_pr,
                                      Index ls_trials,
                                      const IpoptData* ip_data,
                                      IpoptCalculatedQuantities* ip_cq)
    {
      return true;
    }
    //@}

    /** Routines to get the scaling parameters. These do not need to
     *  be overloaded unless the options are set for User scaling
     */
    //@{
    virtual void GetScalingParameters(
      const SmartPtr<const VectorSpace> x_space,
      const SmartPtr<const VectorSpace> c_space,
      const SmartPtr<const VectorSpace> d_space,
      Number& obj_scaling,
      SmartPtr<Vector>& x_scaling,
      SmartPtr<Vector>& c_scaling,
      SmartPtr<Vector>& d_scaling) const
    {
      THROW_EXCEPTION(USER_SCALING_NOT_IMPLEMENTED,
                      "You have set options for user provided scaling, but have"
                      " not implemented GetScalingParameters in the NLP interface");
    }
    //@}

    /** Method for obtaining the subspace in which the limited-memory
     *  Hessian approximation should be done.  This is only called if
     *  the limited-memory Hessian approximation is chosen.  Since the
     *  Hessian is zero in the space of all variables that appear in
     *  the problem functions only linearly, this allows the user to
     *  provide a VectorSpace for all nonlinear variables, and an
     *  ExpansionMatrix to lift from this VectorSpace to the
     *  VectorSpace of the primal variables x.  If the returned values
     *  are NULL, it is assumed that the Hessian is to be approximated
     *  in the space of all x variables.  The default instantiation of
     *  this method returns NULL, and a user only has to overwrite
     *  this method if the approximation is to be done only in a
     *  subspace. */
    virtual void
    GetQuasiNewtonApproximationSpaces(SmartPtr<VectorSpace>& approx_space,
                                      SmartPtr<Matrix>& P_approx)
    {
      approx_space = NULL;
      P_approx = NULL;
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
    NLP(const NLP&);

    /** Overloaded Equals Operator */
    void operator=(const NLP&);
    //@}
  };

} // namespace Ipopt

#endif
