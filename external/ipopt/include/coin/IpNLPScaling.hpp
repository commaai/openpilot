// Copyright (C) 2004, 2007 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpNLPScaling.hpp 2269 2013-05-05 11:32:40Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPNLPSCALING_HPP__
#define __IPNLPSCALING_HPP__

#include "IpOptionsList.hpp"
#include "IpRegOptions.hpp"

namespace Ipopt
{
  // forward declarations
  class Vector;
  class VectorSpace;
  class Matrix;
  class MatrixSpace;
  class SymMatrix;
  class SymMatrixSpace;
  class ScaledMatrixSpace;
  class SymScaledMatrixSpace;
  
  /** This is the abstract base class for problem scaling.
   *  It is repsonsible for determining the scaling factors
   *  and mapping quantities in and out of scaled and unscaled
   *  versions 
   */
  class NLPScalingObject : public ReferencedObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    NLPScalingObject();

    /** Default destructor */
    virtual ~NLPScalingObject();
    //@}

    /** Method to initialize the options */
    bool Initialize(const Journalist& jnlst,
                    const OptionsList& options,
                    const std::string& prefix)
    {
      jnlst_ = &jnlst;
      return InitializeImpl(options, prefix);
    }

    /** Methods to map scaled and unscaled matrices */
    //@{
    /** Returns an obj-scaled version of the given scalar */
    virtual Number apply_obj_scaling(const Number& f)=0;
    /** Returns an obj-unscaled version of the given scalar */
    virtual Number unapply_obj_scaling(const Number& f)=0;
    /** Returns an x-scaled version of the given vector */
    virtual SmartPtr<Vector>
    apply_vector_scaling_x_NonConst(const SmartPtr<const Vector>& v)=0;
    /** Returns an x-scaled version of the given vector */
    virtual SmartPtr<const Vector>
    apply_vector_scaling_x(const SmartPtr<const Vector>& v)=0;
    /** Returns an x-unscaled version of the given vector */
    virtual SmartPtr<Vector>
    unapply_vector_scaling_x_NonConst(const SmartPtr<const Vector>& v)=0;
    /** Returns an x-unscaled version of the given vector */
    virtual SmartPtr<const Vector>
    unapply_vector_scaling_x(const SmartPtr<const Vector>& v)=0;
    /** Returns an c-scaled version of the given vector */
    virtual SmartPtr<const Vector>
    apply_vector_scaling_c(const SmartPtr<const Vector>& v)=0;
    /** Returns an c-unscaled version of the given vector */
    virtual SmartPtr<const Vector>
    unapply_vector_scaling_c(const SmartPtr<const Vector>& v)=0;
    /** Returns an c-scaled version of the given vector */
    virtual SmartPtr<Vector>
    apply_vector_scaling_c_NonConst(const SmartPtr<const Vector>& v)=0;
    /** Returns an c-unscaled version of the given vector */
    virtual SmartPtr<Vector>
    unapply_vector_scaling_c_NonConst(const SmartPtr<const Vector>& v)=0;
    /** Returns an d-scaled version of the given vector */
    virtual SmartPtr<const Vector>
    apply_vector_scaling_d(const SmartPtr<const Vector>& v)=0;
    /** Returns an d-unscaled version of the given vector */
    virtual SmartPtr<const Vector>
    unapply_vector_scaling_d(const SmartPtr<const Vector>& v)=0;
    /** Returns an d-scaled version of the given vector */
    virtual SmartPtr<Vector>
    apply_vector_scaling_d_NonConst(const SmartPtr<const Vector>& v)=0;
    /** Returns an d-unscaled version of the given vector */
    virtual SmartPtr<Vector>
    unapply_vector_scaling_d_NonConst(const SmartPtr<const Vector>& v)=0;
    /** Returns a scaled version of the jacobian for c.  If the
     *  overloaded method does not make a new matrix, make sure to set
     *  the matrix ptr passed in to NULL.
     */
    virtual SmartPtr<const Matrix>
    apply_jac_c_scaling(SmartPtr<const Matrix> matrix)=0;
    /** Returns a scaled version of the jacobian for d If the
     *  overloaded method does not create a new matrix, make sure to
     *  set the matrix ptr passed in to NULL.
     */
    virtual SmartPtr<const Matrix>
    apply_jac_d_scaling(SmartPtr<const Matrix> matrix)=0;
    /** Returns a scaled version of the hessian of the lagrangian If
     *  the overloaded method does not create a new matrix, make sure
     *  to set the matrix ptr passed in to NULL.
     */
    virtual SmartPtr<const SymMatrix>
    apply_hessian_scaling(SmartPtr<const SymMatrix> matrix)=0;
    //@}

    /** Methods for scaling bounds - these wrap those above */
    //@{
    /** Returns an x-scaled vector in the x_L or x_U space */
    SmartPtr<Vector> apply_vector_scaling_x_LU_NonConst(
      const Matrix& Px_LU,
      const SmartPtr<const Vector>& lu,
      const VectorSpace& x_space);
    /** Returns an x-scaled vector in the x_L or x_U space */
    SmartPtr<const Vector> apply_vector_scaling_x_LU(
      const Matrix& Px_LU,
      const SmartPtr<const Vector>& lu,
      const VectorSpace& x_space);
    /** Returns an d-scaled vector in the d_L or d_U space */
    SmartPtr<Vector> apply_vector_scaling_d_LU_NonConst(
      const Matrix& Pd_LU,
      const SmartPtr<const Vector>& lu,
      const VectorSpace& d_space);
    /** Returns an d-scaled vector in the d_L or d_U space */
    SmartPtr<const Vector> apply_vector_scaling_d_LU(
      const Matrix& Pd_LU,
      const SmartPtr<const Vector>& lu,
      const VectorSpace& d_space);
    /** Returns an d-unscaled vector in the d_L or d_U space */
    SmartPtr<Vector> unapply_vector_scaling_d_LU_NonConst(
      const Matrix& Pd_LU,
      const SmartPtr<const Vector>& lu,
      const VectorSpace& d_space);
    /** Returns an d-unscaled vector in the d_L or d_U space */
    SmartPtr<const Vector> unapply_vector_scaling_d_LU(
      const Matrix& Pd_LU,
      const SmartPtr<const Vector>& lu,
      const VectorSpace& d_space);
    //@}

    /** Methods for scaling the gradient of the objective - wraps the
     *  virtual methods above
     */
    //@{
    /** Returns a grad_f scaled version (d_f * D_x^{-1}) of the given vector */
    virtual SmartPtr<Vector>
    apply_grad_obj_scaling_NonConst(const SmartPtr<const Vector>& v);
    /** Returns a grad_f scaled version (d_f * D_x^{-1}) of the given vector */
    virtual SmartPtr<const Vector>
    apply_grad_obj_scaling(const SmartPtr<const Vector>& v);
    /** Returns a grad_f unscaled version (d_f * D_x^{-1}) of the
     *  given vector */
    virtual SmartPtr<Vector>
    unapply_grad_obj_scaling_NonConst(const SmartPtr<const Vector>& v);
    /** Returns a grad_f unscaled version (d_f * D_x^{-1}) of the
     *  given vector */
    virtual SmartPtr<const Vector>
    unapply_grad_obj_scaling(const SmartPtr<const Vector>& v);
    //@}

    /** @name Methods for determining whether scaling for entities is
     *  done */
    //@{
    /** Returns true if the primal x variables are scaled. */
    virtual bool have_x_scaling()=0;
    /** Returns true if the equality constraints are scaled. */
    virtual bool have_c_scaling()=0;
    /** Returns true if the inequality constraints are scaled. */
    virtual bool have_d_scaling()=0;
    //@}

    /** This method is called by the IpoptNLP's at a convenient time to
     *  compute and/or read scaling factors 
     */
    virtual void DetermineScaling(const SmartPtr<const VectorSpace> x_space,
                                  const SmartPtr<const VectorSpace> c_space,
                                  const SmartPtr<const VectorSpace> d_space,
                                  const SmartPtr<const MatrixSpace> jac_c_space,
                                  const SmartPtr<const MatrixSpace> jac_d_space,
                                  const SmartPtr<const SymMatrixSpace> h_space,
                                  SmartPtr<const MatrixSpace>& new_jac_c_space,
                                  SmartPtr<const MatrixSpace>& new_jac_d_space,
                                  SmartPtr<const SymMatrixSpace>& new_h_space,
                                  const Matrix& Px_L, const Vector& x_L,
                                  const Matrix& Px_U, const Vector& x_U)=0;
  protected:
    /** Implementation of the initialization method that has to be
     *  overloaded by for each derived class. */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix)=0;

    /** Accessor method for the journalist */
    const Journalist& Jnlst() const
    {
      return *jnlst_;
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
    NLPScalingObject(const NLPScalingObject&);

    /** Overloaded Equals Operator */
    void operator=(const NLPScalingObject&);
    //@}

    SmartPtr<const Journalist> jnlst_;
  };

  /** This is a base class for many standard scaling
   *  techniques. The overloaded classes only need to
   *  provide the scaling parameters
   */
  class StandardScalingBase : public NLPScalingObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    StandardScalingBase();

    /** Default destructor */
    virtual ~StandardScalingBase();
    //@}

    /** Methods to map scaled and unscaled matrices */
    //@{
    /** Returns an obj-scaled version of the given scalar */
    virtual Number apply_obj_scaling(const Number& f);
    /** Returns an obj-unscaled version of the given scalar */
    virtual Number unapply_obj_scaling(const Number& f);
    /** Returns an x-scaled version of the given vector */
    virtual SmartPtr<Vector>
    apply_vector_scaling_x_NonConst(const SmartPtr<const Vector>& v);
    /** Returns an x-scaled version of the given vector */
    virtual SmartPtr<const Vector>
    apply_vector_scaling_x(const SmartPtr<const Vector>& v);
    /** Returns an x-unscaled version of the given vector */
    virtual SmartPtr<Vector>
    unapply_vector_scaling_x_NonConst(const SmartPtr<const Vector>& v);
    /** Returns an x-unscaled version of the given vector */
    virtual SmartPtr<const Vector>
    unapply_vector_scaling_x(const SmartPtr<const Vector>& v);
    /** Returns an c-scaled version of the given vector */
    virtual SmartPtr<const Vector>
    apply_vector_scaling_c(const SmartPtr<const Vector>& v);
    /** Returns an c-unscaled version of the given vector */
    virtual SmartPtr<const Vector>
    unapply_vector_scaling_c(const SmartPtr<const Vector>& v);
    /** Returns an c-scaled version of the given vector */
    virtual SmartPtr<Vector>
    apply_vector_scaling_c_NonConst(const SmartPtr<const Vector>& v);
    /** Returns an c-unscaled version of the given vector */
    virtual SmartPtr<Vector>
    unapply_vector_scaling_c_NonConst(const SmartPtr<const Vector>& v);
    /** Returns an d-scaled version of the given vector */
    virtual SmartPtr<const Vector>
    apply_vector_scaling_d(const SmartPtr<const Vector>& v);
    /** Returns an d-unscaled version of the given vector */
    virtual SmartPtr<const Vector>
    unapply_vector_scaling_d(const SmartPtr<const Vector>& v);
    /** Returns an d-scaled version of the given vector */
    virtual SmartPtr<Vector>
    apply_vector_scaling_d_NonConst(const SmartPtr<const Vector>& v);
    /** Returns an d-unscaled version of the given vector */
    virtual SmartPtr<Vector>
    unapply_vector_scaling_d_NonConst(const SmartPtr<const Vector>& v);
    /** Returns a scaled version of the jacobian for c.  If the
     *  overloaded method does not make a new matrix, make sure to set
     *  the matrix ptr passed in to NULL.
     */
    virtual SmartPtr<const Matrix>
    apply_jac_c_scaling(SmartPtr<const Matrix> matrix);
    /** Returns a scaled version of the jacobian for d If the
     *  overloaded method does not create a new matrix, make sure to
     *  set the matrix ptr passed in to NULL.
     */
    virtual SmartPtr<const Matrix>
    apply_jac_d_scaling(SmartPtr<const Matrix> matrix);
    /** Returns a scaled version of the hessian of the lagrangian If
     *  the overloaded method does not create a new matrix, make sure
     *  to set the matrix ptr passed in to NULL.
     */
    virtual SmartPtr<const SymMatrix>
    apply_hessian_scaling(SmartPtr<const SymMatrix> matrix);
    //@}

    /** @name Methods for determining whether scaling for entities is
     *  done */
    //@{
    virtual bool have_x_scaling();
    virtual bool have_c_scaling();
    virtual bool have_d_scaling();
    //@}

    /** This method is called by the IpoptNLP's at a convenient time to
     *  compute and/or read scaling factors 
     */
    virtual void DetermineScaling(const SmartPtr<const VectorSpace> x_space,
                                  const SmartPtr<const VectorSpace> c_space,
                                  const SmartPtr<const VectorSpace> d_space,
                                  const SmartPtr<const MatrixSpace> jac_c_space,
                                  const SmartPtr<const MatrixSpace> jac_d_space,
                                  const SmartPtr<const SymMatrixSpace> h_space,
                                  SmartPtr<const MatrixSpace>& new_jac_c_space,
                                  SmartPtr<const MatrixSpace>& new_jac_d_space,
                                  SmartPtr<const SymMatrixSpace>& new_h_space,
                                  const Matrix& Px_L, const Vector& x_L,
                                  const Matrix& Px_U, const Vector& x_U);

    /** Methods for IpoptType */
    //@{
    static void RegisterOptions(SmartPtr<RegisteredOptions> roptions);
    //@}

  protected:
    /** Overloaded initialization method */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix);

    /** This is the method that has to be overloaded by a particular
     *  scaling method that somehow computes the scaling vectors dx,
     *  dc, and dd.  The pointers to those vectors can be NULL, in
     *  which case no scaling for that item will be done later. */
    virtual void DetermineScalingParametersImpl(
      const SmartPtr<const VectorSpace> x_space,
      const SmartPtr<const VectorSpace> c_space,
      const SmartPtr<const VectorSpace> d_space,
      const SmartPtr<const MatrixSpace> jac_c_space,
      const SmartPtr<const MatrixSpace> jac_d_space,
      const SmartPtr<const SymMatrixSpace> h_space,
      const Matrix& Px_L, const Vector& x_L,
      const Matrix& Px_U, const Vector& x_U,
      Number& df,
      SmartPtr<Vector>& dx,
      SmartPtr<Vector>& dc,
      SmartPtr<Vector>& dd)=0;

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
    StandardScalingBase(const StandardScalingBase&);

    /** Overloaded Equals Operator */
    void operator=(const StandardScalingBase&);
    //@}

    /** Scaling parameters - we only need to keep copies of
     *  the objective scaling and the x scaling - the others we can
     *  get from the scaled matrix spaces.
     */
    //@{
    /** objective scaling parameter */
    Number df_;
    /** x scaling */
    SmartPtr<Vector> dx_;
    //@}

    /** Scaled Matrix Spaces */
    //@{
    /** Scaled jacobian of c space */
    SmartPtr<ScaledMatrixSpace> scaled_jac_c_space_;
    /** Scaled jacobian of d space */
    SmartPtr<ScaledMatrixSpace> scaled_jac_d_space_;
    /** Scaled hessian of lagrangian spacea */
    SmartPtr<SymScaledMatrixSpace> scaled_h_space_;
    //@}

    /** @name Algorithmic parameters */
    //@{
    /** Additional scaling value for the objective function */
    Number obj_scaling_factor_;
    //@}
  };

  /** Class implementing the scaling object that doesn't to any scaling */
  class NoNLPScalingObject : public StandardScalingBase
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    NoNLPScalingObject()
    {}

    /** Default destructor */
    virtual ~NoNLPScalingObject()
    {}
    //@}


  protected:
    /** Overloaded from StandardScalingBase */
    virtual void DetermineScalingParametersImpl(
      const SmartPtr<const VectorSpace> x_space,
      const SmartPtr<const VectorSpace> c_space,
      const SmartPtr<const VectorSpace> d_space,
      const SmartPtr<const MatrixSpace> jac_c_space,
      const SmartPtr<const MatrixSpace> jac_d_space,
      const SmartPtr<const SymMatrixSpace> h_space,
      const Matrix& Px_L, const Vector& x_L,
      const Matrix& Px_U, const Vector& x_U,
      Number& df,
      SmartPtr<Vector>& dx,
      SmartPtr<Vector>& dc,
      SmartPtr<Vector>& dd);

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
    NoNLPScalingObject(const NoNLPScalingObject&);

    /** Overloaded Equals Operator */
    void operator=(const NoNLPScalingObject&);
    //@}
  };

} // namespace Ipopt

#endif
