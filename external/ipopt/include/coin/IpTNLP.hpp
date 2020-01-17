// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpTNLP.hpp 2212 2013-04-14 14:51:52Z stefan $
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPTNLP_HPP__
#define __IPTNLP_HPP__

#include "IpUtils.hpp"
#include "IpReferenced.hpp"
#include "IpException.hpp"
#include "IpAlgTypes.hpp"
#include "IpReturnCodes.hpp"

#include <map>

namespace Ipopt
{
  // forward declarations
  class IpoptData;
  class IpoptCalculatedQuantities;
  class IteratesVector;

  /** Base class for all NLP's that use standard triplet matrix form
   *  and dense vectors.  This is the standard base class for all
   *  NLP's that use the standard triplet matrix form (as for Harwell
   *  routines) and dense vectors. The class TNLPAdapter then converts
   *  this interface to an interface that can be used directly by
   *  ipopt.
   *
   *  This interface presents the problem form:
   *  
   *     min f(x)
   *
   *     s.t. gL <= g(x) <= gU
   *
   *          xL <=  x   <= xU
   *
   *  In order to specify an equality constraint, set gL_i = gU_i =
   *  rhs.  The value that indicates "infinity" for the bounds
   *  (i.e. the variable or constraint has no lower bound (-infinity)
   *  or upper bound (+infinity)) is set through the option
   *  nlp_lower_bound_inf and nlp_upper_bound_inf.  To indicate that a
   *  variable has no upper or lower bound, set the bound to
   *  -ipopt_inf or +ipopt_inf respectively
   */
  class TNLP : public ReferencedObject
  {
  public:
    /** Type of the constraints*/
    enum LinearityType
    {
      LINEAR/** Constraint/Variable is linear.*/,
      NON_LINEAR/**Constraint/Varaible is non-linear.*/
    };

    /**@name Constructors/Destructors */
    //@{
    TNLP()
    {}

    /** Default destructor */
    virtual ~TNLP()
    {}
    //@}

    DECLARE_STD_EXCEPTION(INVALID_TNLP);

    /**@name methods to gather information about the NLP */
    //@{
    /** overload this method to return the number of variables
     *  and constraints, and the number of non-zeros in the jacobian and
     *  the hessian. The index_style parameter lets you specify C or Fortran
     *  style indexing for the sparse matrix iRow and jCol parameters.
     *  C_STYLE is 0-based, and FORTRAN_STYLE is 1-based.
     */
    enum IndexStyleEnum { C_STYLE=0, FORTRAN_STYLE=1 };
    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                              Index& nnz_h_lag, IndexStyleEnum& index_style)=0;

    typedef std::map<std::string, std::vector<std::string> > StringMetaDataMapType;
    typedef std::map<std::string, std::vector<Index> > IntegerMetaDataMapType;
    typedef std::map<std::string, std::vector<Number> > NumericMetaDataMapType;

    /** overload this method to return any meta data for
     *  the variables and the constraints */
    virtual bool get_var_con_metadata(Index n,
                                      StringMetaDataMapType& var_string_md,
                                      IntegerMetaDataMapType& var_integer_md,
                                      NumericMetaDataMapType& var_numeric_md,
                                      Index m,
                                      StringMetaDataMapType& con_string_md,
                                      IntegerMetaDataMapType& con_integer_md,
                                      NumericMetaDataMapType& con_numeric_md)

    {
      return false;
    }

    /** overload this method to return the information about the bound
     *  on the variables and constraints. The value that indicates
     *  that a bound does not exist is specified in the parameters
     *  nlp_lower_bound_inf and nlp_upper_bound_inf.  By default,
     *  nlp_lower_bound_inf is -1e19 and nlp_upper_bound_inf is
     *  1e19. (see TNLPAdapter) */
    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                                 Index m, Number* g_l, Number* g_u)=0;

    /** overload this method to return scaling parameters. This is
     *  only called if the options are set to retrieve user scaling.
     *  There, use_x_scaling (or use_g_scaling) should get set to true
     *  only if the variables (or constraints) are to be scaled.  This
     *  method should return true only if the scaling parameters could
     *  be provided.
     */
    virtual bool get_scaling_parameters(Number& obj_scaling,
                                        bool& use_x_scaling, Index n,
                                        Number* x_scaling,
                                        bool& use_g_scaling, Index m,
                                        Number* g_scaling)
    {
      return false;
    }

    /** overload this method to return the variables linearity
     * (TNLP::LINEAR or TNLP::NON_LINEAR). The var_types
     *  array has been allocated with length at least n. (default implementation
     *  just return false and does not fill the array).*/
    virtual bool get_variables_linearity(Index n, LinearityType* var_types)
    {
      return false;
    }

    /** overload this method to return the constraint linearity.
     *  array has been allocated with length at least n. (default implementation
     *  just return false and does not fill the array).*/
    virtual bool get_constraints_linearity(Index m, LinearityType* const_types)
    {
      return false;
    }

    /** overload this method to return the starting point. The bool
     *  variables indicate whether the algorithm wants you to
     *  initialize x, z_L/z_u, and lambda, respectively.  If, for some
     *  reason, the algorithm wants you to initialize these and you
     *  cannot, return false, which will cause Ipopt to stop.  You
     *  will have to run Ipopt with different options then.
     */
    virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda,
                                    Number* lambda)=0;

    /** overload this method to provide an Ipopt iterate (already in
     *  the form Ipopt requires it internally) for a warm start.
     *  Since this is only for expert users, a default dummy
     *  implementation is provided and returns false. */
    virtual bool get_warm_start_iterate(IteratesVector& warm_start_iterate)
    {
      return false;
    }

    /** overload this method to return the value of the objective function */
    virtual bool eval_f(Index n, const Number* x, bool new_x,
                        Number& obj_value)=0;

    /** overload this method to return the vector of the gradient of
     *  the objective w.r.t. x */
    virtual bool eval_grad_f(Index n, const Number* x, bool new_x,
                             Number* grad_f)=0;

    /** overload this method to return the vector of constraint values */
    virtual bool eval_g(Index n, const Number* x, bool new_x,
                        Index m, Number* g)=0;
    /** overload this method to return the jacobian of the
     *  constraints. The vectors iRow and jCol only need to be set
     *  once. The first call is used to set the structure only (iRow
     *  and jCol will be non-NULL, and values will be NULL) For
     *  subsequent calls, iRow and jCol will be NULL. */
    virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow,
                            Index *jCol, Number* values)=0;

    /** overload this method to return the hessian of the
     *  lagrangian. The vectors iRow and jCol only need to be set once
     *  (during the first call). The first call is used to set the
     *  structure only (iRow and jCol will be non-NULL, and values
     *  will be NULL) For subsequent calls, iRow and jCol will be
     *  NULL. This matrix is symmetric - specify the lower diagonal
     *  only.  A default implementation is provided, in case the user
     *  wants to se quasi-Newton approximations to estimate the second
     *  derivatives and doesn't not neet to implement this method. */
    virtual bool eval_h(Index n, const Number* x, bool new_x,
                        Number obj_factor, Index m, const Number* lambda,
                        bool new_lambda, Index nele_hess,
                        Index* iRow, Index* jCol, Number* values)
    {
      return false;
    }
    //@}

    /** @name Solution Methods */
    //@{
    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(SolverReturn status,
                                   Index n, const Number* x, const Number* z_L, const Number* z_U,
                                   Index m, const Number* g, const Number* lambda,
                                   Number obj_value,
                                   const IpoptData* ip_data,
                                   IpoptCalculatedQuantities* ip_cq)=0;
    /** This method is called just before finalize_solution.  With
     *  this method, the algorithm returns any metadata collected
     *  during its run, including the metadata provided by the user
     *  with the above get_var_con_metadata.  Each metadata can be of
     *  type string, integer, and numeric. It can be associated to
     *  either the variables or the constraints.  The metadata that
     *  was associated with the primal variable vector is stored in
     *  var_..._md.  The metadata associated with the constraint
     *  multipliers is stored in con_..._md.  The metadata associated
     *  with the bound multipliers is stored in var_..._md, with the
     *  suffixes "_z_L", and "_z_U", denoting lower and upper
     *  bounds. */
    virtual void finalize_metadata(Index n,
                                   const StringMetaDataMapType& var_string_md,
                                   const IntegerMetaDataMapType& var_integer_md,
                                   const NumericMetaDataMapType& var_numeric_md,
                                   Index m,
                                   const StringMetaDataMapType& con_string_md,
                                   const IntegerMetaDataMapType& con_integer_md,
                                   const NumericMetaDataMapType& con_numeric_md)
    {}


    /** Intermediate Callback method for the user.  Providing dummy
     *  default implementation.  For details see IntermediateCallBack
     *  in IpNLP.hpp. */
    virtual bool intermediate_callback(AlgorithmMode mode,
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

    /** @name Methods for quasi-Newton approximation.  If the second
     *  derivatives are approximated by Ipopt, it is better to do this
     *  only in the space of nonlinear variables.  The following
     *  methods are call by Ipopt if the quasi-Newton approximation is
     *  selected.  If -1 is returned as number of nonlinear variables,
     *  Ipopt assumes that all variables are nonlinear.  Otherwise, it
     *  calls get_list_of_nonlinear_variables with an array into which
     *  the indices of the nonlinear variables should be written - the
     *  array has the lengths num_nonlin_vars, which is identical with
     *  the return value of get_number_of_nonlinear_variables().  It
     *  is assumed that the indices are counted starting with 1 in the
     *  FORTRAN_STYLE, and 0 for the C_STYLE. */
    //@{
    virtual Index get_number_of_nonlinear_variables()
    {
      return -1;
    }

    virtual bool get_list_of_nonlinear_variables(Index num_nonlin_vars,
        Index* pos_nonlin_vars)
    {
      return false;
    }
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
    //TNLP();

    /** Copy Constructor */
    TNLP(const TNLP&);

    /** Overloaded Equals Operator */
    void operator=(const TNLP&);
    //@}
  };

} // namespace Ipopt

#endif
