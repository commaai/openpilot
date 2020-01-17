/*************************************************************************
   Copyright (C) 2004, 2010 International Business Machines and others.
   All Rights Reserved.
   This code is published under the Eclipse Public License.
 
   $Id: IpStdCInterface.h 2082 2012-02-16 03:00:34Z andreasw $
 
   Authors:  Carl Laird, Andreas Waechter     IBM    2004-09-02
 *************************************************************************/

#ifndef __IPSTDCINTERFACE_H__
#define __IPSTDCINTERFACE_H__

#ifndef IPOPT_EXPORT
#ifdef _MSC_VER
#ifdef IPOPT_DLL
#define IPOPT_EXPORT(type) __declspec(dllexport) type __cdecl
#else
#define IPOPT_EXPORT(type) type __cdecl
#endif
#else 
#define IPOPT_EXPORT(type) type
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /** Type for all number.  We need to make sure that this is
      identical with what is defined in Common/IpTypes.hpp */
  typedef double Number;

  /** Type for all incides.  We need to make sure that this is
      identical with what is defined in Common/IpTypes.hpp */
  typedef int Index;

  /** Type for all integers.  We need to make sure that this is
      identical with what is defined in Common/IpTypes.hpp */
  typedef int Int;

  /* This includes the SolverReturn enum type */
#include "IpReturnCodes.h"

  /** Structure collecting all information about the problem
   *  definition and solve statistics etc.  This is defined in the
   *  source file. */
  struct IpoptProblemInfo;

  /** Pointer to a Ipopt Problem. */
  typedef struct IpoptProblemInfo* IpoptProblem;

  /** define a boolean type for C */
  typedef int Bool;
#ifndef TRUE
# define TRUE (1)
#endif
#ifndef FALSE
# define FALSE (0)
#endif

  /** A pointer for anything that is to be passed between the called
   *  and individual callback function */
  typedef void * UserDataPtr;

  /** Type defining the callback function for evaluating the value of
   *  the objective function.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_F_CB)(Index n, Number* x, Bool new_x,
                            Number* obj_value, UserDataPtr user_data);

  /** Type defining the callback function for evaluating the gradient of
   *  the objective function.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_Grad_F_CB)(Index n, Number* x, Bool new_x,
                                 Number* grad_f, UserDataPtr user_data);

  /** Type defining the callback function for evaluating the value of
   *  the constraint functions.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_G_CB)(Index n, Number* x, Bool new_x,
                            Index m, Number* g, UserDataPtr user_data);

  /** Type defining the callback function for evaluating the Jacobian of
   *  the constrant functions.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_Jac_G_CB)(Index n, Number *x, Bool new_x,
                                Index m, Index nele_jac,
                                Index *iRow, Index *jCol, Number *values,
                                UserDataPtr user_data);

  /** Type defining the callback function for evaluating the Hessian of
   *  the Lagrangian function.  Return value should be set to false if
   *  there was a problem doing the evaluation. */
  typedef Bool (*Eval_H_CB)(Index n, Number *x, Bool new_x, Number obj_factor,
                            Index m, Number *lambda, Bool new_lambda,
                            Index nele_hess, Index *iRow, Index *jCol,
                            Number *values, UserDataPtr user_data);

  /** Type defining the callback function for giving intermediate
   *  execution control to the user.  If set, it is called once per
   *  iteration, providing the user with some information on the state
   *  of the optimization.  This can be used to print some
   *  user-defined output.  It also gives the user a way to terminate
   *  the optimization prematurely.  If this method returns false,
   *  Ipopt will terminate the optimization. */
  typedef Bool (*Intermediate_CB)(Index alg_mod, /* 0 is regular, 1 is resto */
				  Index iter_count, Number obj_value,
				  Number inf_pr, Number inf_du,
				  Number mu, Number d_norm,
				  Number regularization_size,
				  Number alpha_du, Number alpha_pr,
				  Index ls_trials, UserDataPtr user_data);

  /** Function for creating a new Ipopt Problem object.  This function
   *  returns an object that can be passed to the IpoptSolve call.  It
   *  contains the basic definition of the optimization problem, such
   *  as number of variables and constraints, bounds on variables and
   *  constraints, information about the derivatives, and the callback
   *  function for the computation of the optimization problem
   *  functions and derivatives.  During this call, the options file
   *  PARAMS.DAT is read as well.
   *
   *  If NULL is returned, there was a problem with one of the inputs
   *  or reading the options file. */
  IPOPT_EXPORT(IpoptProblem) CreateIpoptProblem(
      Index n             /** Number of optimization variables */
    , Number* x_L         /** Lower bounds on variables. This array of
                              size n is copied internally, so that the
                              caller can change the incoming data after
                              return without that IpoptProblem is
                              modified.  Any value less or equal than
                              the number specified by option
                              'nlp_lower_bound_inf' is interpreted to
                              be minus infinity. */
    , Number* x_U         /** Upper bounds on variables. This array of
                              size n is copied internally, so that the
                              caller can change the incoming data after
                              return without that IpoptProblem is
                              modified.  Any value greater or equal
                              than the number specified by option
                              'nlp_upper_bound_inf' is interpreted to
                              be plus infinity. */
    , Index m             /** Number of constraints. */
    , Number* g_L         /** Lower bounds on constraints. This array of
                              size m is copied internally, so that the
                              caller can change the incoming data after
                              return without that IpoptProblem is
                              modified.  Any value less or equal than
                              the number specified by option
                              'nlp_lower_bound_inf' is interpreted to
                              be minus infinity. */
    , Number* g_U         /** Upper bounds on constraints. This array of
                              size m is copied internally, so that the
                              caller can change the incoming data after
                              return without that IpoptProblem is
                              modified.  Any value greater or equal
                              than the number specified by option
                              'nlp_upper_bound_inf' is interpreted to
                              be plus infinity. */
    , Index nele_jac      /** Number of non-zero elements in constraint
                              Jacobian. */
    , Index nele_hess     /** Number of non-zero elements in Hessian of
                              Lagrangian. */
    , Index index_style   /** indexing style for iRow & jCol,
				 0 for C style, 1 for Fortran style */
    , Eval_F_CB eval_f    /** Callback function for evaluating
                              objective function */
    , Eval_G_CB eval_g    /** Callback function for evaluating
                              constraint functions */
    , Eval_Grad_F_CB eval_grad_f
                          /** Callback function for evaluating gradient
                              of objective function */
    , Eval_Jac_G_CB eval_jac_g
                          /** Callback function for evaluating Jacobian
                              of constraint functions */
    , Eval_H_CB eval_h    /** Callback function for evaluating Hessian
                              of Lagrangian function */
  );

  /** Method for freeing a previously created IpoptProblem.  After
      freeing an IpoptProblem, it cannot be used anymore. */
  IPOPT_EXPORT(void) FreeIpoptProblem(IpoptProblem ipopt_problem);


  /** Function for adding a string option.  Returns FALSE the option
   *  could not be set (e.g., if keyword is unknown) */
  IPOPT_EXPORT(Bool) AddIpoptStrOption(IpoptProblem ipopt_problem, char* keyword, char* val);

  /** Function for adding a Number option.  Returns FALSE the option
   *  could not be set (e.g., if keyword is unknown) */
  IPOPT_EXPORT(Bool) AddIpoptNumOption(IpoptProblem ipopt_problem, char* keyword, Number val);

  /** Function for adding an Int option.  Returns FALSE the option
   *  could not be set (e.g., if keyword is unknown) */
  IPOPT_EXPORT(Bool) AddIpoptIntOption(IpoptProblem ipopt_problem, char* keyword, Int val);

  /** Function for opening an output file for a given name with given
   *  printlevel.  Returns false, if there was a problem opening the
   *  file. */
  IPOPT_EXPORT(Bool) OpenIpoptOutputFile(IpoptProblem ipopt_problem, char* file_name,
                           Int print_level);

  /** Optional function for setting scaling parameter for the NLP.
   *  This corresponds to the get_scaling_parameters method in TNLP.
   *  If the pointers x_scaling or g_scaling are NULL, then no scaling
   *  for x resp. g is done. */
  IPOPT_EXPORT(Bool) SetIpoptProblemScaling(IpoptProblem ipopt_problem,
			      Number obj_scaling,
			      Number* x_scaling,
			      Number* g_scaling);

  /** Setting a callback function for the "intermediate callback"
   *  method in the TNLP.  This gives control back to the user once
   *  per iteration.  If set, it provides the user with some
   *  information on the state of the optimization.  This can be used
   *  to print some user-defined output.  It also gives the user a way
   *  to terminate the optimization prematurely.  If the callback
   *  method returns false, Ipopt will terminate the optimization.
   *  Calling this set method to set the CB pointer to NULL disables
   *  the intermediate callback functionality. */
  IPOPT_EXPORT(Bool) SetIntermediateCallback(IpoptProblem ipopt_problem,
					     Intermediate_CB intermediate_cb);

  /** Function calling the Ipopt optimization algorithm for a problem
      previously defined with CreateIpoptProblem.  The return
      specified outcome of the optimization procedure (e.g., success,
      failure etc).
   */
  IPOPT_EXPORT(enum ApplicationReturnStatus) IpoptSolve(
      IpoptProblem ipopt_problem
                         /** Problem that is to be optimized.  Ipopt
                             will use the options previously specified with
                             AddIpoptOption (etc) for this problem. */
    , Number* x          /** Input:  Starting point
                             Output: Optimal solution */
    , Number* g          /** Values of constraint at final point
                             (output only - ignored if set to NULL) */
    , Number* obj_val    /** Final value of objective function
                             (output only - ignored if set to NULL) */
    , Number* mult_g     /** Input: Initial values for the constraint
                                    multipliers (only if warm start option
                                    is chosen)
                             Output: Final multipliers for constraints
                                     (ignored if set to NULL) */
    , Number* mult_x_L   /** Input: Initial values for the multipliers for
                                    lower variable bounds (only if warm start
                                    option is chosen)
                             Output: Final multipliers for lower variable
                                     bounds (ignored if set to NULL) */
    , Number* mult_x_U   /** Input: Initial values for the multipliers for
                                    upper variable bounds (only if warm start
                                    option is chosen)
                             Output: Final multipliers for upper variable
                                     bounds (ignored if set to NULL) */
    , UserDataPtr user_data
                         /** Pointer to user data.  This will be
                             passed unmodified to the callback
                             functions. */
  );

  /**
  void IpoptStatisticsCounts;

  void IpoptStatisticsInfeasibilities; */
#ifdef __cplusplus
} /* extern "C" { */
#endif

#endif
