// Copyright (C) 2008 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// $Id: IpTNLPReducer.hpp 1861 2010-12-21 21:34:47Z andreasw $
//
// Authors:  Andreas Waechter                  IBM    2008-08-10

#ifndef __IPTNLPREDUCER_HPP__
#define __IPTNLPREDUCER_HPP__

#include "IpTNLP.hpp"

namespace Ipopt
{
  /** This is a wrapper around a given TNLP class that takes out a
   *  list of constraints that are given to the constructor.  It is
   *  provided for convenience, if one wants to experiment with
   *  problems that consist of only a subset of the constraints.  But
   *  keep in mind that this is not efficient, since behind the scenes
   *  we are still evaluation all functions and derivatives, and are
   *  making copies of the original data. */
  class TNLPReducer : public TNLP
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Constructor is given the indices of the constraints that
     *  should be taken out of the problem statement, as well as the
     *  original TNLP. */
    TNLPReducer(TNLP& tnlp, Index n_g_skip, const Index* index_g_skip,
                Index n_xL_skip, const Index* index_xL_skip,
                Index n_xU_skip, const Index* index_xU_skip,
                Index n_x_fix, const Index* index_f_fix);

    /** Default destructor */
    virtual ~TNLPReducer();
    //@}

    /** @name Overloaded methods from TNLP */
    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                              Index& nnz_h_lag, IndexStyleEnum& index_style);

    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                                 Index m, Number* g_l, Number* g_u);

    virtual bool get_scaling_parameters(Number& obj_scaling,
                                        bool& use_x_scaling, Index n,
                                        Number* x_scaling,
                                        bool& use_g_scaling, Index m,
                                        Number* g_scaling);

    virtual bool get_variables_linearity(Index n, LinearityType* var_types);

    virtual bool get_constraints_linearity(Index m, LinearityType* const_types);

    virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda,
                                    Number* lambda);

    virtual bool get_warm_start_iterate(IteratesVector& warm_start_iterate);

    virtual bool eval_f(Index n, const Number* x, bool new_x,
                        Number& obj_value);

    virtual bool eval_grad_f(Index n, const Number* x, bool new_x,
                             Number* grad_f);

    virtual bool eval_g(Index n, const Number* x, bool new_x,
                        Index m, Number* g);

    virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow,
                            Index *jCol, Number* values);

    virtual bool eval_h(Index n, const Number* x, bool new_x,
                        Number obj_factor, Index m, const Number* lambda,
                        bool new_lambda, Index nele_hess,
                        Index* iRow, Index* jCol, Number* values);

    virtual void finalize_solution(SolverReturn status,
                                   Index n, const Number* x, const Number* z_L, const Number* z_U,
                                   Index m, const Number* g, const Number* lambda,
                                   Number obj_value,
                                   const IpoptData* ip_data,
                                   IpoptCalculatedQuantities* ip_cq);

    virtual bool intermediate_callback(AlgorithmMode mode,
                                       Index iter, Number obj_value,
                                       Number inf_pr, Number inf_du,
                                       Number mu, Number d_norm,
                                       Number regularization_size,
                                       Number alpha_du, Number alpha_pr,
                                       Index ls_trials,
                                       const IpoptData* ip_data,
                                       IpoptCalculatedQuantities* ip_cq);

    virtual Index get_number_of_nonlinear_variables();

    virtual bool get_list_of_nonlinear_variables(Index num_nonlin_vars,
        Index* pos_nonlin_vars);
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
    TNLPReducer();

    /** Copy Constructor */
    TNLPReducer(const TNLPReducer&);

    /** Overloaded Equals Operator */
    void operator=(const TNLPReducer&);
    //@}

    /** @name original TNLP */
    //@{
    SmartPtr<TNLP> tnlp_;
    Index m_orig_;
    Index nnz_jac_g_orig_;
    //@}

    /** Number of constraints to be skipped */
    Index n_g_skip_;

    /** Array of indices of the constraints that are to be skipped.
     *  This is provided at the beginning in the constructor. */
    Index* index_g_skip_;

    /** Index style for original problem.  Internally, we use C-Style
     *  now. */
    IndexStyleEnum index_style_orig_;

    /** Map from original constraints to new constraints.  A -1 means
     *  that a constraint is skipped. */
    Index* g_keep_map_;

    /** Number of constraints in reduced NLP */
    Index m_reduced_;

    /** Number of Jacobian nonzeros in the reduced NLP */
    Index nnz_jac_g_reduced_;

    /** Number of Jacobian nonzeros that are skipped */
    Index nnz_jac_g_skipped_;

    /** Array of Jacobian elements that are to be skipped.  This is in
     *  increasing order. */
    Index* jac_g_skipped_;

    /** Number of lower variable bounds to be skipped. */
    Index n_xL_skip_;

    /** Array of indices of the lower variable bounds to be skipped. */
    Index* index_xL_skip_;

    /** Number of upper variable bounds to be skipped. */
    Index n_xU_skip_;

    /** Array of indices of the upper variable bounds to be skipped. */
    Index* index_xU_skip_;

    /** Number of variables that are to be fixed to initial value. */
    Index n_x_fix_;

    /** Array of indices of the variables that are to be fixed. */
    Index* index_x_fix_;
  };

} // namespace Ipopt

#endif
