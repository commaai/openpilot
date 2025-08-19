/**************************************************************************************************
*                                                                                                 *
* This file is part of HPIPM.                                                                     *
*                                                                                                 *
* HPIPM -- High-Performance Interior Point Method.                                                *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#ifndef HPIPM_S_OCP_QP_IPM_H_
#define HPIPM_S_OCP_QP_IPM_H_



#include <blasfeo_target.h>
#include <blasfeo_common.h>

#include <hpipm_common.h>
#include <hpipm_s_ocp_qp_dim.h>
#include <hpipm_s_ocp_qp.h>
#include <hpipm_s_ocp_qp_res.h>
#include <hpipm_s_ocp_qp_sol.h>



#ifdef __cplusplus
extern "C" {
#endif



struct s_ocp_qp_ipm_arg
	{
	float mu0; // initial value for complementarity slackness
	float alpha_min; // exit cond on step length
	float res_g_max; // exit cond on inf norm of residuals
	float res_b_max; // exit cond on inf norm of residuals
	float res_d_max; // exit cond on inf norm of residuals
	float res_m_max; // exit cond on inf norm of residuals
	float reg_prim; // reg of primal hessian
	float lam_min; // min value in lam vector
	float t_min; // min value in t vector
	float tau_min; // min value of barrier parameter
	int iter_max; // exit cond in iter number
	int stat_max; // iterations saved in stat
	int pred_corr; // use Mehrotra's predictor-corrector IPM algirthm
	int cond_pred_corr; // conditional Mehrotra's predictor-corrector
	int itref_pred_max; // max number of iterative refinement steps for predictor step
	int itref_corr_max; // max number of iterative refinement steps for corrector step
	int warm_start; // 0 no warm start, 1 warm start primal sol, 2 warm start primal and dual sol
	int square_root_alg; // 0 classical Riccati, 1 square-root Riccati
	int lq_fact; // 0 syrk+potrf, 1 mix, 2 lq (for square_root_alg==1)
	int abs_form; // absolute IPM formulation
	int comp_dual_sol_eq; // dual solution of equality constraints (only for abs_form==1)
	int comp_res_exit; // compute residuals on exit (only for abs_form==1 and comp_dual_sol_eq==1)
	int comp_res_pred; // compute residuals of prediction
	int split_step; // use different steps for primal and dual variables
	int var_init_scheme; // variables initialization scheme
	int t_lam_min; // clip t and lam: 0 no, 1 in Gamma computation, 2 in solution
	int mode;
	hpipm_size_t memsize;
	};



struct s_ocp_qp_ipm_ws
	{
	struct s_core_qp_ipm_workspace *core_workspace;
	struct s_ocp_qp_res_ws *res_workspace;
	struct s_ocp_qp_dim *dim; // cache dim
	struct s_ocp_qp_sol *sol_step;
	struct s_ocp_qp_sol *sol_itref;
	struct s_ocp_qp *qp_step;
	struct s_ocp_qp *qp_itref;
	struct s_ocp_qp_res *res;
	struct s_ocp_qp_res *res_itref;
	struct blasfeo_svec *Gamma; // hessian update
	struct blasfeo_svec *gamma; // hessian update
	struct blasfeo_svec *tmp_nuxM; // work space of size nxM
	struct blasfeo_svec *tmp_nbgM; // work space of size nbM+ngM
	struct blasfeo_svec *tmp_nsM; // work space of size nsM
	struct blasfeo_svec *Pb; // Pb
	struct blasfeo_svec *Zs_inv;
	struct blasfeo_svec *tmp_m;
	struct blasfeo_svec *l; // cache linear part for _get_ric_xxx
	struct blasfeo_smat *L;
	struct blasfeo_smat *Ls;
	struct blasfeo_smat *P;
	struct blasfeo_smat *Lh;
	struct blasfeo_smat *AL;
	struct blasfeo_smat *lq0;
	struct blasfeo_smat *tmp_nxM_nxM;
	float *stat; // convergence statistics
	int *use_hess_fact;
	void *lq_work0;
	float qp_res[4]; // infinity norm of residuals
	int iter; // iteration number
	int stat_max; // iterations saved in stat
	int stat_m; // number of recorded stat per IPM iter
	int use_Pb;
	int status; // solver status
	int square_root_alg; // cache from arg
	int lq_fact; // cache from arg
	int mask_constr; // use constr mask
	int valid_ric_vec; // meaningful riccati vectors
	int valid_ric_p; // form of riccati p: 0 p*inv(L), 1 p
	hpipm_size_t memsize;
	};



//
hpipm_size_t s_ocp_qp_ipm_arg_strsize();
//
hpipm_size_t s_ocp_qp_ipm_arg_memsize(struct s_ocp_qp_dim *ocp_dim);
//
void s_ocp_qp_ipm_arg_create(struct s_ocp_qp_dim *ocp_dim, struct s_ocp_qp_ipm_arg *arg, void *mem);
//
void s_ocp_qp_ipm_arg_set_default(enum hpipm_mode mode, struct s_ocp_qp_ipm_arg *arg);
//
void s_ocp_qp_ipm_arg_set(char *field, void *value, struct s_ocp_qp_ipm_arg *arg);
// set maximum number of iterations
void s_ocp_qp_ipm_arg_set_iter_max(int *iter_max, struct s_ocp_qp_ipm_arg *arg);
// set minimum step lenght
void s_ocp_qp_ipm_arg_set_alpha_min(float *alpha_min, struct s_ocp_qp_ipm_arg *arg);
// set initial value of barrier parameter
void s_ocp_qp_ipm_arg_set_mu0(float *mu0, struct s_ocp_qp_ipm_arg *arg);
// set exit tolerance on stationarity condition
void s_ocp_qp_ipm_arg_set_tol_stat(float *tol_stat, struct s_ocp_qp_ipm_arg *arg);
// set exit tolerance on equality constr
void s_ocp_qp_ipm_arg_set_tol_eq(float *tol_eq, struct s_ocp_qp_ipm_arg *arg);
// set exit tolerance on inequality constr
void s_ocp_qp_ipm_arg_set_tol_ineq(float *tol_ineq, struct s_ocp_qp_ipm_arg *arg);
// set exit tolerance on complementarity condition
void s_ocp_qp_ipm_arg_set_tol_comp(float *tol_comp, struct s_ocp_qp_ipm_arg *arg);
// set regularization of primal variables
void s_ocp_qp_ipm_arg_set_reg_prim(float *tol_comp, struct s_ocp_qp_ipm_arg *arg);
// set warm start: 0 no warm start, 1 primal var
void s_ocp_qp_ipm_arg_set_warm_start(int *warm_start, struct s_ocp_qp_ipm_arg *arg);
// Mehrotra's predictor-corrector IPM algorithm: 0 no predictor-corrector, 1 use predictor-corrector
void s_ocp_qp_ipm_arg_set_pred_corr(int *pred_corr, struct s_ocp_qp_ipm_arg *arg);
// conditional predictor-corrector: 0 no conditinal predictor-corrector, 1 conditional predictor-corrector
void s_ocp_qp_ipm_arg_set_cond_pred_corr(int *value, struct s_ocp_qp_ipm_arg *arg);
// set riccati algorithm: 0 classic, 1 square-root
void s_ocp_qp_ipm_arg_set_ric_alg(int *alg, struct s_ocp_qp_ipm_arg *arg);
// dual solution of equality constraints (only for abs_form==1)
void s_ocp_qp_ipm_arg_set_comp_dual_sol_eq(int *value, struct s_ocp_qp_ipm_arg *arg);
// compute residuals after solution
void s_ocp_qp_ipm_arg_set_comp_res_exit(int *value, struct s_ocp_qp_ipm_arg *arg);
// compute residuals of prediction
void s_ocp_qp_ipm_arg_set_comp_res_pred(int *alg, struct s_ocp_qp_ipm_arg *arg);
// min value of lam in the solution
void s_ocp_qp_ipm_arg_set_lam_min(float *value, struct s_ocp_qp_ipm_arg *arg);
// min value of t in the solution
void s_ocp_qp_ipm_arg_set_t_min(float *value, struct s_ocp_qp_ipm_arg *arg);
// min value of tau in the solution
void s_ocp_qp_ipm_arg_set_tau_min(float *value, struct s_ocp_qp_ipm_arg *arg);
// set split step: 0 same step, 1 different step for primal and dual variables
void s_ocp_qp_ipm_arg_set_split_step(int *value, struct s_ocp_qp_ipm_arg *arg);
// variables initialization scheme
void s_ocp_qp_ipm_arg_set_var_init_scheme(int *value, struct s_ocp_qp_ipm_arg *arg);
// clip t and lam: 0 no, 1 in Gamma computation, 2 in solution
void s_ocp_qp_ipm_arg_set_t_lam_min(int *value, struct s_ocp_qp_ipm_arg *arg);

//
hpipm_size_t s_ocp_qp_ipm_ws_strsize();
//
hpipm_size_t s_ocp_qp_ipm_ws_memsize(struct s_ocp_qp_dim *ocp_dim, struct s_ocp_qp_ipm_arg *arg);
//
void s_ocp_qp_ipm_ws_create(struct s_ocp_qp_dim *ocp_dim, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws, void *mem);
//
void s_ocp_qp_ipm_get(char *field, struct s_ocp_qp_ipm_ws *ws, void *value);
//
void s_ocp_qp_ipm_get_status(struct s_ocp_qp_ipm_ws *ws, int *status);
//
void s_ocp_qp_ipm_get_iter(struct s_ocp_qp_ipm_ws *ws, int *iter);
//
void s_ocp_qp_ipm_get_max_res_stat(struct s_ocp_qp_ipm_ws *ws, float *res_stat);
//
void s_ocp_qp_ipm_get_max_res_eq(struct s_ocp_qp_ipm_ws *ws, float *res_eq);
//
void s_ocp_qp_ipm_get_max_res_ineq(struct s_ocp_qp_ipm_ws *ws, float *res_ineq);
//
void s_ocp_qp_ipm_get_max_res_comp(struct s_ocp_qp_ipm_ws *ws, float *res_comp);
//
void s_ocp_qp_ipm_get_stat(struct s_ocp_qp_ipm_ws *ws, float **stat);
//
void s_ocp_qp_ipm_get_stat_m(struct s_ocp_qp_ipm_ws *ws, int *stat_m);
//
void s_ocp_qp_ipm_get_ric_Lr(struct s_ocp_qp *qp, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws, int stage, float *Lr);
//
void s_ocp_qp_ipm_get_ric_Ls(struct s_ocp_qp *qp, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws, int stage, float *Ls);
//
void s_ocp_qp_ipm_get_ric_P(struct s_ocp_qp *qp, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws, int stage, float *P);
//
void s_ocp_qp_ipm_get_ric_lr(struct s_ocp_qp *qp, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws, int stage, float *lr);
//
void s_ocp_qp_ipm_get_ric_p(struct s_ocp_qp *qp, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws, int stage, float *p);
// feedback control gain in the form u = K x + k
void s_ocp_qp_ipm_get_ric_K(struct s_ocp_qp *qp, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws, int stage, float *K);
// feedback control gain in the form u = K x + k
void s_ocp_qp_ipm_get_ric_k(struct s_ocp_qp *qp, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws, int stage, float *k);
//
void s_ocp_qp_init_var(struct s_ocp_qp *qp, struct s_ocp_qp_sol *qp_sol, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws);
//
void s_ocp_qp_ipm_abs_step(int kk, struct s_ocp_qp *qp, struct s_ocp_qp_sol *qp_sol, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws);
//
void s_ocp_qp_ipm_delta_step(int kk, struct s_ocp_qp *qp, struct s_ocp_qp_sol *qp_sol, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws);
//
void s_ocp_qp_ipm_solve(struct s_ocp_qp *qp, struct s_ocp_qp_sol *qp_sol, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws);
//
void s_ocp_qp_ipm_predict(struct s_ocp_qp *qp, struct s_ocp_qp_sol *qp_sol, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws);
//
void s_ocp_qp_ipm_sens(struct s_ocp_qp *qp, struct s_ocp_qp_sol *qp_sol, struct s_ocp_qp_ipm_arg *arg, struct s_ocp_qp_ipm_ws *ws);



#ifdef __cplusplus
} /* extern "C" */
#endif



#endif // HPIPM_S_OCP_QP_IPM_H_
