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

#ifndef HPIPM_D_CORE_QP_IPM_
#define HPIPM_D_CORE_QP_IPM_

#include "hpipm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct d_core_qp_ipm_workspace
	{
	double *v; // primal variables
	double *pi; // equality constraints multipliers
	double *lam; // inequality constraints multipliers
	double *t; // inequality constraints slacks
	double *t_inv; // inverse of t
	double *v_bkp; // backup of primal variables
	double *pi_bkp; // backup of equality constraints multipliers
	double *lam_bkp; // backup of inequality constraints multipliers
	double *t_bkp; // backup of inequality constraints slacks
	double *dv; // step in v
	double *dpi; // step in pi
	double *dlam; // step in lam
	double *dt; // step in t
	double *res_g; // q-residuals
	double *res_b; // b-residuals
	double *res_d; // d-residuals
	double *res_m; // m-residuals
	double *res_m_bkp; // m-residuals
	double *Gamma; // Hessian update
	double *gamma; // gradient update
	double alpha; // step length
	double alpha_prim; // step length
	double alpha_dual; // step length
	double sigma; // centering XXX
	double mu; // duality measuere
	double mu_aff; // affine duality measuere
	double nc_inv; // 1.0/nc, where nc is the total number of inequality constraints
	double nc_mask_inv; // 1.0/nc_mask
	double lam_min; // min value in lam vector
	double t_min; // min value in t vector
	double t_min_inv; // inverse of min value in t vector
	double tau_min; // min value of barrier parameter
	int nv; // number of primal variables
	int ne; // number of equality constraints
	int nc; // (twice the) number of (two-sided) inequality constraints
	int nc_mask; // total number of ineq constr after masking
	int split_step; // use different step for primal and dual variables
	int t_lam_min; // clip t and lam also in solution, or only in Gamma computation
	hpipm_size_t memsize; // memory size (in bytes) of workspace
	};



//
hpipm_size_t d_memsize_core_qp_ipm(int nv, int ne, int nc);
//
void d_create_core_qp_ipm(int nv, int ne, int nc, struct d_core_qp_ipm_workspace *workspace, void *mem);
//
void d_core_qp_ipm(struct d_core_qp_ipm_workspace *workspace);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // HPIPM_D_CORE_QP_IPM_
