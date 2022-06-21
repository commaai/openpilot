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

#ifndef HPIPM_D_OCP_QP_H_
#define HPIPM_D_OCP_QP_H_



#include <blasfeo_target.h>
#include <blasfeo_common.h>

#include "hpipm_d_ocp_qp_dim.h"



#ifdef __cplusplus
extern "C" {
#endif



struct d_ocp_qp
	{
	struct d_ocp_qp_dim *dim;
	struct blasfeo_dmat *BAbt; // dynamics matrix & vector work space
	struct blasfeo_dmat *RSQrq; // hessian of cost & vector work space
	struct blasfeo_dmat *DCt; // inequality constraints matrix
	struct blasfeo_dvec *b; // dynamics vector
	struct blasfeo_dvec *rqz; // gradient of cost & gradient of slacks
	struct blasfeo_dvec *d; // inequality constraints vector
	struct blasfeo_dvec *d_mask; // inequality constraints mask vector
	struct blasfeo_dvec *m; // rhs of complementarity condition
	struct blasfeo_dvec *Z; // (diagonal) hessian of slacks
	int **idxb; // indices of box constrained variables within [u; x]
	int **idxs_rev; // index of soft constraints (reverse storage)
	int **idxe; // indices of constraints within [bu, bx, g] that are equalities, subset of [0, ..., nbu+nbx+ng-1]
	int *diag_H_flag; // flag the fact that Hessian is diagonal
	hpipm_size_t memsize; // memory size in bytes
	};



//
hpipm_size_t d_ocp_qp_strsize();
//
hpipm_size_t d_ocp_qp_memsize(struct d_ocp_qp_dim *dim);
//
void d_ocp_qp_create(struct d_ocp_qp_dim *dim, struct d_ocp_qp *qp, void *memory);
//
void d_ocp_qp_copy_all(struct d_ocp_qp *qp_orig, struct d_ocp_qp *qp_dest);

// setters
//
void d_ocp_qp_set_all_zero(struct d_ocp_qp *qp);
//
void d_ocp_qp_set_rhs_zero(struct d_ocp_qp *qp);
//
void d_ocp_qp_set_all(double **A, double **B, double **b, double **Q, double **S, double **R, double **q, double **r, int **idxbx, double **lbx, double **ubx, int **idxbu, double **lbu, double **ubu, double **C, double **D, double **lg, double **ug, double **Zl, double **Zu, double **zl, double **zu, int **idxs, double **ls, double **us, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_all_rowmaj(double **A, double **B, double **b, double **Q, double **S, double **R, double **q, double **r, int **idxbx, double **lbx, double **ubx, int **idxbu, double **lbu, double **ubu, double **C, double **D, double **lg, double **ug, double **Zl, double **Zu, double **zl, double **zu, int **idxs, double **ls, double **us, struct d_ocp_qp *qp);
//
void d_ocp_qp_set(char *field_name, int stage, void *value, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_el(char *field_name, int stage, int index, void *value, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_A(int stage, double *mat, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_B(int stage, double *mat, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_b(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Q(int stage, double *mat, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_S(int stage, double *mat, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_R(int stage, double *mat, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_q(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_r(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lb(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lb_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_ub(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_ub_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lbx(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lbx_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_el_lbx(int stage, int index, double *elem, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_ubx(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_ubx_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_el_ubx(int stage, int index, double *elem, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lbu(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lbu_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_ubu(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_ubu_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxb(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxbx(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Jbx(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxbu(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Jbu(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_C(int stage, double *mat, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_D(int stage, double *mat, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lg(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lg_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_ug(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_ug_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Zl(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Zu(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_zl(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_zu(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lls(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lls_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lus(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_lus_mask(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxs(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxs_rev(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Jsbu(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Jsbx(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Jsg(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxe(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxbxe(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxbue(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_idxge(int stage, int *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Jbxe(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Jbue(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_Jge(int stage, double *vec, struct d_ocp_qp *qp);
//
void d_ocp_qp_set_diag_H_flag(int stage, int *value, struct d_ocp_qp *qp);

// getters
//
void d_ocp_qp_get(char *field, int stage, struct d_ocp_qp *qp, void *value);
//
void d_ocp_qp_get_A(int stage, struct d_ocp_qp *qp, double *mat);
//
void d_ocp_qp_get_B(int stage, struct d_ocp_qp *qp, double *mat);
//
void d_ocp_qp_get_b(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_Q(int stage, struct d_ocp_qp *qp, double *mat);
//
void d_ocp_qp_get_S(int stage, struct d_ocp_qp *qp, double *mat);
//
void d_ocp_qp_get_R(int stage, struct d_ocp_qp *qp, double *mat);
//
void d_ocp_qp_get_q(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_r(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_ub(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_ub_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lb(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lb_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lbx(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lbx_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_ubx(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_ubx_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lbu(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lbu_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_ubu(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_ubu_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_idxb(int stage, struct d_ocp_qp *qp, int *vec);
//
//void d_ocp_qp_get_idxbx(int stage, struct d_ocp_qp *qp, int *vec);
//
//void d_ocp_qp_get_Jbx(int stage, struct d_ocp_qp *qp, double *vec);
//
//void d_ocp_qp_get_idxbu(int stage, struct d_ocp_qp *qp, int *vec);
//
//void d_ocp_qp_get_Jbu(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_C(int stage, struct d_ocp_qp *qp, double *mat);
//
void d_ocp_qp_get_D(int stage, struct d_ocp_qp *qp, double *mat);
//
void d_ocp_qp_get_lg(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lg_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_ug(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_ug_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_Zl(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_Zu(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_zl(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_zu(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lls(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lls_mask(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lus(int stage, struct d_ocp_qp *qp, double *vec);
//
void d_ocp_qp_get_lus_mask(int stage, struct d_ocp_qp *qp, double *vec);
// XXX only valid if there is one slack per softed constraint !!!
void d_ocp_qp_get_idxs(int stage, struct d_ocp_qp *qp, int *vec);
//
void d_ocp_qp_get_idxs_rev(int stage, struct d_ocp_qp *qp, int *vec);
//
//void d_ocp_qp_get_Jsbu(int stage, struct d_ocp_qp *qp, float *vec);
//
//void d_ocp_qp_get_Jsbx(int stage, struct d_ocp_qp *qp, float *vec);
//
//void d_ocp_qp_get_Jsg(int stage, struct d_ocp_qp *qp, float *vec);



#ifdef __cplusplus
}	// #extern "C"
#endif



#endif // HPIPM_D_OCP_QP_H_
