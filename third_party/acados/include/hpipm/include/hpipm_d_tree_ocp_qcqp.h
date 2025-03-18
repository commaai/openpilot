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

#ifndef HPIPM_D_TREE_OCP_QCQP_H_
#define HPIPM_D_TREE_OCP_QCQP_H_



#include <blasfeo_target.h>
#include <blasfeo_common.h>

#include "hpipm_common.h"
#include "hpipm_d_tree_ocp_qcqp_dim.h"



#ifdef __cplusplus
extern "C" {
#endif



struct d_tree_ocp_qcqp
	{
	struct d_tree_ocp_qcqp_dim *dim;
	struct blasfeo_dmat *BAbt; // Nn-1
	struct blasfeo_dmat *RSQrq; // Nn
	struct blasfeo_dmat *DCt; // Nn
	struct blasfeo_dmat **Hq; // Nn
	struct blasfeo_dvec *b; // Nn-1
	struct blasfeo_dvec *rqz; // Nn
	struct blasfeo_dvec *d; // Nn
	struct blasfeo_dvec *d_mask; // Nn
	struct blasfeo_dvec *m; // Nn
	struct blasfeo_dvec *Z; // Nn
	int **idxb; // indices of box constrained variables within [u; x] // Nn
	int **idxs_rev; // index of soft constraints
	hpipm_size_t memsize; // memory size in bytes
	};



//
hpipm_size_t d_tree_ocp_qcqp_memsize(struct d_tree_ocp_qcqp_dim *dim);
//
void d_tree_ocp_qcqp_create(struct d_tree_ocp_qcqp_dim *dim, struct d_tree_ocp_qcqp *qp, void *memory);
//
void d_tree_ocp_qcqp_set_all(double **A, double **B, double **b, double **Q, double **S, double **R, double **q, double **r, int **idxb, double **d_lb, double **d_ub, double **C, double **D, double **d_lg, double **d_ug, double **Zl, double **Zu, double **zl, double **zu, int **idxs, double **d_ls, double **d_us, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set(char *field_name, int node_edge, void *value, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_A(int edge, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_B(int edge, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_b(int edge, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Q(int node, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_S(int node, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_R(int node, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_q(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_r(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lb(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lb_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_ub(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_ub_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lbx(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lbx_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_ubx(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_ubx_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lbu(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lbu_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_ubu(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_ubu_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_idxb(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_idxbx(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Jbx(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_idxbu(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Jbu(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_C(int node, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_D(int node, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lg(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lg_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_ug(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_ug_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Qq(int node, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Sq(int node, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Rq(int node, double *mat, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_qq(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_rq(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_uq(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_uq_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Zl(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Zu(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_zl(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_zu(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lls(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lls_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lus(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_lus_mask(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_idxs(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_idxs_rev(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Jsbu(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Jsbx(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Jsg(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
void d_tree_ocp_qcqp_set_Jsq(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
//void d_tree_ocp_qcqp_set_idxe(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
//void d_tree_ocp_qcqp_set_idxbxe(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
//void d_tree_ocp_qcqp_set_idxbue(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
//void d_tree_ocp_qcqp_set_idxge(int node, int *vec, struct d_tree_ocp_qcqp *qp);
//
//void d_tree_ocp_qcqp_set_Jbxe(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
//void d_tree_ocp_qcqp_set_Jbue(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
//void d_tree_ocp_qcqp_set_Jge(int node, double *vec, struct d_tree_ocp_qcqp *qp);
//
//void d_tree_ocp_qcqp_set_diag_H_flag(int node, int *value, struct d_tree_ocp_qcqp *qp);



#ifdef __cplusplus
} /* extern "C" */
#endif



#endif // HPIPM_D_TREE_OCP_QCQP_H_

