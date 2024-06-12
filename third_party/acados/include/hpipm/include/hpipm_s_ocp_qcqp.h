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

#ifndef HPIPM_S_OCP_QCQP_H_
#define HPIPM_S_OCP_QCQP_H_



#include <blasfeo_target.h>
#include <blasfeo_common.h>

#include "hpipm_s_ocp_qcqp_dim.h"



#ifdef __cplusplus
extern "C" {
#endif



struct s_ocp_qcqp
	{
	struct s_ocp_qcqp_dim *dim;
	struct blasfeo_smat *BAbt; // dynamics matrix & vector work space
	struct blasfeo_smat *RSQrq; // hessian of cost & vector work space
	struct blasfeo_smat *DCt; // inequality constraints matrix
	struct blasfeo_smat **Hq; // hessians of quadratic constraints
	struct blasfeo_svec *b; // dynamics vector
	struct blasfeo_svec *rqz; // gradient of cost & gradient of slacks
	struct blasfeo_svec *d; // inequality constraints vector
	struct blasfeo_svec *d_mask; // inequality constraints mask vector
	struct blasfeo_svec *m; // rhs of complementarity condition
	struct blasfeo_svec *Z; // (diagonal) hessian of slacks
	int **idxb; // indices of box constrained variables within [u; x]
	int **idxs_rev; // index of soft constraints (reverse storage)
	int **Hq_nzero; // for each int, the last 3 bits ...abc, {a,b,c}=0 => {R,S,Q}=0
	hpipm_size_t memsize; // memory size in bytes
	};



//
hpipm_size_t s_ocp_qcqp_strsize();
//
hpipm_size_t s_ocp_qcqp_memsize(struct s_ocp_qcqp_dim *dim);
//
void s_ocp_qcqp_create(struct s_ocp_qcqp_dim *dim, struct s_ocp_qcqp *qp, void *memory);
//
void s_ocp_qcqp_copy_all(struct s_ocp_qcqp *qp_orig, struct s_ocp_qcqp *qp_dest);

// setters
//
void s_ocp_qcqp_set_all_zero(struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_rhs_zero(struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set(char *fiels_name, int stage, void *value, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_el(char *fiels_name, int stage, int index, void *value, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_A(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_B(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_b(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Q(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_S(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_R(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_q(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_r(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lb(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lb_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_ub(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_ub_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lbx(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lbx_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_el_lbx(int stage, int index, float *elem, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_ubx(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_ubx_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_el_ubx(int stage, int index, float *elem, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lbu(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lbu_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_ubu(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_ubu_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_idxb(int stage, int *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_idxbx(int stage, int *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Jbx(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_idxbu(int stage, int *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Jbu(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_C(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_D(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lg(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lg_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_ug(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_ug_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Qq(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Sq(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Rq(int stage, float *mat, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_qq(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_rq(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_uq(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_uq_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Zl(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Zu(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_zl(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_zu(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lls(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lls_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lus(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_lus_mask(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_idxs(int stage, int *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_idxs_rev(int stage, int *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Jsbu(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Jsbx(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Jsg(int stage, float *vec, struct s_ocp_qcqp *qp);
//
void s_ocp_qcqp_set_Jsq(int stage, float *vec, struct s_ocp_qcqp *qp);

// getters
//
void s_ocp_qcqp_get(char *field, int stage, struct s_ocp_qcqp *qp, void *value);
//
void s_ocp_qcqp_get_A(int stage, struct s_ocp_qcqp *qp, float *mat);
//
void s_ocp_qcqp_get_B(int stage, struct s_ocp_qcqp *qp, float *mat);
//
void s_ocp_qcqp_get_b(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_Q(int stage, struct s_ocp_qcqp *qp, float *mat);
//
void s_ocp_qcqp_get_S(int stage, struct s_ocp_qcqp *qp, float *mat);
//
void s_ocp_qcqp_get_R(int stage, struct s_ocp_qcqp *qp, float *mat);
//
void s_ocp_qcqp_get_q(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_r(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_ub(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_ub_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lb(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lb_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lbx(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lbx_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_ubx(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_ubx_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lbu(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lbu_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_ubu(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_ubu_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_idxb(int stage, struct s_ocp_qcqp *qp, int *vec);
//
//void s_ocp_qcqp_get_idxbx(int stage, struct s_ocp_qcqp *qp, int *vec);
//
//void s_ocp_qcqp_get_Jbx(int stage, struct s_ocp_qcqp *qp, float *vec);
//
//void s_ocp_qcqp_get_idxbu(int stage, struct s_ocp_qcqp *qp, int *vec);
//
//void s_ocp_qcqp_get_Jbu(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_C(int stage, struct s_ocp_qcqp *qp, float *mat);
//
void s_ocp_qcqp_get_D(int stage, struct s_ocp_qcqp *qp, float *mat);
//
void s_ocp_qcqp_get_lg(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lg_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_ug(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_ug_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_Zl(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_Zu(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_zl(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_zu(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lls(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lls_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lus(int stage, struct s_ocp_qcqp *qp, float *vec);
//
void s_ocp_qcqp_get_lus_mask(int stage, struct s_ocp_qcqp *qp, float *vec);
// XXX only valid if there is one slack per softed constraint !!!
void s_ocp_qcqp_get_idxs(int stage, struct s_ocp_qcqp *qp, int *vec);
//
void s_ocp_qcqp_get_idxs_rev(int stage, struct s_ocp_qcqp *qp, int *vec);
//
//void s_ocp_qcqp_get_Jsbu(int stage, struct s_ocp_qcqp *qp, float *vec);
//
//void s_ocp_qcqp_get_Jsbx(int stage, struct s_ocp_qcqp *qp, float *vec);
//
//void s_ocp_qcqp_get_Jsg(int stage, struct s_ocp_qcqp *qp, float *vec);



#ifdef __cplusplus
} /* extern "C" */
#endif



#endif // HPIPM_S_OCP_QCQP_H_

