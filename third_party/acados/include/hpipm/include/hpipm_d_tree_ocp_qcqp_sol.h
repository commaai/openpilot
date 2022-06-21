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

#ifndef HPIPM_D_TREE_OCP_QCQP_SOL_H_
#define HPIPM_D_TREE_OCP_QCQP_SOL_H_



#include <blasfeo_target.h>
#include <blasfeo_common.h>

#include "hpipm_d_tree_ocp_qcqp_dim.h"



#ifdef __cplusplus
extern "C" {
#endif

struct d_tree_ocp_qcqp_sol
	{
	struct d_tree_ocp_qcqp_dim *dim;
	struct blasfeo_dvec *ux;
	struct blasfeo_dvec *pi;
	struct blasfeo_dvec *lam;
	struct blasfeo_dvec *t;
	void *misc;
	hpipm_size_t memsize; // memory size in bytes
	};



//
hpipm_size_t d_tree_ocp_qcqp_sol_memsize(struct d_tree_ocp_qcqp_dim *dim);
//
void d_tree_ocp_qcqp_sol_create(struct d_tree_ocp_qcqp_dim *dim, struct d_tree_ocp_qcqp_sol *qp_sol, void *memory);
//
void d_tree_ocp_qcqp_sol_get(char *field, int node_edge, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_u(int node, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_x(int node, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_sl(int node, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_su(int node, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_pi(int edge, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_lam_lb(int node, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_lam_ub(int node, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_lam_lg(int node, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);
//
void d_tree_ocp_qcqp_sol_get_lam_ug(int node, struct d_tree_ocp_qcqp_sol *qp_sol, double *vec);



#ifdef __cplusplus
} /* extern "C" */
#endif



#endif // HPIPM_D_TREE_OCP_QCQP_SOL_H_

