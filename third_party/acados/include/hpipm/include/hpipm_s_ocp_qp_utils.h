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

#ifndef HPIPM_S_OCP_QP_UTILS_H_
#define HPIPM_S_OCP_QP_UTILS_H_



#include <blasfeo_target.h>
#include <blasfeo_common.h>

#include "hpipm_s_ocp_qp_dim.h"
#include "hpipm_s_ocp_qp.h"
#include "hpipm_s_ocp_qp_sol.h"
#include "hpipm_s_ocp_qp_ipm.h"



#ifdef __cplusplus
extern "C" {
#endif



//
void s_ocp_qp_dim_print(struct s_ocp_qp_dim *qp_dim);
//
void s_ocp_qp_dim_codegen(char *file_name, char *mode, struct s_ocp_qp_dim *qp_dim);
//
void s_ocp_qp_print(struct s_ocp_qp_dim *qp_dim, struct s_ocp_qp *qp);
//
void s_ocp_qp_codegen(char *file_name, char *mode, struct s_ocp_qp_dim *qp_dim, struct s_ocp_qp *qp);
//
void s_ocp_qp_sol_print(struct s_ocp_qp_dim *qp_dim, struct s_ocp_qp_sol *ocp_qp_sol);
//
void s_ocp_qp_ipm_arg_print(struct s_ocp_qp_dim *qp_dim, struct s_ocp_qp_ipm_arg *arg);
//
void s_ocp_qp_ipm_arg_codegen(char *file_name, char *mode, struct s_ocp_qp_dim *qp_dim, struct s_ocp_qp_ipm_arg *arg);
//
void s_ocp_qp_res_print(struct s_ocp_qp_dim *qp_dim, struct s_ocp_qp_res *ocp_qp_res);



#ifdef __cplusplus
}	// #extern "C"
#endif



#endif // HPIPM_D_OCP_QP_UTILS_H_

