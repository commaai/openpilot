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

#ifndef HPIPM_S_CORE_QP_IPM_AUX_
#define HPIPM_S_CORE_QP_IPM_AUX_

#ifdef __cplusplus
extern "C" {
#endif

//
void s_compute_Gamma_gamma_qp(float *res_d, float *res_m, struct s_core_qp_ipm_workspace *rws);
//
void s_compute_gamma_qp(float *res_d, float *res_m, struct s_core_qp_ipm_workspace *rws);
//
void s_compute_lam_t_qp(float *res_d, float *res_m, float *dlam, float *dt, struct s_core_qp_ipm_workspace *rws);
//
void s_compute_alpha_qp(struct s_core_qp_ipm_workspace *rws);
//
void s_update_var_qp(struct s_core_qp_ipm_workspace *rws);
//
void s_compute_mu_aff_qp(struct s_core_qp_ipm_workspace *rws);
//
void s_backup_res_m(struct s_core_qp_ipm_workspace *rws);
//
void s_compute_centering_correction_qp(struct s_core_qp_ipm_workspace *rws);
//
void s_compute_centering_qp(struct s_core_qp_ipm_workspace *rws);
//
void s_compute_tau_min_qp(struct s_core_qp_ipm_workspace *rws);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // HPIPM_S_CORE_QP_IPM_AUX_
