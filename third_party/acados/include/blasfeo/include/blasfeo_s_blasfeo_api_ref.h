/**************************************************************************************************
*                                                                                                 *
* This file is part of BLASFEO.                                                                   *
*                                                                                                 *
* BLASFEO -- BLAS For Embedded Optimization.                                                      *
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

#ifndef BLASFEO_S_BLASFEO_API_REF_H_
#define BLASFEO_S_BLASFEO_API_REF_H_

#include "blasfeo_common.h"


#ifdef __cplusplus
extern "C" {
#endif


// expose reference BLASFEO for testing

// --- level 1

void blasfeo_saxpy_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_saxpby_ref(int kmax, float alpha, struct blasfeo_svec_ref *sx, int xi, float beta, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_svecmul_ref(int m, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_svecmulacc_ref(int m, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);
float blasfeo_svecmuldot_ref(int m, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);
float blasfeo_sdot_ref(int m, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sy, int yi);
void blasfeo_srotg_ref(float a, float b, float *c, float *s);
void blasfeo_scolrot_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj0, int aj1, float c, float s);
void blasfeo_srowrot_ref(int m, struct blasfeo_smat_ref *sA, int ai0, int ai1, int aj, float c, float s);


// --- level 2

// dense
void blasfeo_sgemv_n_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, float beta, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_sgemv_t_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, float beta, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strsv_lnn_mn_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strsv_ltn_mn_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strsv_lnn_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strsv_lnu_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strsv_ltn_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strsv_ltu_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strsv_unn_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strsv_utn_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strmv_unn_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strmv_utn_ref(int m, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strmv_lnn_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strmv_ltn_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strmv_lnu_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_strmv_ltu_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, struct blasfeo_svec_ref *sz, int zi);
void blasfeo_sgemv_nt_ref(int m, int n, float alpha_n, float alpha_t, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx_n, int xi_n, struct blasfeo_svec_ref *sx_t, int xi_t, float beta_n, float beta_t, struct blasfeo_svec_ref *sy_n, int yi_n, struct blasfeo_svec_ref *sy_t, int yi_t, struct blasfeo_svec_ref *sz_n, int zi_n, struct blasfeo_svec_ref *sz_t, int zi_t);
void blasfeo_ssymv_l_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sx, int xi, float beta, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);

// diagonal
void blasfeo_sgemv_d_ref(int m, float alpha, struct blasfeo_svec_ref *sA, int ai, struct blasfeo_svec_ref *sx, int xi, float beta, struct blasfeo_svec_ref *sy, int yi, struct blasfeo_svec_ref *sz, int zi);


// --- level 3

// dense
void blasfeo_sgemm_nn_ref( int m, int n, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sgemm_nt_ref( int m, int n, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sgemm_tn_ref(int m, int n, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sgemm_tt_ref(int m, int n, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);

void blasfeo_ssyrk_ln_mn_ref( int m, int n, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_ssyrk_ln_ref( int m, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_ssyrk_lt_ref( int m, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_ssyrk_un_ref( int m, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_ssyrk_ut_ref( int m, int k, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);

void blasfeo_strmm_rutn_ref( int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_strmm_rlnn_ref( int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_strsm_rltn_ref( int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_strsm_rltu_ref( int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_strsm_rutn_ref( int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_strsm_llnu_ref( int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_strsm_lunn_ref( int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sD, int di, int dj);

// diagonal
void dgemm_diag_left_lib_ref(int m, int n, float alpha, float *dA, float *pB, int sdb, float beta, float *pC, int sdc, float *pD, int sdd);
void blasfeo_sgemm_dn_ref(int m, int n, float alpha, struct blasfeo_svec_ref *sA, int ai, struct blasfeo_smat_ref *sB, int bi, int bj, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sgemm_nd_ref(int m, int n, float alpha, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_svec_ref *sB, int bi, float beta, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);

// --- lapack

void blasfeo_sgetrf_nopivot_ref(int m, int n, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sgetrf_rowpivot_ref(int m, int n, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj, int *ipiv);
void blasfeo_spotrf_l_ref(int m, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_spotrf_l_mn_ref(int m, int n, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_ssyrk_dpotrf_ln_ref(int m, int k, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_ssyrk_dpotrf_ln_mn_ref(int m, int n, int k, struct blasfeo_smat_ref *sA, int ai, int aj, struct blasfeo_smat_ref *sB, int bi, int bj, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sgetrf_nopivot_ref(int m, int n, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj);
void blasfeo_sgetrf_rowpivot_ref(int m, int n, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj, int *ipiv);
void blasfeo_sgeqrf_ref(int m, int n, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj, void *work);
void blasfeo_sgelqf_ref(int m, int n, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj, void *work);
void blasfeo_sgelqf_pd_ref(int m, int n, struct blasfeo_smat_ref *sC, int ci, int cj, struct blasfeo_smat_ref *sD, int di, int dj, void *work);
void blasfeo_sgelqf_pd_la_ref(int m, int n1, struct blasfeo_smat_ref *sL, int li, int lj, struct blasfeo_smat_ref *sA, int ai, int aj, void *work);
void blasfeo_sgelqf_pd_lla_ref(int m, int n1, struct blasfeo_smat_ref *sL0, int l0i, int l0j, struct blasfeo_smat_ref *sL1, int l1i, int l1j, struct blasfeo_smat_ref *sA, int ai, int aj, void *work);

#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_S_BLASFEO_API_REF_H_
