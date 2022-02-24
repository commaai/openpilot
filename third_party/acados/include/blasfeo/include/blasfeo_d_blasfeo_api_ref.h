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

#ifndef BLASFEO_D_BLASFEO_API_REF_H_
#define BLASFEO_D_BLASFEO_API_REF_H_

#include "blasfeo_common.h"


#ifdef __cplusplus
extern "C" {
#endif


// expose reference BLASFEO for testing

// --- level 1

void blasfeo_daxpy_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_daxpby_ref(int kmax, double alpha, struct blasfeo_dvec_ref *sx, int xi, double beta, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dvecmul_ref(int m, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dvecmulacc_ref(int m, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);
double blasfeo_dvecmuldot_ref(int m, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);
double blasfeo_ddot_ref(int m, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sy, int yi);
void blasfeo_drotg_ref(double a, double b, double *c, double *s);
void blasfeo_dcolrot_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj0, int aj1, double c, double s);
void blasfeo_drowrot_ref(int m, struct blasfeo_dmat_ref *sA, int ai0, int ai1, int aj, double c, double s);


// --- level 2

// dense
void blasfeo_dgemv_n_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, double beta, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dgemv_t_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, double beta, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrsv_lnn_mn_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrsv_ltn_mn_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrsv_lnn_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrsv_lnu_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrsv_ltn_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrsv_ltu_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrsv_unn_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrsv_utn_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrmv_unn_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrmv_utn_ref(int m, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrmv_lnn_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrmv_ltn_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrmv_lnu_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dtrmv_ltu_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, struct blasfeo_dvec_ref *sz, int zi);
void blasfeo_dgemv_nt_ref(int m, int n, double alpha_n, double alpha_t, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx_n, int xi_n, struct blasfeo_dvec_ref *sx_t, int xi_t, double beta_n, double beta_t, struct blasfeo_dvec_ref *sy_n, int yi_n, struct blasfeo_dvec_ref *sy_t, int yi_t, struct blasfeo_dvec_ref *sz_n, int zi_n, struct blasfeo_dvec_ref *sz_t, int zi_t);
void blasfeo_dsymv_l_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sx, int xi, double beta, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);

// diagonal
void blasfeo_dgemv_d_ref(int m, double alpha, struct blasfeo_dvec_ref *sA, int ai, struct blasfeo_dvec_ref *sx, int xi, double beta, struct blasfeo_dvec_ref *sy, int yi, struct blasfeo_dvec_ref *sz, int zi);


// --- level 3

// dense
void blasfeo_dgemm_nn_ref( int m, int n, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dgemm_nt_ref( int m, int n, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dgemm_tn_ref(int m, int n, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dgemm_tt_ref(int m, int n, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);

void blasfeo_dsyrk_ln_mn_ref( int m, int n, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dsyrk_ln_ref( int m, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dsyrk_lt_ref( int m, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dsyrk_un_ref( int m, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dsyrk_ut_ref( int m, int k, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);

void blasfeo_dtrmm_rutn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrmm_rlnn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);

void blasfeo_dtrsm_lunu_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_lunn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_lutu_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_lutn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_llnu_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_llnn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_lltu_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_lltn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_runu_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_runn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_rutu_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_rutn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_rlnu_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_rlnn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_rltu_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dtrsm_rltn_ref( int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sD, int di, int dj);

// diagonal
void dgemm_diag_left_lib_ref(int m, int n, double alpha, double *dA, double *pB, int sdb, double beta, double *pC, int sdc, double *pD, int sdd);
void blasfeo_dgemm_dn_ref(int m, int n, double alpha, struct blasfeo_dvec_ref *sA, int ai, struct blasfeo_dmat_ref *sB, int bi, int bj, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dgemm_nd_ref(int m, int n, double alpha, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dvec_ref *sB, int bi, double beta, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);

// --- lapack

void blasfeo_dgetrf_nopivot_ref(int m, int n, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dgetrf_rowpivot_ref(int m, int n, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj, int *ipiv);
void blasfeo_dpotrf_l_ref(int m, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dpotrf_l_mn_ref(int m, int n, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dsyrk_dpotrf_ln_ref(int m, int k, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dsyrk_dpotrf_ln_mn_ref(int m, int n, int k, struct blasfeo_dmat_ref *sA, int ai, int aj, struct blasfeo_dmat_ref *sB, int bi, int bj, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dgetrf_nopivot_ref(int m, int n, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj);
void blasfeo_dgetrf_rowpivot_ref(int m, int n, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj, int *ipiv);
void blasfeo_dgeqrf_ref(int m, int n, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj, void *work);
void blasfeo_dgelqf_ref(int m, int n, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj, void *work);
void blasfeo_dgelqf_pd_ref(int m, int n, struct blasfeo_dmat_ref *sC, int ci, int cj, struct blasfeo_dmat_ref *sD, int di, int dj, void *work);
void blasfeo_dgelqf_pd_la_ref(int m, int n1, struct blasfeo_dmat_ref *sL, int li, int lj, struct blasfeo_dmat_ref *sA, int ai, int aj, void *work);
void blasfeo_dgelqf_pd_lla_ref(int m, int n1, struct blasfeo_dmat_ref *sL0, int l0i, int l0j, struct blasfeo_dmat_ref *sL1, int l1i, int l1j, struct blasfeo_dmat_ref *sA, int ai, int aj, void *work);

#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_BLASFEO_API_REF_H_
