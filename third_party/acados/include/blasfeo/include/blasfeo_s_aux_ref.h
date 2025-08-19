
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

#ifndef BLASFEO_S_AUX_REF_H_
#define BLASFEO_S_AUX_REF_H_



#include <stdlib.h>

#include "blasfeo_s_aux_old.h"
#include "blasfeo_common.h"



#ifdef __cplusplus
extern "C" {
#endif


/************************************************
* d_aux_lib.c
************************************************/

// returns the memory size (in bytes) needed for a smat
size_t blasfeo_ref_memsize_smat(int m, int n);
size_t blasfeo_ref_memsize_smat_ps(int ps, int m, int n);
// returns the memory size (in bytes) needed for the diagonal of a smat
size_t blasfeo_ref_memsize_diag_smat(int m, int n);
// returns the memory size (in bytes) needed for a svec
size_t blasfeo_ref_memsize_svec(int m);
// create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
void blasfeo_ref_create_smat(int m, int n, struct blasfeo_smat *sA, void *memory);
void blasfeo_ref_create_smat_ps(int ps, int m, int n, struct blasfeo_smat *sA, void *memory);
// create a strvec for a vector of size m by using memory passed by a pointer (pointer is not updated)
void blasfeo_ref_create_svec(int m, struct blasfeo_svec *sA, void *memory);
void blasfeo_ref_pack_smat(int m, int n, float *A, int lda, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_pack_l_smat(int m, int n, float *A, int lda, struct blasfeo_smat *sB, int bi, int bj);
void blasfeo_ref_pack_l_smat(int m, int n, float *A, int lda, struct blasfeo_smat *sB, int bi, int bj);
void blasfeo_ref_pack_tran_smat(int m, int n, float *A, int lda, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_pack_svec(int m, float *x, int xi, struct blasfeo_svec *sa, int ai);
void blasfeo_ref_unpack_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj, float *A, int lda);
void blasfeo_ref_unpack_tran_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj, float *A, int lda);
void blasfeo_ref_unpack_svec(int m, struct blasfeo_svec *sa, int ai, float *x, int xi);
void ref_s_cast_mat2strmat(float *A, struct blasfeo_smat *sA);
void ref_s_cast_diag_mat2strmat(float *dA, struct blasfeo_smat *sA);
void ref_s_cast_vec2vecmat(float *a, struct blasfeo_svec *sa);

// ge
void blasfeo_ref_sgese(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_sgecpsc(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_ref_sgecp(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_ref_sgesc(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_sgein1(float a, struct blasfeo_smat *sA, int ai, int aj);
float blasfeo_ref_sgeex1(struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_sgead(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_ref_sgetr(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
// tr
void blasfeo_ref_strcp_l(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_ref_strtr_l(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_ref_strtr_u(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
// dia
void blasfeo_ref_sdiare(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_sdiaex(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi);
void blasfeo_ref_sdiain(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_sdiain_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_ref_sdiaex_sp(int kmax, float alpha, int *idx, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_svec *sx, int xi);
void blasfeo_ref_sdiaad(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_sdiaad_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_ref_sdiaadin_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sy, int yi, int *idx, struct blasfeo_smat *sD, int di, int dj);
// row
void blasfeo_ref_srowin(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_srowex(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi);
void blasfeo_ref_srowad(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_srowad_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
void blasfeo_ref_srowsw(int kmax, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_ref_srowpe(int kmax, int *ipiv, struct blasfeo_smat *sA);
void blasfeo_ref_srowpei(int kmax, int *ipiv, struct blasfeo_smat *sA);
// col
void blasfeo_ref_scolex(int kmax, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi);
void blasfeo_ref_scolin(int kmax, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_scolad(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_scolsc(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj);
void blasfeo_ref_scolsw(int kmax, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void blasfeo_ref_scolpe(int kmax, int *ipiv, struct blasfeo_smat *sA);
void blasfeo_ref_scolpei(int kmax, int *ipiv, struct blasfeo_smat *sA);
// vec
void blasfeo_ref_svecse(int m, float alpha, struct blasfeo_svec *sx, int xi);
void blasfeo_ref_sveccp(int m, struct blasfeo_svec *sa, int ai, struct blasfeo_svec *sc, int ci);
void blasfeo_ref_svecsc(int m, float alpha, struct blasfeo_svec *sa, int ai);
void blasfeo_ref_sveccpsc(int m, float alpha, struct blasfeo_svec *sa, int ai, struct blasfeo_svec *sc, int ci);
void blasfeo_ref_svecad(int m, float alpha, struct blasfeo_svec *sa, int ai, struct blasfeo_svec *sc, int ci);
void blasfeo_ref_svecin1(float a, struct blasfeo_svec *sx, int xi);
float blasfeo_ref_svecex1(struct blasfeo_svec *sx, int xi);
void blasfeo_ref_svecad_sp(int m, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_svec *sz, int zi);
void blasfeo_ref_svecin_sp(int m, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_svec *sz, int zi);
void blasfeo_ref_svecex_sp(int m, float alpha, int *idx, struct blasfeo_svec *sx, int x, struct blasfeo_svec *sz, int zi);
// z += alpha * x[idx]
void blasfeo_ref_svecexad_sp(int m, double alpha, int *idx, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sz, int zi);
void blasfeo_ref_sveccl(int m, struct blasfeo_svec *sxm, int xim, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sxp, int xip, struct blasfeo_svec *sz, int zi);
void blasfeo_ref_sveccl_mask(int m, struct blasfeo_svec *sxm, int xim, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sxp, int xip, struct blasfeo_svec *sz, int zi, struct blasfeo_svec *sm, int mi);
void blasfeo_ref_svecze(int m, struct blasfeo_svec *sm, int mi, struct blasfeo_svec *sv, int vi, struct blasfeo_svec *se, int ei);
void blasfeo_ref_svecnrm_inf(int m, struct blasfeo_svec *sx, int xi, float *ptr_norm);
void blasfeo_ref_svecpe(int kmax, int *ipiv, struct blasfeo_svec *sx, int xi);
void blasfeo_ref_svecpei(int kmax, int *ipiv, struct blasfeo_svec *sx, int xi);



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_S_AUX_REF_H_

