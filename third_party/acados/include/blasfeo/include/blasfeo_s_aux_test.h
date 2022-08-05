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

#ifndef BLASFEO_S_AUX_TEST_H_
#define BLASFEO_S_AUX_TEST_H_

#include <stdio.h>

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif



/************************************************
* d_aux_lib.c
************************************************/

int test_blasfeo_memsize_smat(int m, int n);
int test_blasfeo_memsize_diag_smat(int m, int n);
int test_blasfeo_memsize_svec(int m);

void test_blasfeo_create_smat(int m, int n, struct blasfeo_smat *sA, void *memory);
void test_blasfeo_create_svec(int m, struct blasfeo_svec *sA, void *memory);

void test_blasfeo_pack_smat(int m, int n, float *A, int lda, struct blasfeo_smat *sA, int ai, int aj);
void test_blasfeo_pack_svec(int m, float *x, int xi, struct blasfeo_svec *sa, int ai);
void test_blasfeo_pack_tran_smat(int m, int n, float *A, int lda, struct blasfeo_smat *sA, int ai, int aj);
void test_blasfeo_unpack_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj, float *A, int lda);
void test_blasfeo_unpack_svec(int m, struct blasfeo_svec *sa, int ai, float *x, int xi);
void test_blasfeo_unpack_tran_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj, float *A, int lda);

void test_s_cast_mat2strmat(float *A, struct blasfeo_smat *sA);
void test_s_cast_diag_mat2strmat(float *dA, struct blasfeo_smat *sA);
void test_s_cast_vec2vecmat(float *a, struct blasfeo_svec *sa);
// copy and scale
void test_blasfeo_sgecpsc(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void test_blasfeo_sgecp(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
void test_blasfeo_sgesc(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj);

// void test_blasfeo_sgein1(float a, struct blasfeo_smat *sA, int ai, int aj);
// float test_blasfeo_sgeex1(struct blasfeo_smat *sA, int ai, int aj);
// void test_blasfeo_svecin1(float a, struct blasfeo_svec *sx, int xi);
// float test_blasfeo_svecex1(struct blasfeo_svec *sx, int xi);

// // A <= alpha
// void test_blasfeo_sgese(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj);
// // a <= alpha
// void test_blasfeo_svecse(int m, float alpha, struct blasfeo_svec *sx, int xi);


// void test_blasfeo_sveccp(int m, struct blasfeo_svec *sa, int ai, struct blasfeo_svec *sc, int ci);
// void test_blasfeo_svecsc(int m, float alpha, struct blasfeo_svec *sa, int ai);

// void test_strcp_l_lib(int m, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
// void test_blasfeo_strcp_l(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);

// void test_sgead_lib(int m, int n, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
// void test_blasfeo_sgead(int m, int n, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
// void test_blasfeo_svecad(int m, float alpha, struct blasfeo_svec *sa, int ai, struct blasfeo_svec *sc, int ci);

// void test_sgetr_lib(int m, int n, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
// void test_blasfeo_sgetr(int m, int n, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);

// void test_strtr_l_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
// void test_blasfeo_strtr_l(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
// void test_strtr_u_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
// void test_blasfeo_strtr_u(int m, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);

// void test_sdiareg_lib(int kmax, float reg, int offset, float *pD, int sdd);
// void test_blasfeo_sdiaex(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi);
// void test_blasfeo_sdiain(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
// void test_sdiain_sqrt_lib(int kmax, float *x, int offset, float *pD, int sdd);
// void test_sdiaex_lib(int kmax, float alpha, int offset, float *pD, int sdd, float *x);
// void test_sdiaad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
// void test_sdiain_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
// void test_blasfeo_sdiain_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
// void test_sdiaex_libsp(int kmax, int *idx, float alpha, float *pD, int sdd, float *x);
// void test_blasfeo_sdiaex_sp(int kmax, float alpha, int *idx, struct blasfeo_smat *sD, int di, int dj, struct blasfeo_svec *sx, int xi);
// void test_blasfeo_sdiaad(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
// void test_sdiaad_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
// void test_blasfeo_sdiaad_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
// void test_sdiaadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD, int sdd);
// void test_blasfeo_sdiaadin_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sy, int yi, int *idx, struct blasfeo_smat *sD, int di, int dj);
// void test_srowin_lib(int kmax, float alpha, float *x, float *pD);
// void test_blasfeo_srowin(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
// void test_srowex_lib(int kmax, float alpha, float *pD, float *x);
// void test_blasfeo_srowex(int kmax, float alpha, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_svec *sx, int xi);
// void test_srowad_lib(int kmax, float alpha, float *x, float *pD);
// void test_blasfeo_srowad(int kmax, float alpha, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
// void test_srowin_libsp(int kmax, float alpha, int *idx, float *x, float *pD);
// void test_srowad_libsp(int kmax, int *idx, float alpha, float *x, float *pD);
// void test_blasfeo_srowad_sp(int kmax, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_smat *sD, int di, int dj);
// void test_srowadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD);
// void test_srowsw_lib(int kmax, float *pA, float *pC);
// void test_blasfeo_srowsw(int kmax, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
// void test_blasfeo_srowpe(int kmax, int *ipiv, struct blasfeo_smat *sA);
// void test_scolin_lib(int kmax, float *x, int offset, float *pD, int sdd);
// void test_blasfeo_scolin(int kmax, struct blasfeo_svec *sx, int xi, struct blasfeo_smat *sA, int ai, int aj);
// void test_scolad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
// void test_scolin_libsp(int kmax, int *idx, float *x, float *pD, int sdd);
// void test_scolad_libsp(int kmax, float alpha, int *idx, float *x, float *pD, int sdd);
// void test_scolsw_lib(int kmax, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
// void test_blasfeo_scolsw(int kmax, struct blasfeo_smat *sA, int ai, int aj, struct blasfeo_smat *sC, int ci, int cj);
// void test_blasfeo_scolpe(int kmax, int *ipiv, struct blasfeo_smat *sA);
// void test_svecin_libsp(int kmax, int *idx, float *x, float *y);
// void test_svecad_libsp(int kmax, int *idx, float alpha, float *x, float *y);
// void test_blasfeo_svecad_sp(int m, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_svec *sz, int zi);
// void test_blasfeo_svecin_sp(int m, float alpha, struct blasfeo_svec *sx, int xi, int *idx, struct blasfeo_svec *sz, int zi);
// void test_blasfeo_svecex_sp(int m, float alpha, int *idx, struct blasfeo_svec *sx, int x, struct blasfeo_svec *sz, int zi);
// void test_blasfeo_sveccl(int m, struct blasfeo_svec *sxm, int xim, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sxp, int xip, struct blasfeo_svec *sz, int zi);
// void test_blasfeo_sveccl_mask(int m, struct blasfeo_svec *sxm, int xim, struct blasfeo_svec *sx, int xi, struct blasfeo_svec *sxp, int xip, struct blasfeo_svec *sz, int zi, struct blasfeo_svec *sm, int mi);
// void test_blasfeo_svecze(int m, struct blasfeo_svec *sm, int mi, struct blasfeo_svec *sv, int vi, struct blasfeo_svec *se, int ei);
// void test_blasfeo_svecnrm_inf(int m, struct blasfeo_svec *sx, int xi, float *ptr_norm);
// void test_blasfeo_svecpe(int kmax, int *ipiv, struct blasfeo_svec *sx, int xi);


// ext_dep

void test_blasfeo_allocate_smat(int m, int n, struct blasfeo_smat *sA);
void test_blasfeo_allocate_svec(int m, struct blasfeo_svec *sa);

void test_blasfeo_free_smat(struct blasfeo_smat *sA);
void test_blasfeo_free_svec(struct blasfeo_svec *sa);

void test_blasfeo_print_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj);
void test_blasfeo_print_svec(int m, struct blasfeo_svec *sa, int ai);
void test_blasfeo_print_tran_svec(int m, struct blasfeo_svec *sa, int ai);

void test_blasfeo_print_to_file_smat(FILE *file, int m, int n, struct blasfeo_smat *sA, int ai, int aj);
void test_blasfeo_print_to_file_svec(FILE *file, int m, struct blasfeo_svec *sa, int ai);
void test_blasfeo_print_to_file_tran_svec(FILE *file, int m, struct blasfeo_svec *sa, int ai);

void test_blasfeo_print_exp_smat(int m, int n, struct blasfeo_smat *sA, int ai, int aj);
void test_blasfeo_print_exp_svec(int m, struct blasfeo_svec *sa, int ai);
void test_blasfeo_print_exp_tran_svec(int m, struct blasfeo_svec *sa, int ai);


#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_S_AUX_TEST_H_
