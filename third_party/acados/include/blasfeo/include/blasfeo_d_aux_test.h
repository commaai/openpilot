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

/*
 * auxiliary algebra operations header
 *
 * include/blasfeo_aux_lib*.h
 *
 */

#ifndef BLASFEO_D_AUX_TEST_H_
#define BLASFEO_D_AUX_TEST_H_

#include "blasfeo_common.h"


#ifdef __cplusplus
extern "C" {
#endif

// --- memory calculations
int test_blasfeo_memsize_dmat(int m, int n);
int test_blasfeo_memsize_diag_dmat(int m, int n);
int test_blasfeo_memsize_dvec(int m);

// --- creation
void test_blasfeo_create_dmat(int m, int n, struct blasfeo_dmat *sA, void *memory);
void test_blasfeo_create_dvec(int m, struct blasfeo_dvec *sA, void *memory);

// --- conversion
void test_blasfeo_pack_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sA, int ai, int aj);
void test_blasfeo_pack_dvec(int m, double *x, int xi, struct blasfeo_dvec *sa, int ai);
void test_blasfeo_pack_tran_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sA, int ai, int aj);
void test_blasfeo_unpack_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, double *A, int lda);
void test_blasfeo_unpack_dvec(int m, struct blasfeo_dvec *sa, int ai, double *x, int xi);
void test_blasfeo_unpack_tran_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, double *A, int lda);

// --- cast
void test_d_cast_mat2strmat(double *A, struct blasfeo_dmat *sA);
void test_d_cast_diag_mat2strmat(double *dA, struct blasfeo_dmat *sA);
void test_d_cast_vec2vecmat(double *a, struct blasfeo_dvec *sa);

// ------ copy / scale

// B <= A
void test_blasfeo_dgecp(int m, int n,
					struct blasfeo_dmat *sA, int ai, int aj,
					struct blasfeo_dmat *sB, int bi, int bj);

// A <= alpha*A
void test_blasfeo_dgesc(int m, int n,
					double alpha,
					struct blasfeo_dmat *sA, int ai, int aj);

// B <= alpha*A
void test_blasfeo_dgecpsc(int m, int n,
					double alpha,
					struct blasfeo_dmat *sA, int ai, int aj,
					struct blasfeo_dmat *sB, int bi, int bj);

// // --- insert/extract
// //
// // <= sA[ai, aj]
// void test_blasfeo_dgein1(double a, struct blasfeo_dmat *sA, int ai, int aj);
// // <= sA[ai, aj]
// double blasfeo_dgeex1(struct blasfeo_dmat *sA, int ai, int aj);
// // sx[xi] <= a
// void test_blasfeo_dvecin1(double a, struct blasfeo_dvec *sx, int xi);
// // <= sx[xi]
// double blasfeo_dvecex1(struct blasfeo_dvec *sx, int xi);
// // A <= alpha

// // --- set
// void test_blasfeo_dgese(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// // a <= alpha
// void test_blasfeo_dvecse(int m, double alpha, struct blasfeo_dvec *sx, int xi);
// // B <= A


// // --- vector
// // y <= x
// void test_blasfeo_dveccp(int m, struct blasfeo_dvec *sa, int ai, struct blasfeo_dvec *sc, int ci);
// // x <= alpha*x
// void test_blasfeo_dvecsc(int m, double alpha, struct blasfeo_dvec *sa, int ai);
// // TODO
// // x <= alpha*x
// void test_blasfeo_dveccpsc(int m, double alpha, struct blasfeo_dvec *sa, int ai, struct blasfeo_dvec *sc, int ci);


// // B <= A, A lower triangular
// void test_blasfeo_dtrcp_l(int m,
//                     struct blasfeo_dmat *sA, int ai, int aj,
//                     struct blasfeo_dmat *sB, int bi, int bj);

// void test_blasfeo_dtrcpsc_l(int m, double alpha,
//                     struct blasfeo_dmat *sA, int ai, int aj,
//                     struct blasfeo_dmat *sB, int bi, int bj);

// void test_blasfeo_dtrsc_l(int m, double alpha,
//                     struct blasfeo_dmat *sA, int ai, int aj);


// // B <= B + alpha*A
// void test_blasfeo_dgead(int m, int n, double alpha,
//                     struct blasfeo_dmat *sA, int ai, int aj,
//                     struct blasfeo_dmat *sC, int ci, int cj);

// // y <= y + alpha*x
// void test_blasfeo_dvecad(int m, double alpha,
//                     struct blasfeo_dvec *sa, int ai,
//                     struct blasfeo_dvec *sc, int ci);

// // --- traspositions
// void test_dgetr_lib(int m, int n, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
// void test_blasfeo_dgetr(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
// void test_dtrtr_l_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
// void test_blasfeo_dtrtr_l(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
// void test_dtrtr_u_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
// void test_blasfeo_dtrtr_u(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
// void test_blasfeo_ddiare(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// void test_blasfeo_ddiain(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// void test_ddiaex_lib(int kmax, double alpha, int offset, double *pD, int sdd, double *x);
// void test_ddiaad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
// void test_ddiain_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
// void test_blasfeo_ddiain_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
// void test_ddiaex_libsp(int kmax, int *idx, double alpha, double *pD, int sdd, double *x);
// void test_blasfeo_ddiaex(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
// void test_blasfeo_ddiaex_sp(int kmax, double alpha, int *idx, struct blasfeo_dmat *sD, int di, int dj, struct blasfeo_dvec *sx, int xi);
// void test_blasfeo_ddiaad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// void test_ddiaad_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
// void test_blasfeo_ddiaad_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
// void test_ddiaadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD, int sdd);
// void test_blasfeo_ddiaadin_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
// void test_drowin_lib(int kmax, double alpha, double *x, double *pD);
// void test_blasfeo_drowin(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// void test_drowex_lib(int kmax, double alpha, double *pD, double *x);
// void test_blasfeo_drowex(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
// void test_drowad_lib(int kmax, double alpha, double *x, double *pD);
// void test_blasfeo_drowad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// void test_drowin_libsp(int kmax, double alpha, int *idx, double *x, double *pD);
// void test_drowad_libsp(int kmax, int *idx, double alpha, double *x, double *pD);
// void test_blasfeo_drowad_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
// void test_drowadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD);
// void test_drowsw_lib(int kmax, double *pA, double *pC);
// void test_blasfeo_drowsw(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
// void test_blasfeo_drowpe(int kmax, int *ipiv, struct blasfeo_dmat *sA);
// void test_blasfeo_dcolex(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
// void test_dcolin_lib(int kmax, double *x, int offset, double *pD, int sdd);
// void test_blasfeo_dcolin(int kmax, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// void test_dcolad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
// void test_dcolin_libsp(int kmax, int *idx, double *x, double *pD, int sdd);
// void test_dcolad_libsp(int kmax, double alpha, int *idx, double *x, double *pD, int sdd);
// void test_dcolsw_lib(int kmax, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
// void test_blasfeo_dcolsw(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
// void test_blasfeo_dcolpe(int kmax, int *ipiv, struct blasfeo_dmat *sA);
// void test_dvecin_libsp(int kmax, int *idx, double *x, double *y);
// void test_dvecad_libsp(int kmax, int *idx, double alpha, double *x, double *y);
// void test_blasfeo_dvecad_sp(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi);
// void test_blasfeo_dvecin_sp(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi);
// void test_blasfeo_dvecex_sp(int m, double alpha, int *idx, struct blasfeo_dvec *sx, int x, struct blasfeo_dvec *sz, int zi);
// void test_blasfeo_dveccl(int m, struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi);
// void test_blasfeo_dveccl_mask(int m, struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi, struct blasfeo_dvec *sm, int mi);
// void test_blasfeo_dvecze(int m, struct blasfeo_dvec *sm, int mi, struct blasfeo_dvec *sv, int vi, struct blasfeo_dvec *se, int ei);
// void test_blasfeo_dvecnrm_inf(int m, struct blasfeo_dvec *sx, int xi, double *ptr_norm);
// void test_blasfeo_dvecpe(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi);
// void test_blasfeo_dvecpei(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi);

// ext_dep

void test_blasfeo_allocate_dmat(int m, int n, struct blasfeo_dmat *sA);
void test_blasfeo_allocate_dvec(int m, struct blasfeo_dvec *sa);

void test_blasfeo_free_dmat(struct blasfeo_dmat *sA);
void test_blasfeo_free_dvec(struct blasfeo_dvec *sa);

void test_blasfeo_print_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj);
void test_blasfeo_print_dvec(int m, struct blasfeo_dvec *sa, int ai);
void test_blasfeo_print_tran_dvec(int m, struct blasfeo_dvec *sa, int ai);

void test_blasfeo_print_to_file_dmat(FILE *file, int m, int n, struct blasfeo_dmat *sA, int ai, int aj);
void test_blasfeo_print_to_file_dvec(FILE *file, int m, struct blasfeo_dvec *sa, int ai);
void test_blasfeo_print_to_file_tran_dvec(FILE *file, int m, struct blasfeo_dvec *sa, int ai);

void test_blasfeo_print_exp_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj);
void test_blasfeo_print_exp_dvec(int m, struct blasfeo_dvec *sa, int ai);
void test_blasfeo_print_exp_tran_dvec(int m, struct blasfeo_dvec *sa, int ai);



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_AUX_TEST_H_
