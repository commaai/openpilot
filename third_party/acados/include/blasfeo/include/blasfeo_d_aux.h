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

#ifndef BLASFEO_D_AUX_H_
#define BLASFEO_D_AUX_H_



#include <stdlib.h>

#include "blasfeo_common.h"
#include "blasfeo_d_aux_old.h"



#ifdef __cplusplus
extern "C" {
#endif


// --- memory size calculations
//
// returns the memory size (in bytes) needed for a dmat
size_t blasfeo_memsize_dmat(int m, int n);
// returns the memory size (in bytes) needed for the diagonal of a dmat
size_t blasfeo_memsize_diag_dmat(int m, int n);
// returns the memory size (in bytes) needed for a dvec
size_t blasfeo_memsize_dvec(int m);

// --- creation
//
// create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
void blasfeo_create_dmat(int m, int n, struct blasfeo_dmat *sA, void *memory);
// create a strvec for a vector of size m by using memory passed by a pointer (pointer is not updated)
void blasfeo_create_dvec(int m, struct blasfeo_dvec *sA, void *memory);

// --- packing
// pack the column-major matrix A into the matrix struct B
void blasfeo_pack_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sB, int bi, int bj);
// pack the lower-triangular column-major matrix A into the matrix struct B
void blasfeo_pack_l_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sB, int bi, int bj);
// pack the upper-triangular column-major matrix A into the matrix struct B
void blasfeo_pack_u_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sB, int bi, int bj);
// transpose and pack the column-major matrix A into the matrix struct B
void blasfeo_pack_tran_dmat(int m, int n, double *A, int lda, struct blasfeo_dmat *sB, int bi, int bj);
// pack the vector x into the vector structure y
void blasfeo_pack_dvec(int m, double *x, int xi, struct blasfeo_dvec *sy, int yi);
// unpack the matrix structure A into the column-major matrix B
void blasfeo_unpack_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, double *B, int ldb);
// transpose and unpack the matrix structure A into the column-major matrix B
void blasfeo_unpack_tran_dmat(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, double *B, int ldb);
// pack the vector structure x into the vector y
void blasfeo_unpack_dvec(int m, struct blasfeo_dvec *sx, int xi, double *y,  int yi);

// --- cast
//
//void d_cast_mat2strmat(double *A, struct blasfeo_dmat *sA); // TODO
//void d_cast_diag_mat2strmat(double *dA, struct blasfeo_dmat *sA); // TODO
//void d_cast_vec2vecmat(double *a, struct blasfeo_dvec *sx); // TODO


// ge
// --- insert/extract
//
// sA[ai, aj] <= a
void blasfeo_dgein1(double a, struct blasfeo_dmat *sA, int ai, int aj);
// <= sA[ai, aj]
double blasfeo_dgeex1(struct blasfeo_dmat *sA, int ai, int aj);

// --- set
// A <= alpha
void blasfeo_dgese(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj);

// --- copy / scale
// B <= A
void blasfeo_dgecp(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// A <= alpha*A
void blasfeo_dgesc(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// B <= alpha*A
void blasfeo_dgecpsc(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// B <= A, A lower triangular
void blasfeo_dtrcp_l(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
void blasfeo_dtrcpsc_l(int m, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
void blasfeo_dtrsc_l(int m, double alpha, struct blasfeo_dmat *sA, int ai, int aj);

// --- sum
// B <= B + alpha*A
void blasfeo_dgead(int m, int n, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// y <= y + alpha*x
void blasfeo_dvecad(int m, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi);

// --- traspositions
// B <= A'
void blasfeo_dgetr(int m, int n, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// B <= A', A lower triangular
void blasfeo_dtrtr_l(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);
// B <= A', A upper triangular
void blasfeo_dtrtr_u(int m, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sB, int bi, int bj);

// dia
// diag(A) += alpha
void blasfeo_ddiare(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
// diag(A) <= alpha*x
void blasfeo_ddiain(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// diag(A)[idx] <= alpha*x
void blasfeo_ddiain_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
// x <= diag(A)
void blasfeo_ddiaex(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
// x <= diag(A)[idx]
void blasfeo_ddiaex_sp(int kmax, double alpha, int *idx, struct blasfeo_dmat *sD, int di, int dj, struct blasfeo_dvec *sx, int xi);
// diag(A) += alpha*x
void blasfeo_ddiaad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
// diag(A)[idx] += alpha*x
void blasfeo_ddiaad_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
// diag(A)[idx] = y + alpha*x
void blasfeo_ddiaadin_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi, int *idx, struct blasfeo_dmat *sD, int di, int dj);

// row
void blasfeo_drowin(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_drowex(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
void blasfeo_drowad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_drowad_sp(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dmat *sD, int di, int dj);
void blasfeo_drowsw(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void blasfeo_drowpe(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void blasfeo_drowpei(int kmax, int *ipiv, struct blasfeo_dmat *sA);

// col
void blasfeo_dcolex(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dvec *sx, int xi);
void blasfeo_dcolin(int kmax, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_dcolad(int kmax, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_dcolsc(int kmax, double alpha, struct blasfeo_dmat *sA, int ai, int aj);
void blasfeo_dcolsw(int kmax, struct blasfeo_dmat *sA, int ai, int aj, struct blasfeo_dmat *sC, int ci, int cj);
void blasfeo_dcolpe(int kmax, int *ipiv, struct blasfeo_dmat *sA);
void blasfeo_dcolpei(int kmax, int *ipiv, struct blasfeo_dmat *sA);

// vec
// a <= alpha
void blasfeo_dvecse(int m, double alpha, struct blasfeo_dvec *sx, int xi);
// sx[xi] <= a
void blasfeo_dvecin1(double a, struct blasfeo_dvec *sx, int xi);
// <= sx[xi]
double blasfeo_dvecex1(struct blasfeo_dvec *sx, int xi);
// y <= x
void blasfeo_dveccp(int m, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi);
// x <= alpha*x
void blasfeo_dvecsc(int m, double alpha, struct blasfeo_dvec *sx, int xi);
// y <= alpha*x
void blasfeo_dveccpsc(int m, double alpha, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sy, int yi);
// z[idx] += alpha * x
void blasfeo_dvecad_sp(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi);
// z[idx] <= alpha * x
void blasfeo_dvecin_sp(int m, double alpha, struct blasfeo_dvec *sx, int xi, int *idx, struct blasfeo_dvec *sz, int zi);
// z <= alpha * x[idx]
void blasfeo_dvecex_sp(int m, double alpha, int *idx, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);
// z += alpha * x[idx]
void blasfeo_dvecexad_sp(int m, double alpha, int *idx, struct blasfeo_dvec *sx, int xi, struct blasfeo_dvec *sz, int zi);

void blasfeo_dveccl(int m,
	struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi,
	struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi);

void blasfeo_dveccl_mask(int m,
	struct blasfeo_dvec *sxm, int xim, struct blasfeo_dvec *sx, int xi,
	struct blasfeo_dvec *sxp, int xip, struct blasfeo_dvec *sz, int zi,
	struct blasfeo_dvec *sm, int mi);

void blasfeo_dvecze(int m, struct blasfeo_dvec *sm, int mi, struct blasfeo_dvec *sv, int vi, struct blasfeo_dvec *se, int ei);
void blasfeo_dvecnrm_inf(int m, struct blasfeo_dvec *sx, int xi, double *ptr_norm);
void blasfeo_dvecnrm_2(int m, struct blasfeo_dvec *sx, int xi, double *ptr_norm);
void blasfeo_dvecpe(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi);
void blasfeo_dvecpei(int kmax, int *ipiv, struct blasfeo_dvec *sx, int xi);





/*
* Explicitly panel-major matrix format
*/

// returns the memory size (in bytes) needed for a dmat
size_t blasfeo_pm_memsize_dmat(int ps, int m, int n);
// create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
void blasfeo_pm_create_dmat(int ps, int m, int n, struct blasfeo_pm_dmat *sA, void *memory);
// print
void blasfeo_pm_print_dmat(int m, int n, struct blasfeo_pm_dmat *sA, int ai, int aj);



/*
* Explicitly panel-major matrix format
*/

// returns the memory size (in bytes) needed for a dmat
size_t blasfeo_cm_memsize_dmat(int m, int n);
// create a strmat for a matrix of size m*n by using memory passed by a pointer (pointer is not updated)
void blasfeo_cm_create_dmat(int m, int n, struct blasfeo_pm_dmat *sA, void *memory);



//
// BLAS API helper functions
//

#if ( defined(BLAS_API) & defined(MF_PANELMAJ) )
// aux
void blasfeo_cm_dgetr(int m, int n, struct blasfeo_cm_dmat *sA, int ai, int aj, struct blasfeo_cm_dmat *sB, int bi, int bj);
#endif



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_AUX_H_
