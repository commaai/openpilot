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



#ifndef BLASFEO_D_BLAS_API_H_
#define BLASFEO_D_BLAS_API_H_



#include "blasfeo_target.h"



#ifdef __cplusplus
extern "C" {
#endif



#ifdef BLAS_API



#ifdef FORTRAN_BLAS_API



// BLAS 1
//
void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
//
void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
//
double ddot_(int *n, double *x, int *incx, double *y, int *incy);

// BLAS 3
//
void dgemm_(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
//
void dsyrk_(char *uplo, char *ta, int *m, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
//
void dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
//
void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);



// LAPACK
//
void dgesv_(int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
//
void dgetrf_np_(int *m, int *n, double *A, int *lda, int *info);
//
void dgetrs_(char *trans, int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void dlaswp_(int *n, double *A, int *lda, int *k1, int *k2, int *ipiv, int *incx);
//
void dposv_(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void dpotrf_(char *uplo, int *m, double *A, int *lda, int *info);
//
void dpotrs_(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void dtrtrs_(char *uplo, char *trans, char *diag, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);



#else // BLASFEO_API



// BLAS 1
//
void blas_daxpy(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
//
double blas_ddot(int *n, double *x, int *incx, double *y, int *incy);
//
void blas_dcopy(int *n, double *x, int *incx, double *y, int *incy);

// BLAS 3
//
void blas_dgemm(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
//
void blas_dsyrk(char *uplo, char *ta, int *m, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
//
void blas_dtrmm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
//
void blas_dtrsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);



// LAPACK
//
void blas_dgesv(int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void blas_dgetrf(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
//
void blas_dgetrf_np(int *m, int *n, double *A, int *lda, int *info);
//
void blas_dgetrs(char *trans, int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void blas_dlaswp(int *n, double *A, int *lda, int *k1, int *k2, int *ipiv, int *incx);
//
void blas_dposv(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void blas_dpotrf(char *uplo, int *m, double *A, int *lda, int *info);
//
void blas_dpotrs(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void blas_dtrtrs(char *uplo, char *trans, char *diag, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);



#endif // BLASFEO_API



#endif // BLAS_API



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_BLAS_API_H_
