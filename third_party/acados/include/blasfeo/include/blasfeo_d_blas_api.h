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



#ifdef BLAS_API
#ifdef CBLAS_API
#ifndef BLASFEO_CBLAS_ENUM
#define BLASFEO_CBLAS_ENUM
#ifdef FORTRAN_BLAS_API
#ifndef CBLAS_H
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};
#endif // CBLAS_H
#else // FORTRAN_BLAS_API
enum BLASFEO_CBLAS_ORDER {BlasfeoCblasRowMajor=101, BlasfeoCblasColMajor=102};
enum BLASFEO_CBLAS_TRANSPOSE {BlasfeoCblasNoTrans=111, BlasfeoCblasTrans=112, BlasfeoCblasConjTrans=113};
enum BLASFEO_CBLAS_UPLO {BlasfeoCblasUpper=121, BlasfeoCblasLower=122};
enum BLASFEO_CBLAS_DIAG {BlasfeoCblasNonUnit=131, BlasfeoCblasUnit=132};
enum BLASFEO_CBLAS_SIDE {BlasfeoCblasLeft=141, BlasfeoCblasRight=142};
#endif // FORTRAN_BLAS_API
#endif // BLASFEO_CBLAS_ENUM
#endif // CBLAS_API
#endif // BLAS_API


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

// BLAS 2
//
void dgemv_(char *tran, int *m, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
//
void dsymv_(char *uplo, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
//
void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *A, int *lda);

// BLAS 3
//
void dgemm_(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
//
void dsyrk_(char *uplo, char *ta, int *m, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
//
void dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
//
void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
//
void dsyr2k_(char *uplo, char *ta, int *m, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);



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



// aux
void dgetr_(int *m, int *n, double *A, int *lda, double *B, int *ldb);



#ifdef CBLAS_API



// CBLAS 1
//
void cblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY);
//
void cblas_dswap(const int N, double *X, const int incX, double *Y, const int incY);
//
void cblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY);

// CBLAS 3
//
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
//
void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double *A, const int lda, const double beta, double *C, const int ldc);
//
void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb);
//
void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb);



#endif // CBLAS_API



#else // BLASFEO_API



// BLAS 1
//
void blasfeo_blas_daxpy(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
//
double blasfeo_blas_ddot(int *n, double *x, int *incx, double *y, int *incy);
//
void blasfeo_blas_dcopy(int *n, double *x, int *incx, double *y, int *incy);

// BLAS 2
//
void blasfeo_blas_dgemv(char *trans, int *m, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
//
void blasfeo_blas_dsymv(char *uplo, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
//
void blasfeo_blas_dger(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *A, int *lda);

// BLAS 3
//
void blasfeo_blas_dgemm(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
//
void blasfeo_blas_dsyrk(char *uplo, char *ta, int *m, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
//
void blasfeo_blas_dtrmm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
//
void blasfeo_blas_dtrsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
//
void blasfeo_blas_dsyr2k(char *uplo, char *ta, int *m, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);



// LAPACK
//
void blasfeo_lapack_dgesv(int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void blasfeo_lapack_dgetrf(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
//
void blasfeo_lapack_dgetrf_np(int *m, int *n, double *A, int *lda, int *info);
//
void blasfeo_lapack_dgetrs(char *trans, int *m, int *n, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
//
void blasfeo_lapack_dlaswp(int *n, double *A, int *lda, int *k1, int *k2, int *ipiv, int *incx);
//
void blasfeo_lapack_dposv(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void blasfeo_lapack_dpotrf(char *uplo, int *m, double *A, int *lda, int *info);
//
void blasfeo_lapack_dpotrs(char *uplo, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);
//
void blasfeo_lapack_dtrtrs(char *uplo, char *trans, char *diag, int *m, int *n, double *A, int *lda, double *B, int *ldb, int *info);



// aux
void blasfeo_blas_dgetr(int *m, int *n, double *A, int *lda, double *B, int *ldb);



#ifdef CBLAS_API



// CBLAS 1
//
void blasfeo_cblas_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY);
//
void blasfeo_cblas_dswap(const int N, double *X, const int incX, double *Y, const int incY);
//
void blasfeo_cblas_dcopy(const int N, const double *X, const int incX, double *Y, const int incY);

// CBLAS 3
//
void blasfeo_cblas_dgemm(const enum BLASFEO_CBLAS_ORDER Order, const enum BLASFEO_CBLAS_TRANSPOSE TransA, const enum BLASFEO_CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
//
void blasfeo_cblas_dsyrk(const enum BLASFEO_CBLAS_ORDER Order, const enum BLASFEO_CBLAS_UPLO Uplo, const enum BLASFEO_CBLAS_TRANSPOSE Trans, const int N, const int K, const double alpha, const double *A, const int lda, const double beta, double *C, const int ldc);
//
void blasfeo_cblas_dtrmm(const enum BLASFEO_CBLAS_ORDER Order, const enum BLASFEO_CBLAS_SIDE Side, const enum BLASFEO_CBLAS_UPLO Uplo, const enum BLASFEO_CBLAS_TRANSPOSE TransA, const enum BLASFEO_CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb);
//
void blasfeo_cblas_dtrsm(const enum BLASFEO_CBLAS_ORDER Order, const enum BLASFEO_CBLAS_SIDE Side, const enum BLASFEO_CBLAS_UPLO Uplo, const enum BLASFEO_CBLAS_TRANSPOSE TransA, const enum BLASFEO_CBLAS_DIAG Diag, const int M, const int N, const double alpha, const double *A, const int lda, double *B, const int ldb);



#endif // CBLAS_API



#endif // BLASFEO_API



#endif // BLAS_API



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_BLAS_API_H_
