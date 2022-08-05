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



#ifndef BLASFEO_S_BLAS_API_H_
#define BLASFEO_S_BLAS_API_H_



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
enum BLASFEO_CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum BLASFEO_CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum BLASFEO_CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum BLASFEO_CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum BLASFEO_CBLAS_SIDE {CblasLeft=141, CblasRight=142};
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
void saxpy_(int *n, float *alpha, float *x, int *incx, float *y, int *incy);
//
float sdot_(int *n, float *x, int *incx, float *y, int *incy);

// BLAS 3
//
void sgemm_(char *ta, char *tb, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc);
//
void strsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *A, int *lda, float *B, int *ldb);



// LAPACK
//
void spotrf_(char *uplo, int *m, float *A, int *lda, int *info);



#ifdef CBLAS_API



// CBLAS 1
//
void cblas_saxpy(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);

// CBLAS 3
//
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
//
void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side, const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float *A, const int lda, float *B, const int ldb);



#endif // CBLAS_API



#else // BLASFEO_API



// BLAS 1
//
void blasfeo_blas_saxpy(int *n, float *alpha, float *x, int *incx, float *y, int *incy);
//
float blasfeo_blas_sdot(int *n, float *x, int *incx, float *y, int *incy);

// BLAS 3
//
void blasfeo_blas_sgemm(char *ta, char *tb, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc);
//
void blasfeo_blas_strsm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *A, int *lda, float *B, int *ldb);



// LAPACK
//
void blasfeo_lapack_spotrf(char *uplo, int *m, float *A, int *lda, int *info);



#ifdef CBLAS_API



// CBLAS 1
//
void blasfeo_cblas_saxpy(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);

// CBLAS 3
//
void blasfeo_cblas_sgemm(const enum BLASFEO_CBLAS_ORDER Order, const enum BLASFEO_CBLAS_TRANSPOSE TransA, const enum BLASFEO_CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
//
void blasfeo_cblas_strsm(const enum BLASFEO_CBLAS_ORDER Order, const enum BLASFEO_CBLAS_SIDE Side, const enum BLASFEO_CBLAS_UPLO Uplo, const enum BLASFEO_CBLAS_TRANSPOSE TransA, const enum BLASFEO_CBLAS_DIAG Diag, const int M, const int N, const float alpha, const float *A, const int lda, float *B, const int ldb);



#endif // CBLAS_API



#endif // BLASFEO_API



#endif // BLAS_API



#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_S_BLAS_API_H_
