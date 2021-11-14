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

#ifdef __cplusplus
extern "C" {
#endif



// headers to reference BLAS and LAPACK routines employed in BLASFEO WR

// level 1
double ddot_(int *m, double *x, int *incx, double *y, int *incy);
void dcopy_(int *m, double *x, int *incx, double *y, int *incy);
void daxpy_(int *m, double *alpha, double *x, int *incx, double *y, int *incy);
void dscal_(int *m, double *alpha, double *x, int *incx);
void drot_(int *m, double *x, int *incx, double *y, int *incy, double *c, double *s);
void drotg_(double *a, double *b, double *c, double *s);

// level 2
void dgemv_(char *ta, int *m, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dsymv_(char *uplo, int *m, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
void dtrmv_(char *uplo, char *trans, char *diag, int *n, double *A, int *lda, double *x, int *incx);
void dtrsv_(char *uplo, char *trans, char *diag, int *n, double *A, int *lda, double *x, int *incx);
void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *A, int *lda);

// level 3
void dgemm_(char *ta, char *tb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, double *A, int *lda, double *beta, double *C, int *ldc);
void dtrmm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);
void dtrsm_(char *side, char *uplo, char *trans, char *diag, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb);

// lapack
void dpotrf_(char *uplo, int *m, double *A, int *lda, int *info);
void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
void dgeqrf_(int *m, int *n, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
void dgeqr2_(int *m, int *n, double *A, int *lda, double *tau, double *work, int *info);
void dgelqf_(int *m, int *n, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
void dorglq_(int *m, int *n, int *k, double *A, int *lda, double *tau, double *work, int *lwork, int *info);



#ifdef __cplusplus
}
#endif
