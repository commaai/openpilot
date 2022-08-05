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
double ddot_(long long *m, double *x, long long *incx, double *y, long long *incy);
void dcopy_(long long *m, double *x, long long *incx, double *y, long long *incy);
void daxpy_(long long *m, double *alpha, double *x, long long *incx, double *y, long long *incy);
void dscal_(long long *m, double *alpha, double *x, long long *incx);

// level 2
void dgemv_(char *ta, long long *m, long long *n, double *alpha, double *A, long long *lda, double *x, long long *incx, double *beta, double *y, long long *incy);
void dsymv_(char *uplo, long long *m, double *alpha, double *A, long long *lda, double *x, long long *incx, double *beta, double *y, long long *incy);
void dtrmv_(char *uplo, char *trans, char *diag, long long *n, double *A, long long *lda, double *x, long long *incx);
void dtrsv_(char *uplo, char *trans, char *diag, long long *n, double *A, long long *lda, double *x, long long *incx);
void dger_(long long *m, long long *n, double *alpha, double *x, long long *incx, double *y, long long *incy, double *A, long long *lda);

// level 3
void dgemm_(char *ta, char *tb, long long *m, long long *n, long long *k, double *alpha, double *A, long long *lda, double *B, long long *ldb, double *beta, double *C, long long *ldc);
void dsyrk_(char *uplo, char *trans, long long *n, long long *k, double *alpha, double *A, long long *lda, double *beta, double *C, long long *ldc);
void dtrmm_(char *side, char *uplo, char *trans, char *diag, long long *m, long long *n, double *alpha, double *A, long long *lda, double *B, long long *ldb);
void dtrsm_(char *side, char *uplo, char *trans, char *diag, long long *m, long long *n, double *alpha, double *A, long long *lda, double *B, long long *ldb);

// lapack
void dpotrf_(char *uplo, long long *m, double *A, long long *lda, long long *info);
void dgetrf_(long long *m, long long *n, double *A, long long *lda, long long *ipiv, long long *info);
void dgeqrf_(long long *m, long long *n, double *A, long long *lda, double *tau, double *work, long long *lwork, long long *info);
void dgeqr2_(long long *m, long long *n, double *A, long long *lda, double *tau, double *work, long long *info);



#ifdef __cplusplus
}
#endif
