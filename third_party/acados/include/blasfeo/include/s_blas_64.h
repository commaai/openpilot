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
float sdot_(long long *m, float *x, long long *incx, float *y, long long *incy);
void scopy_(long long *m, float *x, long long *incx, float *y, long long *incy);
void saxpy_(long long *m, float *alpha, float *x, long long *incx, float *y, long long *incy);
void sscal_(long long *m, float *alpha, float *x, long long *incx);

// level 2
void sgemv_(char *ta, long long *m, long long *n, float *alpha, float *A, long long *lda, float *x, long long *incx, float *beta, float *y, long long *incy);
void ssymv_(char *uplo, long long *m, float *alpha, float *A, long long *lda, float *x, long long *incx, float *beta, float *y, long long *incy);
void strmv_(char *uplo, char *trans, char *diag, long long *n, float *A, long long *lda, float *x, long long *incx);
void strsv_(char *uplo, char *trans, char *diag, long long *n, float *A, long long *lda, float *x, long long *incx);
void sger_(long long *m, long long *n, float *alpha, float *x, long long *incx, float *y, long long *incy, float *A, long long *lda);

// level 3
void sgemm_(char *ta, char *tb, long long *m, long long *n, long long *k, float *alpha, float *A, long long *lda, float *B, long long *ldb, float *beta, float *C, long long *ldc);
void ssyrk_(char *uplo, char *trans, long long *n, long long *k, float *alpha, float *A, long long *lda, float *beta, float *C, long long *ldc);
void strmm_(char *side, char *uplo, char *transa, char *diag, long long *m, long long *n, float *alpha, float *A, long long *lda, float *B, long long *ldb);
void strsm_(char *side, char *uplo, char *transa, char *diag, long long *m, long long *n, float *alpha, float *A, long long *lda, float *B, long long *ldb);

// lapack
void spotrf_(char *uplo, long long *m, float *A, long long *lda, long long *info);
void sgetrf_(long long *m, long long *n, float *A, long long *lda, long long *ipiv, long long *info);
void sgeqrf_(long long *m, long long *n, float *A, long long *lda, float *tau, float *work, long long *lwork, long long *info);
void sgeqr2_(long long *m, long long *n, float *A, long long *lda, float *tau, float *work, long long *info);



#ifdef __cplusplus
}
#endif
