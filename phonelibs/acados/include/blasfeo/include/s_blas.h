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
float sdot_(int *m, float *x, int *incx, float *y, int *incy);
void scopy_(int *m, float *x, int *incx, float *y, int *incy);
void saxpy_(int *m, float *alpha, float *x, int *incx, float *y, int *incy);
void sscal_(int *m, float *alpha, float *x, int *incx);
void srot_(int *m, float *x, int *incx, float *y, int *incy, float *c, float *s);
void srotg_(float *a, float *b, float *c, float *s);

// level 2
void sgemv_(char *ta, int *m, int *n, float *alpha, float *A, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
void ssymv_(char *uplo, int *m, float *alpha, float *A, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
void strmv_(char *uplo, char *trans, char *diag, int *n, float *A, int *lda, float *x, int *incx);
void strsv_(char *uplo, char *trans, char *diag, int *n, float *A, int *lda, float *x, int *incx);
void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *A, int *lda);

// level 3
void sgemm_(char *ta, char *tb, int *m, int *n, int *k, float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, float *C, int *ldc);
void ssyrk_(char *uplo, char *trans, int *n, int *k, float *alpha, float *A, int *lda, float *beta, float *C, int *ldc);
void strmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *A, int *lda, float *B, int *ldb);
void strsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *A, int *lda, float *B, int *ldb);

// lapack
void spotrf_(char *uplo, int *m, float *A, int *lda, int *info);
void sgetrf_(int *m, int *n, float *A, int *lda, int *ipiv, int *info);
void sgeqrf_(int *m, int *n, float *A, int *lda, float *tau, float *work, int *lwork, int *info);
void sgeqr2_(int *m, int *n, float *A, int *lda, float *tau, float *work, int *info);
void sgelqf_(int *m, int *n, float *A, int *lda, float *tau, float *work, int *lwork, int *info);
void sorglq_(int *m, int *n, int *k, float *A, int *lda, float *tau, float *work, int *lwork, int *info);



#ifdef __cplusplus
}
#endif
