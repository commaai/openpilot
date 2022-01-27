/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_UTILS_MATH_H_
#define ACADOS_UTILS_MATH_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/utils/types.h"

#if defined(__MABX2__)
double fmax(double a, double b);
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void dgemm_nn_3l(int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc);
// void dgemv_n_3l(int m, int n, double *A, int lda, double *x, double *y);
// void dgemv_t_3l(int m, int n, double *A, int lda, double *x, double *y);
// void dcopy_3l(int n, double *x, int incx, double *y, int incy);
void daxpy_3l(int n, double da, double *dx, double *dy);
void dscal_3l(int n, double da, double *dx);
double twonormv(int n, double *ptrv);

/* copies a matrix into another matrix */
void dmcopy(int row, int col, double *ptrA, int lda, double *ptrB, int ldb);

/* solution of a system of linear equations */
void dgesv_3l(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb, int *info);

/* matrix exponential */
void expm(int row, double *A);

int idamax_3l(int n, double *x);

void dswap_3l(int n, double *x, int incx, double *y, int incy);

void dger_3l(int m, int n, double alpha, double *x, int incx, double *y, int incy, double *A,
             int lda);

void dgetf2_3l(int m, int n, double *A, int lda, int *ipiv, int *info);

void dlaswp_3l(int n, double *A, int lda, int k1, int k2, int *ipiv);

void dtrsm_l_l_n_u_3l(int m, int n, double *A, int lda, double *B, int ldb);

void dgetrs_3l(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb);

void dgesv_3l(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb, int *info);

double onenorm(int row, int col, double *ptrA);

// double twonormv(int n, double *ptrv);

void padeapprox(int m, int row, double *A);

void expm(int row, double *A);

// void d_compute_qp_size_ocp2dense_rev(int N, int *nx, int *nu, int *nb, int **hidxb, int *ng,
//                                      int *nvd, int *ned, int *nbd, int *ngd);

void acados_eigen_decomposition(int dim, double *A, double *V, double *d, double *e);

double minimum_of_doubles(double *x, int n);

void neville_algorithm(double xx, int n, double *x, double *Q, double *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_UTILS_MATH_H_
