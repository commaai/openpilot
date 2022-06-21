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
 * ----------- TOMOVE
 *
 * expecting column major matrices
 *
 */

#include "blasfeo_common.h"


void dtrcp_l_lib(int m, double alpha, int offsetA, double *A, int sda, int offsetB, double *B, int sdb);
void dgead_lib(int m, int n, double alpha, int offsetA, double *A, int sda, int offsetB, double *B, int sdb);
// TODO remove ???
void ddiain_sqrt_lib(int kmax, double *x, int offset, double *pD, int sdd);
// TODO ddiaad1
void ddiareg_lib(int kmax, double reg, int offset, double *pD, int sdd);


void dgetr_lib(int m, int n, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrtr_l_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dtrtr_u_lib(int m, double alpha, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void ddiaex_lib(int kmax, double alpha, int offset, double *pD, int sdd, double *x);
void ddiaad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void ddiain_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
void ddiaex_libsp(int kmax, int *idx, double alpha, double *pD, int sdd, double *x);
void ddiaad_libsp(int kmax, int *idx, double alpha, double *x, double *pD, int sdd);
void ddiaadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD, int sdd);
void drowin_lib(int kmax, double alpha, double *x, double *pD);
void drowex_lib(int kmax, double alpha, double *pD, double *x);
void drowad_lib(int kmax, double alpha, double *x, double *pD);
void drowin_libsp(int kmax, double alpha, int *idx, double *x, double *pD);
void drowad_libsp(int kmax, int *idx, double alpha, double *x, double *pD);
void drowadin_libsp(int kmax, int *idx, double alpha, double *x, double *y, double *pD);
void dcolin_lib(int kmax, double *x, int offset, double *pD, int sdd);
void dcolad_lib(int kmax, double alpha, double *x, int offset, double *pD, int sdd);
void dcolin_libsp(int kmax, int *idx, double *x, double *pD, int sdd);
void dcolad_libsp(int kmax, double alpha, int *idx, double *x, double *pD, int sdd);
void dcolsw_lib(int kmax, int offsetA, double *pA, int sda, int offsetC, double *pC, int sdc);
void dvecin_libsp(int kmax, int *idx, double *x, double *y);
void dvecad_libsp(int kmax, int *idx, double alpha, double *x, double *y);
