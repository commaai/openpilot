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

// TODO remove
//
void strcp_l_lib(int m, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
void sgead_lib(int m, int n, float alpha, int offsetA, float *A, int sda, int offsetB, float *B, int sdb);
void sgetr_lib(int m, int n, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void strtr_l_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void strtr_u_lib(int m, float alpha, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void sdiareg_lib(int kmax, float reg, int offset, float *pD, int sdd);
void sdiain_sqrt_lib(int kmax, float *x, int offset, float *pD, int sdd);
void sdiaex_lib(int kmax, float alpha, int offset, float *pD, int sdd, float *x);
void sdiaad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
void sdiain_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
void sdiaex_libsp(int kmax, int *idx, float alpha, float *pD, int sdd, float *x);
void sdiaad_libsp(int kmax, int *idx, float alpha, float *x, float *pD, int sdd);
void sdiaadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD, int sdd);
void srowin_lib(int kmax, float alpha, float *x, float *pD);
void srowex_lib(int kmax, float alpha, float *pD, float *x);
void srowad_lib(int kmax, float alpha, float *x, float *pD);
void srowin_libsp(int kmax, float alpha, int *idx, float *x, float *pD);
void srowad_libsp(int kmax, int *idx, float alpha, float *x, float *pD);
void srowadin_libsp(int kmax, int *idx, float alpha, float *x, float *y, float *pD);
void srowsw_lib(int kmax, float *pA, float *pC);
void scolin_lib(int kmax, float *x, int offset, float *pD, int sdd);
void scolad_lib(int kmax, float alpha, float *x, int offset, float *pD, int sdd);
void scolin_libsp(int kmax, int *idx, float *x, float *pD, int sdd);
void scolad_libsp(int kmax, float alpha, int *idx, float *x, float *pD, int sdd);
void scolsw_lib(int kmax, int offsetA, float *pA, int sda, int offsetC, float *pC, int sdc);
void svecin_libsp(int kmax, int *idx, float *x, float *y);
void svecad_libsp(int kmax, int *idx, float alpha, float *x, float *y);
