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
 * auxiliary algebra operation external dependancies header
 *
 * include/blasfeo_d_aux_ext_dep.h
 *
 * - dynamic memory allocation
 * - print
 *
 */

#ifndef BLASFEO_D_AUX_EXT_DEP_REF_H_
#define BLASFEO_D_AUX_EXT_DEP_REF_H_


#include <stdio.h>

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// expose reference BLASFEO for testing
// see blasfeo_d_aux_exp_dep.h for help

void blasfeo_print_dmat_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_allocate_dmat_ref(int m, int n, struct blasfeo_dmat_ref *sA);
void blasfeo_allocate_dvec_ref(int m, struct blasfeo_dvec_ref *sa);
void blasfeo_free_dmat_ref(struct blasfeo_dmat_ref *sA);
void blasfeo_free_dvec_ref(struct blasfeo_dvec_ref *sa);
void blasfeo_print_dmat_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_exp_dmat_ref(int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_to_file_dmat_ref(FILE *file, int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_to_file_exp_dmat_ref(FILE *file, int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_to_string_dmat_ref(char **buf_out, int m, int n, struct blasfeo_dmat_ref *sA, int ai, int aj);
void blasfeo_print_dvec(int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_exp_dvec(int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_to_file_dvec(FILE *file, int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_to_string_dvec(char **buf_out, int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_tran_dvec(int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_exp_tran_dvec(int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_to_file_tran_dvec(FILE *file, int m, struct blasfeo_dvec *sa, int ai);
void blasfeo_print_to_string_tran_dvec(char **buf_out, int m, struct blasfeo_dvec *sa, int ai);

#ifdef __cplusplus
}
#endif

#endif  // BLASFEO_D_AUX_EXT_DEP_REF_H_
