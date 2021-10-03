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

#ifndef BLASFEO_S_AUX_EXT_DEP_REF_H_
#define BLASFEO_S_AUX_EXT_DEP_REF_H_

#if defined(EXT_DEP)



#include <stdio.h>

#include "blasfeo_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// expose reference BLASFEO for testing
// see blasfeo_s_aux_exp_dep.h for help

void blasfeo_print_smat_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_allocate_smat_ref(int m, int n, struct blasfeo_smat_ref *sA);
void blasfeo_allocate_svec_ref(int m, struct blasfeo_svec_ref *sa);
void blasfeo_free_smat_ref(struct blasfeo_smat_ref *sA);
void blasfeo_free_svec_ref(struct blasfeo_svec_ref *sa);
void blasfeo_print_smat_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_exp_smat_ref(int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_to_file_smat_ref(FILE *file, int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_to_file_exp_smat_ref(FILE *file, int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_to_string_smat_ref(char **buf_out, int m, int n, struct blasfeo_smat_ref *sA, int ai, int aj);
void blasfeo_print_svec(int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_exp_svec(int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_to_file_svec(FILE *file, int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_to_string_svec(char **buf_out, int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_tran_svec(int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_exp_tran_svec(int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_to_file_tran_svec(FILE *file, int m, struct blasfeo_svec *sa, int ai);
void blasfeo_print_to_string_tran_svec(char **buf_out, int m, struct blasfeo_svec *sa, int ai);


#ifdef __cplusplus
}
#endif



#endif  // EXT_DEP

#endif  // BLASFEO_S_AUX_EXT_DEP_REF_H_
