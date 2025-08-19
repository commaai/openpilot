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

#ifndef BLASFEO_V_AUX_EXT_DEP_H_
#define BLASFEO_V_AUX_EXT_DEP_H_



#include "blasfeo_target.h"



#ifdef __cplusplus
extern "C" {
#endif



/************************************************
* d_aux_extern_depend_lib.c
************************************************/

#ifdef EXT_DEP

void v_zeros(void **ptrA, int size);
// dynamically allocate size bytes of memory aligned to 64-byte boundaries and set accordingly a pointer to void; set allocated memory to zero
void v_zeros_align(void **ptrA, int size);
// free the memory allocated by v_zeros
void v_free(void *ptrA);
// free the memory allocated by v_zeros_aligned
void v_free_align(void *ptrA);
// dynamically allocate size bytes of memory and set accordingly a pointer to char; set allocated memory to zero
void c_zeros(char **ptrA, int size);
// dynamically allocate size bytes of memory aligned to 64-byte boundaries and set accordingly a pointer to char; set allocated memory to zero
void c_zeros_align(char **ptrA, int size);
// free the memory allocated by c_zeros
void c_free(char *ptrA);
// free the memory allocated by c_zeros_aligned
void c_free_align(char *ptrA);

#endif // EXT_DEP



#ifdef __cplusplus
}
#endif



#endif  // BLASFEO_V_AUX_EXT_DEP_H_
