/*
 * Copyright (c) The acados authors.
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


#ifndef ACADOS_UTILS_MEM_H_
#define ACADOS_UTILS_MEM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>

#include "types.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

// TODO(dimitris): probably does not belong here
typedef struct
{
    int (*fun)(void *);
    acados_size_t (*calculate_args_size)(void *);
    void *(*assign_args)(void *);
    void (*initialize_default_args)(void *);
    acados_size_t (*calculate_memory_size)(void *);
    void *(*assign_memory)(void *);
    acados_size_t (*calculate_workspace_size)(void *);
} module_solver;

// make int counter of memory multiple of a number (typically 8 or 64)
void make_int_multiple_of(acados_size_t num, acados_size_t *size);

// align char pointer to number (typically 8 for pointers and doubles,
// 64 for blasfeo structs) and return offset
int align_char_to(int num, char **c_ptr);

// switch between malloc and calloc (for valgrinding)
void *acados_malloc(size_t nitems, acados_size_t size);

// uses always calloc
void *acados_calloc(size_t nitems, acados_size_t size);

// allocate vector of pointers to vectors of doubles and advance pointer
void assign_and_advance_double_ptrs(int n, double ***v, char **ptr);

// allocate vector of pointers to vectors of ints and advance pointer
void assign_and_advance_int_ptrs(int n, int ***v, char **ptr);

// allocate vector of pointers to strvecs and advance pointer
void assign_and_advance_blasfeo_dvec_structs(int n, struct blasfeo_dvec **sv, char **ptr);

// allocate vector of pointers to strmats and advance pointer
void assign_and_advance_blasfeo_dmat_structs(int n, struct blasfeo_dmat **sm, char **ptr);

// allocate vector of pointers to vector of pointers to strmats and advance pointer
void assign_and_advance_blasfeo_dmat_ptrs(int n, struct blasfeo_dmat ***sm, char **ptr);

// allocate vector of chars and advance pointer
void assign_and_advance_char(int n, char **v, char **ptr);

// allocate vector of ints and advance pointer
void assign_and_advance_int(int n, int **v, char **ptr);

// allocate vector of bools and advance pointer
void assign_and_advance_bool(int n, bool **v, char **ptr);

// allocate vector of doubles and advance pointer
void assign_and_advance_double(int n, double **v, char **ptr);

// allocate strvec and advance pointer
void assign_and_advance_blasfeo_dvec_mem(int n, struct blasfeo_dvec *sv, char **ptr);

// allocate strmat and advance pointer
void assign_and_advance_blasfeo_dmat_mem(int m, int n, struct blasfeo_dmat *sA, char **ptr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_UTILS_MEM_H_
