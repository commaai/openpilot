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


#ifndef ACADOS_UTILS_EXTERNAL_FUNCTION_GENERIC_H_
#define ACADOS_UTILS_EXTERNAL_FUNCTION_GENERIC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/utils/types.h"

/************************************************
 * generic external function
 ************************************************/

// type of arguments
typedef enum {
    COLMAJ,
    BLASFEO_DMAT,
    BLASFEO_DVEC,
    COLMAJ_ARGS,
    BLASFEO_DMAT_ARGS,
    BLASFEO_DVEC_ARGS,
    IGNORE_ARGUMENT
} ext_fun_arg_t;

struct colmaj_args
{
    double *A;
    int lda;
};

struct blasfeo_dmat_args
{
    struct blasfeo_dmat *A;
    int ai;
    int aj;
};

struct blasfeo_dvec_args
{
    struct blasfeo_dvec *x;
    int xi;
};

// prototype of an external function
typedef struct
{
    // public members (have to be before private ones)
    void (*evaluate)(void *, ext_fun_arg_t *, void **, ext_fun_arg_t *, void **);
    // private members
    // .....
} external_function_generic;



/************************************************
 * generic external parametric function
 ************************************************/

// prototype of a parametric external function
typedef struct
{
    // public members for core (have to be before private ones)
    void (*evaluate)(void *, ext_fun_arg_t *, void **, ext_fun_arg_t *, void **);
	// public members for interfaces
    void (*get_nparam)(void *, int *);
    void (*set_param)(void *, double *);
    void (*set_param_sparse)(void *, int n_update, int *idx, double *);
    // private members
    void *ptr_ext_mem;  // pointer to external memory
    int (*fun)(void **, void **, void *);
    double *p;  // parameters
    int np;     // number of parameters
    // .....
} external_function_param_generic;

//
acados_size_t external_function_param_generic_struct_size();
//
void external_function_param_generic_set_fun(external_function_param_generic *fun, void *value);
//
acados_size_t external_function_param_generic_calculate_size(external_function_param_generic *fun, int np);
//
void external_function_param_generic_assign(external_function_param_generic *fun, void *mem);
//
void external_function_param_generic_wrapper(void *self, ext_fun_arg_t *type_in, void **in, ext_fun_arg_t *type_out, void **out);
//
void external_function_param_generic_get_nparam(void *self, int *np);
//
void external_function_param_generic_set_param(void *self, double *p);


/************************************************
 * casadi external function
 ************************************************/

typedef struct
{
    // public members (have to be the same as in the prototype, and before the private ones)
    void (*evaluate)(void *, ext_fun_arg_t *, void **, ext_fun_arg_t *, void **);
    // private members
    void *ptr_ext_mem;  // pointer to external memory
    int (*casadi_fun)(const double **, double **, int *, double *, void *);
    int (*casadi_work)(int *, int *, int *, int *);
    const int *(*casadi_sparsity_in)(int);
    const int *(*casadi_sparsity_out)(int);
    int (*casadi_n_in)();
    int (*casadi_n_out)();
    double **args;
    double **res;
    double *w;
    int *iw;
    int *args_size;     // size of args[i]
    int *res_size;      // size of res[i]
    int args_num;       // number of args arrays
    int args_size_tot;  // total size of args arrays
    int res_num;        // number of res arrays
    int res_size_tot;   // total size of res arrays
    int in_num;         // number of input arrays
    int out_num;        // number of output arrays
    int iw_size;        // number of ints for worksapce
    int w_size;         // number of doubles for workspace
} external_function_casadi;

//
acados_size_t external_function_casadi_struct_size();
//
void external_function_casadi_set_fun(external_function_casadi *fun, void *value);
//
void external_function_casadi_set_work(external_function_casadi *fun, void *value);
//
void external_function_casadi_set_sparsity_in(external_function_casadi *fun, void *value);
//
void external_function_casadi_set_sparsity_out(external_function_casadi *fun, void *value);
//
void external_function_casadi_set_n_in(external_function_casadi *fun, void *value);
//
void external_function_casadi_set_n_out(external_function_casadi *fun, void *value);
//
acados_size_t external_function_casadi_calculate_size(external_function_casadi *fun);
//
void external_function_casadi_assign(external_function_casadi *fun, void *mem);
//
void external_function_casadi_wrapper(void *self, ext_fun_arg_t *type_in, void **in,
                                      ext_fun_arg_t *type_out, void **out);

/************************************************
 * casadi external parametric function
 ************************************************/

typedef struct
{
    // public members for core (have to be the same as in the prototype, and before the private ones)
    void (*evaluate)(void *, ext_fun_arg_t *, void **, ext_fun_arg_t *, void **);
	// public members for interfaces
    void (*get_nparam)(void *, int *);
    void (*set_param)(void *, double *);
    void (*set_param_sparse)(void *, int n_update, int *idx, double *);
    // private members
    void *ptr_ext_mem;  // pointer to external memory
    int (*casadi_fun)(const double **, double **, int *, double *, void *);
    int (*casadi_work)(int *, int *, int *, int *);
    const int *(*casadi_sparsity_in)(int);
    const int *(*casadi_sparsity_out)(int);
    int (*casadi_n_in)();
    int (*casadi_n_out)();
    double **args;
    double **res;
    double *w;
    int *iw;
    int *args_size;     // size of args[i]
    int *res_size;      // size of res[i]
    int args_num;       // number of args arrays
    int args_size_tot;  // total size of args arrays
    int res_num;        // number of res arrays
    int res_size_tot;   // total size of res arrays
    int in_num;         // number of input arrays
    int out_num;        // number of output arrays
    int iw_size;        // number of ints for worksapce
    int w_size;         // number of doubles for workspace
    int np;             // number of parameters
} external_function_param_casadi;

//
acados_size_t external_function_param_casadi_struct_size();
//
void external_function_param_casadi_set_fun(external_function_param_casadi *fun, void *value);
//
void external_function_param_casadi_set_work(external_function_param_casadi *fun, void *value);
//
void external_function_param_casadi_set_sparsity_in(external_function_param_casadi *fun, void *value);
//
void external_function_param_casadi_set_sparsity_out(external_function_param_casadi *fun, void *value);
//
void external_function_param_casadi_set_n_in(external_function_param_casadi *fun, void *value);
//
void external_function_param_casadi_set_n_out(external_function_param_casadi *fun, void *value);
//
acados_size_t external_function_param_casadi_calculate_size(external_function_param_casadi *fun, int np);
//
void external_function_param_casadi_assign(external_function_param_casadi *fun, void *mem);
//
void external_function_param_casadi_wrapper(void *self, ext_fun_arg_t *type_in, void **in,
                                            ext_fun_arg_t *type_out, void **out);
//
void external_function_param_casadi_get_nparam(void *self, int *np);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_UTILS_EXTERNAL_FUNCTION_GENERIC_H_
