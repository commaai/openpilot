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


#ifndef INTERFACES_ACADOS_C_DENSE_QP_INTERFACE_H_
#define INTERFACES_ACADOS_C_DENSE_QP_INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/dense_qp/dense_qp_common.h"

typedef enum { DENSE_QP_HPIPM, DENSE_QP_QORE, DENSE_QP_QPOASES, DENSE_QP_OOQP } dense_qp_solver_t;

typedef struct
{
    dense_qp_solver_t qp_solver;
} dense_qp_solver_plan;

typedef struct
{
    qp_solver_config *config;
    void *dims;
    void *opts;
    void *mem;
    void *work;
} dense_qp_solver;

qp_solver_config *dense_qp_config_create(dense_qp_solver_plan *plan);
//
dense_qp_dims *dense_qp_dims_create();
//
dense_qp_in *dense_qp_in_create(qp_solver_config *config, dense_qp_dims *dims);
//
dense_qp_out *dense_qp_out_create(qp_solver_config *config, dense_qp_dims *dims);
//
void *dense_qp_opts_create(qp_solver_config *config, dense_qp_dims *dims);
//
acados_size_t dense_qp_calculate_size(qp_solver_config *config, dense_qp_dims *dims, void *opts_);
//
dense_qp_solver *dense_qp_assign(qp_solver_config *config, dense_qp_dims *dims, void *opts_,
                                 void *raw_memory);
//
dense_qp_solver *dense_qp_create(qp_solver_config *config, dense_qp_dims *dims, void *opts_);
//
int dense_qp_solve(dense_qp_solver *solver, dense_qp_in *qp_in, dense_qp_out *qp_out);
//
void dense_qp_inf_norm_residuals(dense_qp_dims *dims, dense_qp_in *qp_in, dense_qp_out *qp_out,
                                 double *res);
//
bool dense_qp_set_field_double_array(const char *field, double *arr, dense_qp_in *qp_in);
//
bool dense_qp_set_field_int_array(const char *field, int *arr, dense_qp_in *qp_in);
//
bool dense_qp_get_field_double_array(const char *field, dense_qp_in *qp_in, double *arr);
//
bool dense_qp_get_field_int_array(const char *field, dense_qp_in *qp_in, int *arr);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // INTERFACES_ACADOS_C_DENSE_QP_INTERFACE_H_
