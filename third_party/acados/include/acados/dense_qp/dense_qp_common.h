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


#ifndef ACADOS_DENSE_QP_DENSE_QP_COMMON_H_
#define ACADOS_DENSE_QP_DENSE_QP_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

// hpipm
#include "hpipm/include/hpipm_d_dense_qp.h"
#include "hpipm/include/hpipm_d_dense_qp_res.h"
#include "hpipm/include/hpipm_d_dense_qp_sol.h"
// acados
#include "acados/utils/types.h"

typedef struct d_dense_qp_dim dense_qp_dims;
typedef struct d_dense_qp dense_qp_in;
typedef struct d_dense_qp_sol dense_qp_out;
typedef struct d_dense_qp_res dense_qp_res;
typedef struct d_dense_qp_res_ws dense_qp_res_ws;



#ifndef QP_SOLVER_CONFIG_
#define QP_SOLVER_CONFIG_
typedef struct
{
    void (*dims_set)(void *config_, void *dims_, const char *field, const int* value);
    acados_size_t (*opts_calculate_size)(void *config, void *dims);
    void *(*opts_assign)(void *config, void *dims, void *raw_memory);
    void (*opts_initialize_default)(void *config, void *dims, void *args);
    void (*opts_update)(void *config, void *dims, void *args);
    void (*opts_set)(void *config_, void *opts_, const char *field, void* value);
    acados_size_t (*memory_calculate_size)(void *config, void *dims, void *args);
    void *(*memory_assign)(void *config, void *dims, void *args, void *raw_memory);
    void (*memory_get)(void *config_, void *mem_, const char *field, void* value);
    acados_size_t (*workspace_calculate_size)(void *config, void *dims, void *args);
    int (*evaluate)(void *config, void *qp_in, void *qp_out, void *args, void *mem, void *work);
    void (*eval_sens)(void *config, void *qp_in, void *qp_out, void *opts, void *mem, void *work);
} qp_solver_config;
#endif



#ifndef QP_INFO_
#define QP_INFO_
typedef struct
{
    double solve_QP_time;
    double condensing_time;
    double interface_time;
    double total_time;
    int num_iter;
    int t_computed;
} qp_info;
#endif



/* config */
//
acados_size_t dense_qp_solver_config_calculate_size();
//
qp_solver_config *dense_qp_solver_config_assign(void *raw_memory);

/* dims */
//
acados_size_t dense_qp_dims_calculate_size();
//
dense_qp_dims *dense_qp_dims_assign(void *raw_memory);
//
void dense_qp_dims_set(void *config_, void *dims_, const char *field, const int* value);
//

/* in */
//
acados_size_t dense_qp_in_calculate_size(dense_qp_dims *dims);
//
dense_qp_in *dense_qp_in_assign(dense_qp_dims *dims, void *raw_memory);

/* out */
//
acados_size_t dense_qp_out_calculate_size(dense_qp_dims *dims);
//
dense_qp_out *dense_qp_out_assign(dense_qp_dims *dims, void *raw_memory);
//
void dense_qp_out_get(dense_qp_out *out, const char *field, void *value);

/* res */
//
acados_size_t dense_qp_res_calculate_size(dense_qp_dims *dims);
//
dense_qp_res *dense_qp_res_assign(dense_qp_dims *dims, void *raw_memory);
//
acados_size_t dense_qp_res_workspace_calculate_size(dense_qp_dims *dims);
//
dense_qp_res_ws *dense_qp_res_workspace_assign(dense_qp_dims *dims, void *raw_memory);
//
void dense_qp_compute_t(dense_qp_in *qp_in, dense_qp_out *qp_out);
//
void dense_qp_res_compute(dense_qp_in *qp_in, dense_qp_out *qp_out, dense_qp_res *qp_res, dense_qp_res_ws *res_ws);
//
void dense_qp_res_compute_nrm_inf(dense_qp_res *qp_res, double res[4]);

/* misc */
//
void dense_qp_stack_slacks_dims(dense_qp_dims *in, dense_qp_dims *out);
//
void dense_qp_stack_slacks(dense_qp_in *in, dense_qp_in *out);
//
void dense_qp_unstack_slacks(dense_qp_out *in, dense_qp_in *qp_out, dense_qp_out *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_DENSE_QP_DENSE_QP_COMMON_H_
