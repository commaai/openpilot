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


#ifndef ACADOS_OCP_QP_OCP_QP_COMMON_H_
#define ACADOS_OCP_QP_OCP_QP_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

// hpipm
#include "hpipm/include/hpipm_d_ocp_qp.h"
#include "hpipm/include/hpipm_d_ocp_qp_dim.h"
#include "hpipm/include/hpipm_d_ocp_qp_res.h"
#include "hpipm/include/hpipm_d_ocp_qp_sol.h"
// acados
#include "acados/utils/types.h"



typedef struct d_ocp_qp_dim ocp_qp_dims;
typedef struct d_ocp_qp ocp_qp_in;
typedef struct d_ocp_qp_sol ocp_qp_out;
typedef struct d_ocp_qp_res ocp_qp_res;
typedef struct d_ocp_qp_res_ws ocp_qp_res_ws;



#ifndef QP_SOLVER_CONFIG_
#define QP_SOLVER_CONFIG_
typedef struct
{
    void (*dims_set)(void *config_, void *dims_, int stage, const char *field, int* value);
    acados_size_t (*opts_calculate_size)(void *config, void *dims);
    void *(*opts_assign)(void *config, void *dims, void *raw_memory);
    void (*opts_initialize_default)(void *config, void *dims, void *opts);
    void (*opts_update)(void *config, void *dims, void *opts);
    void (*opts_set)(void *config_, void *opts_, const char *field, void* value);
    acados_size_t (*memory_calculate_size)(void *config, void *dims, void *opts);
    void *(*memory_assign)(void *config, void *dims, void *opts, void *raw_memory);
    void (*memory_get)(void *config_, void *mem_, const char *field, void* value);
    acados_size_t (*workspace_calculate_size)(void *config, void *dims, void *opts);
    int (*evaluate)(void *config, void *qp_in, void *qp_out, void *opts, void *mem, void *work);
    void (*eval_sens)(void *config, void *qp_in, void *qp_out, void *opts, void *mem, void *work);
} qp_solver_config;
#endif



typedef struct
{
    acados_size_t (*dims_calculate_size)(void *config, int N);
    void *(*dims_assign)(void *config, int N, void *raw_memory);
    void (*dims_set)(void *config, void *dims_, int stage, const char *field, int* value);
    void (*dims_get)(void *config, void *dims, const char *field, void* value);
    // TODO add config everywhere !!!!!
    acados_size_t (*opts_calculate_size)(void *dims);
    void *(*opts_assign)(void *dims, void *raw_memory);
    void (*opts_initialize_default)(void *dims, void *opts);
    void (*opts_update)(void *dims, void *opts);
    void (*opts_set)(void *opts_, const char *field, void* value);
    acados_size_t (*memory_calculate_size)(void *dims, void *opts);
    void *(*memory_assign)(void *dims, void *opts, void *raw_memory);
    void (*memory_get)(void *config, void *mem, const char *field, void* value);
    acados_size_t (*workspace_calculate_size)(void *dims, void *opts);
    int (*condensing)(void *qp_in, void *qp_out, void *opts, void *mem, void *work);
    int (*condensing_rhs)(void *qp_in, void *qp_out, void *opts, void *mem, void *work);
    int (*expansion)(void *qp_in, void *qp_out, void *opts, void *mem, void *work);
} ocp_qp_xcond_config;



/// Struct containing metrics of the qp solver.
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
acados_size_t ocp_qp_solver_config_calculate_size();
//
qp_solver_config *ocp_qp_solver_config_assign(void *raw_memory);
//
acados_size_t ocp_qp_condensing_config_calculate_size();
//
ocp_qp_xcond_config *ocp_qp_condensing_config_assign(void *raw_memory);


/* dims */
//
acados_size_t ocp_qp_dims_calculate_size(int N);
//
ocp_qp_dims *ocp_qp_dims_assign(int N, void *raw_memory);
//
void ocp_qp_dims_set(void *config_, void *dims, int stage, const char *field, int* value);
//
void ocp_qp_dims_get(void *config_, void *dims, int stage, const char *field, int* value);


/* in */
//
acados_size_t ocp_qp_in_calculate_size(ocp_qp_dims *dims);
//
ocp_qp_in *ocp_qp_in_assign(ocp_qp_dims *dims, void *raw_memory);


/* out */
//
acados_size_t ocp_qp_out_calculate_size(ocp_qp_dims *dims);
//
ocp_qp_out *ocp_qp_out_assign(ocp_qp_dims *dims, void *raw_memory);

/* res */
//
acados_size_t ocp_qp_res_calculate_size(ocp_qp_dims *dims);
//
ocp_qp_res *ocp_qp_res_assign(ocp_qp_dims *dims, void *raw_memory);
//
acados_size_t ocp_qp_res_workspace_calculate_size(ocp_qp_dims *dims);
//
ocp_qp_res_ws *ocp_qp_res_workspace_assign(ocp_qp_dims *dims, void *raw_memory);
//
void ocp_qp_res_compute(ocp_qp_in *qp_in, ocp_qp_out *qp_out, ocp_qp_res *qp_res, ocp_qp_res_ws *res_ws);
//
void ocp_qp_res_compute_nrm_inf(ocp_qp_res *qp_res, double res[4]);


/* misc */
//
void ocp_qp_stack_slacks_dims(ocp_qp_dims *in, ocp_qp_dims *out);
//
void ocp_qp_stack_slacks(ocp_qp_in *in, ocp_qp_in *out);
//
void ocp_qp_compute_t(ocp_qp_in *qp_in, ocp_qp_out *qp_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_QP_OCP_QP_COMMON_H_
