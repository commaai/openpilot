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


#ifndef ACADOS_OCP_QP_OCP_QP_PARTIAL_CONDENSING_SOLVER_H_
#define ACADOS_OCP_QP_OCP_QP_PARTIAL_CONDENSING_SOLVER_H_

#ifdef __cplusplus
extern "C" {
#endif

// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/utils/types.h"



typedef struct
{
    ocp_qp_dims *orig_dims;
    void *xcond_dims;
} ocp_qp_xcond_solver_dims;



typedef struct ocp_qp_xcond_solver_opts_
{
    void *xcond_opts;
    void *qp_solver_opts;
} ocp_qp_xcond_solver_opts;



typedef struct ocp_qp_xcond_solver_memory_
{
    void *xcond_memory;
    void *solver_memory;
    void *xcond_qp_in;
    void *xcond_qp_out;
} ocp_qp_xcond_solver_memory;



typedef struct ocp_qp_xcond_solver_workspace_
{
    void *xcond_work;
    void *qp_solver_work;
} ocp_qp_xcond_solver_workspace;



typedef struct
{
    acados_size_t (*dims_calculate_size)(void *config, int N);
    ocp_qp_xcond_solver_dims *(*dims_assign)(void *config, int N, void *raw_memory);
    void (*dims_set)(void *config_, ocp_qp_xcond_solver_dims *dims, int stage, const char *field, int* value);
    void (*dims_get)(void *config_, ocp_qp_xcond_solver_dims *dims, int stage, const char *field, int* value);
    acados_size_t (*opts_calculate_size)(void *config, ocp_qp_xcond_solver_dims *dims);
    void *(*opts_assign)(void *config, ocp_qp_xcond_solver_dims *dims, void *raw_memory);
    void (*opts_initialize_default)(void *config, ocp_qp_xcond_solver_dims *dims, void *opts);
    void (*opts_update)(void *config, ocp_qp_xcond_solver_dims *dims, void *opts);
    void (*opts_set)(void *config_, void *opts_, const char *field, void* value);
    acados_size_t (*memory_calculate_size)(void *config, ocp_qp_xcond_solver_dims *dims, void *opts);
    void *(*memory_assign)(void *config, ocp_qp_xcond_solver_dims *dims, void *opts, void *raw_memory);
    void (*memory_get)(void *config_, void *mem_, const char *field, void* value);
    void (*memory_reset)(void *config, ocp_qp_xcond_solver_dims *dims, ocp_qp_in *qp_in, ocp_qp_out *qp_out, void *opts, void *mem, void *work);
    acados_size_t (*workspace_calculate_size)(void *config, ocp_qp_xcond_solver_dims *dims, void *opts);
    int (*evaluate)(void *config, ocp_qp_xcond_solver_dims *dims, ocp_qp_in *qp_in, ocp_qp_out *qp_out, void *opts, void *mem, void *work);
    void (*eval_sens)(void *config, ocp_qp_xcond_solver_dims *dims, ocp_qp_in *param_qp_in, ocp_qp_out *sens_qp_out, void *opts, void *mem, void *work);
    qp_solver_config *qp_solver;  // either ocp_qp_solver or dense_solver
    ocp_qp_xcond_config *xcond;
} ocp_qp_xcond_solver_config;  // pcond - partial condensing or fcond - full condensing



/* config */
//
acados_size_t ocp_qp_xcond_solver_config_calculate_size();
//
ocp_qp_xcond_solver_config *ocp_qp_xcond_solver_config_assign(void *raw_memory);

/* dims */
//
acados_size_t ocp_qp_xcond_solver_dims_calculate_size(void *config, int N);
//
ocp_qp_xcond_solver_dims *ocp_qp_xcond_solver_dims_assign(void *config, int N, void *raw_memory);
//
void ocp_qp_xcond_solver_dims_set_(void *config, ocp_qp_xcond_solver_dims *dims, int stage, const char *field, int* value);

/* opts */
//
acados_size_t ocp_qp_xcond_solver_opts_calculate_size(void *config, ocp_qp_xcond_solver_dims *dims);
//
void *ocp_qp_xcond_solver_opts_assign(void *config, ocp_qp_xcond_solver_dims *dims, void *raw_memory);
//
void ocp_qp_xcond_solver_opts_initialize_default(void *config, ocp_qp_xcond_solver_dims *dims, void *opts_);
//
void ocp_qp_xcond_solver_opts_update(void *config, ocp_qp_xcond_solver_dims *dims, void *opts_);
//
void ocp_qp_xcond_solver_opts_set_(void *config_, void *opts_, const char *field, void* value);

/* memory */
//
acados_size_t ocp_qp_xcond_solver_memory_calculate_size(void *config, ocp_qp_xcond_solver_dims *dims, void *opts_);
//
void *ocp_qp_xcond_solver_memory_assign(void *config, ocp_qp_xcond_solver_dims *dims, void *opts_, void *raw_memory);

/* workspace */
//
acados_size_t ocp_qp_xcond_solver_workspace_calculate_size(void *config, ocp_qp_xcond_solver_dims *dims, void *opts_);

/* config */
//
int ocp_qp_xcond_solver(void *config, ocp_qp_xcond_solver_dims *dims, ocp_qp_in *qp_in, ocp_qp_out *qp_out, void *opts_, void *mem_, void *work_);

//
void ocp_qp_xcond_solver_config_initialize_default(void *config_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_QP_OCP_QP_PARTIAL_CONDENSING_SOLVER_H_
