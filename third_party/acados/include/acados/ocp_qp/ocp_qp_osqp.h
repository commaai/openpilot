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


#ifndef ACADOS_OCP_QP_OCP_QP_OSQP_H_
#define ACADOS_OCP_QP_OCP_QP_OSQP_H_

#ifdef __cplusplus
extern "C" {
#endif

// osqp
#include "osqp/include/types.h"

// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/utils/types.h"

typedef struct ocp_qp_osqp_opts_
{
    OSQPSettings *osqp_opts;
} ocp_qp_osqp_opts;


typedef struct ocp_qp_osqp_memory_
{
    c_int first_run;

    c_float *q;
    c_float *l;
    c_float *u;

    c_int P_nnzmax;
    c_int *P_i;
    c_int *P_p;
    c_float *P_x;

    c_int A_nnzmax;
    c_int *A_i;
    c_int *A_p;
    c_float *A_x;

    OSQPData *osqp_data;
    OSQPWorkspace *osqp_work;

    double time_qp_solver_call;
    int iter;
    int status;

} ocp_qp_osqp_memory;

acados_size_t ocp_qp_osqp_opts_calculate_size(void *config, void *dims);
//
void *ocp_qp_osqp_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_qp_osqp_opts_initialize_default(void *config, void *dims, void *opts_);
//
void ocp_qp_osqp_opts_update(void *config, void *dims, void *opts_);
//
acados_size_t ocp_qp_osqp_memory_calculate_size(void *config, void *dims, void *opts_);
//
void *ocp_qp_osqp_memory_assign(void *config, void *dims, void *opts_, void *raw_memory);
//
acados_size_t ocp_qp_osqp_workspace_calculate_size(void *config, void *dims, void *opts_);
//
int ocp_qp_osqp(void *config, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void ocp_qp_osqp_eval_sens(void *config_, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void ocp_qp_osqp_config_initialize_default(void *config);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_QP_OCP_QP_OSQP_H_
