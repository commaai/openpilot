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


#ifndef ACADOS_OCP_QP_OCP_QP_QPDUNES_H_
#define ACADOS_OCP_QP_OCP_QP_QPDUNES_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "qpDUNES.h"

#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/utils/types.h"

typedef enum qpdunes_options_t_ {
    QPDUNES_DEFAULT_ARGUMENTS,
    QPDUNES_LINEAR_MPC,     // TODO(dimitris): partly implemented
    QPDUNES_NONLINEAR_MPC,  // TODO(dimitris): not implemented yet
    QPDUNES_ACADO_SETTINGS
} qpdunes_options_t;

typedef enum { QPDUNES_WITH_QPOASES, QPDUNES_WITH_CLIPPING } qpdunes_stage_qp_solver_t;

typedef struct ocp_qp_qpdunes_opts_
{
    qpOptions_t options;
    qpdunes_stage_qp_solver_t stageQpSolver;
    int warmstart;  // warmstart = 0: all multipliers set to zero, warmstart = 1: use previous mult.
    bool isLinearMPC;
} ocp_qp_qpdunes_opts;

typedef struct ocp_qp_qpdunes_memory_
{
    int firstRun;
    int nx;
    int nu;
    int nz;
    int nDmax;  // max(dims->ng)
    qpData_t qpData;
    double time_qp_solver_call;
    int iter;
    int status;

} ocp_qp_qpdunes_memory;

typedef struct ocp_qp_qpdunes_workspace_
{
    double *H;
    double *Q;
    double *R;
    double *S;
    double *g;
    double *ABt;
    double *b;
    double *Ct;
    double *lc;
    double *uc;
    double *zLow;
    double *zUpp;
} ocp_qp_qpdunes_workspace;

//
acados_size_t ocp_qp_qpdunes_opts_calculate_size(void *config_, ocp_qp_dims *dims);
//
void *ocp_qp_qpdunes_opts_assign(void *config_, ocp_qp_dims *dims, void *raw_memory);
//
void ocp_qp_qpdunes_opts_initialize_default(void *config_, ocp_qp_dims *dims, void *opts_);
//
void ocp_qp_qpdunes_opts_update(void *config_, ocp_qp_dims *dims, void *opts_);
//
acados_size_t ocp_qp_qpdunes_memory_calculate_size(void *config_, ocp_qp_dims *dims, void *opts_);
//
void *ocp_qp_qpdunes_memory_assign(void *config_, ocp_qp_dims *dims, void *opts_, void *raw_memory);
//
acados_size_t ocp_qp_qpdunes_workspace_calculate_size(void *config_, ocp_qp_dims *dims, void *opts_);
//
int ocp_qp_qpdunes(void *config_, ocp_qp_in *qp_in, ocp_qp_out *qp_out, void *opts_, void *memory_,
                   void *work_);
//
void ocp_qp_qpdunes_free_memory(void *mem_);
//
void ocp_qp_qpdunes_eval_sens(void *config_, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void ocp_qp_qpdunes_config_initialize_default(void *config_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_QP_OCP_QP_QPDUNES_H_
