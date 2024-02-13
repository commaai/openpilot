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


#ifndef ACADOS_DENSE_QP_DENSE_QP_DAQP_H_
#define ACADOS_DENSE_QP_DENSE_QP_DAQP_H_

#ifdef __cplusplus
extern "C" {
#endif

// blasfeo
#include "blasfeo/include/blasfeo_common.h"

// daqp
#include "daqp/include/types.h"

// acados
#include "acados/dense_qp/dense_qp_common.h"
#include "acados/utils/types.h"


typedef struct dense_qp_daqp_opts_
{
    DAQPSettings* daqp_opts;
    int warm_start;
} dense_qp_daqp_opts;


typedef struct dense_qp_daqp_memory_
{
    double* lb_tmp;
    double* ub_tmp;
    int* idxb;
    int* idxv_to_idxb;
    int* idxs;
    int* idxdaqp_to_idxs;

    double* Zl;
    double* Zu;
    double* zl;
    double* zu;
    double* d_ls;
    double* d_us;

    double time_qp_solver_call;
    int iter;
    DAQPWorkspace * daqp_work;

} dense_qp_daqp_memory;

// opts
acados_size_t dense_qp_daqp_opts_calculate_size(void *config, dense_qp_dims *dims);
//
void *dense_qp_daqp_opts_assign(void *config, dense_qp_dims *dims, void *raw_memory);
//
void dense_qp_daqp_opts_initialize_default(void *config, dense_qp_dims *dims, void *opts_);
//
void dense_qp_daqp_opts_update(void *config, dense_qp_dims *dims, void *opts_);
//
// memory 
acados_size_t dense_qp_daqp_workspace_calculate_size(void *config, dense_qp_dims *dims, void *opts_);
//
void *dense_qp_daqp_workspace_assign(void *config, dense_qp_dims *dims, void *raw_memory);
//
acados_size_t dense_qp_daqp_memory_calculate_size(void *config, dense_qp_dims *dims, void *opts_);
//
void *dense_qp_daqp_memory_assign(void *config, dense_qp_dims *dims, void *opts_, void *raw_memory);
//
// functions
int dense_qp_daqp(void *config, dense_qp_in *qp_in, dense_qp_out *qp_out, void *opts_, void *memory_, void *work_);
//
void dense_qp_daqp_eval_sens(void *config_, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void dense_qp_daqp_memory_reset(void *config_, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void dense_qp_daqp_config_initialize_default(void *config_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_DENSE_QP_DENSE_QP_DAQP_H_
