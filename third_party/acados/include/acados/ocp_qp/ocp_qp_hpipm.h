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


#ifndef ACADOS_OCP_QP_OCP_QP_HPIPM_H_
#define ACADOS_OCP_QP_OCP_QP_HPIPM_H_

#ifdef __cplusplus
extern "C" {
#endif

// hpipm
#include "hpipm/include/hpipm_d_ocp_qp_ipm.h"
// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/utils/types.h"



// struct of arguments to the solver
// TODO(roversch): why not make this a typedef of the underlying struct?
typedef struct ocp_qp_hpipm_opts_
{
    struct d_ocp_qp_ipm_arg *hpipm_opts;
} ocp_qp_hpipm_opts;



// TODO(roversch): why not make this a typedef of the underlying struct?
// struct of the solver memory
typedef struct ocp_qp_hpipm_memory_
{
    struct d_ocp_qp_ipm_ws *hpipm_workspace;
    double time_qp_solver_call;
    int iter;

} ocp_qp_hpipm_memory;



//
acados_size_t ocp_qp_hpipm_opts_calculate_size(void *config, void *dims);
//
void *ocp_qp_hpipm_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_qp_hpipm_opts_initialize_default(void *config, void *dims, void *opts_);
//
void ocp_qp_hpipm_opts_update(void *config, void *dims, void *opts_);
//
void ocp_qp_hpipm_opts_set(void *config_, void *opts_, const char *field, void *value);
//
acados_size_t ocp_qp_hpipm_memory_calculate_size(void *config, void *dims, void *opts_);
//
void *ocp_qp_hpipm_memory_assign(void *config, void *dims, void *opts_, void *raw_memory);
//
acados_size_t ocp_qp_hpipm_workspace_calculate_size(void *config, void *dims, void *opts_);
//
int ocp_qp_hpipm(void *config, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void ocp_qp_hpipm_eval_sens(void *config, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void ocp_qp_hpipm_config_initialize_default(void *config);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_QP_OCP_QP_HPIPM_H_
