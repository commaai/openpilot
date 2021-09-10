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


///
/// \defgroup ocp_nlp_cost ocp_nlp_cost 
/// 

/// \addtogroup ocp_nlp_cost ocp_nlp_cost
/// @{
/// \addtogroup ocp_nlp_cost_common ocp_nlp_cost_common
/// @{

#ifndef ACADOS_OCP_NLP_OCP_NLP_COST_COMMON_H_
#define ACADOS_OCP_NLP_OCP_NLP_COST_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

// acados
#include "acados/utils/external_function_generic.h"
#include "acados/utils/types.h"



/************************************************
 * config
 ************************************************/

typedef struct
{
    acados_size_t (*dims_calculate_size)(void *config);
    void *(*dims_assign)(void *config, void *raw_memory);
    void (*dims_initialize)(void *config, void *dims, int nx, int nu, int ny, int ns, int nz);
    void (*dims_set)(void *config_, void *dims_, const char *field, int *value);
    void (*dims_get)(void *config_, void *dims_, const char *field, int *value);
    acados_size_t (*model_calculate_size)(void *config, void *dims);
    void *(*model_assign)(void *config, void *dims, void *raw_memory);
    int (*model_set)(void *config_, void *dims_, void *model_, const char *field, void *value_);
    acados_size_t (*opts_calculate_size)(void *config, void *dims);
    void *(*opts_assign)(void *config, void *dims, void *raw_memory);
    void (*opts_initialize_default)(void *config, void *dims, void *opts);
    void (*opts_update)(void *config, void *dims, void *opts);
    void (*opts_set)(void *config, void *opts, const char *field, void *value);
    acados_size_t (*memory_calculate_size)(void *config, void *dims, void *opts);
	double *(*memory_get_fun_ptr)(void *memory);
    struct blasfeo_dvec *(*memory_get_grad_ptr)(void *memory);
    void (*memory_set_ux_ptr)(struct blasfeo_dvec *ux, void *memory);
    void (*memory_set_tmp_ux_ptr)(struct blasfeo_dvec *tmp_ux, void *memory);
    void (*memory_set_z_alg_ptr)(struct blasfeo_dvec *z_alg, void *memory);
    void (*memory_set_dzdux_tran_ptr)(struct blasfeo_dmat *dzdux, void *memory);
    void (*memory_set_RSQrq_ptr)(struct blasfeo_dmat *RSQrq, void *memory);
    void (*memory_set_Z_ptr)(struct blasfeo_dvec *Z, void *memory);
    void *(*memory_assign)(void *config, void *dims, void *opts, void *raw_memory);
    acados_size_t (*workspace_calculate_size)(void *config, void *dims, void *opts);
    void (*initialize)(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);

    // computes the function value, gradient and hessian (approximation) of the cost function
    void (*update_qp_matrices)(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);
    // computes the cost function value (intended for globalization)
    void (*compute_fun)(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);
    void (*config_initialize_default)(void *config);
} ocp_nlp_cost_config;

//
acados_size_t ocp_nlp_cost_config_calculate_size();
//
ocp_nlp_cost_config *ocp_nlp_cost_config_assign(void *raw_memory);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_COST_COMMON_H_
/// @} 
/// @} 
