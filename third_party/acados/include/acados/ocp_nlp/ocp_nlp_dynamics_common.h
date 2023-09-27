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


/// \ingroup ocp_nlp
/// @{

/// \defgroup ocp_nlp_dynamics ocp_nlp_dynamics
/// @{

#ifndef ACADOS_OCP_NLP_OCP_NLP_DYNAMICS_COMMON_H_
#define ACADOS_OCP_NLP_OCP_NLP_DYNAMICS_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif



// blasfeo
#include "blasfeo/include/blasfeo_common.h"

// acados
#include "acados/sim/sim_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/types.h"



/************************************************
 * config
 ************************************************/

typedef struct
{
    void (*config_initialize_default)(void *config);
    sim_config *sim_solver;
    /* dims */
    acados_size_t (*dims_calculate_size)(void *config);
    void *(*dims_assign)(void *config, void *raw_memory);
    void (*dims_initialize)(void *config, void *dims, int nx, int nu, int nx1, int nu1, int nz);
    void (*dims_set)(void *config_, void *dims_, const char *field, int *value);
    void (*dims_get)(void *config_, void *dims_, const char *field, int* value);
    /* model */
    acados_size_t (*model_calculate_size)(void *config, void *dims);
    void *(*model_assign)(void *config, void *dims, void *raw_memory);
    void (*model_set)(void *config_, void *dims_, void *model_, const char *field, void *value_);
    /* opts */
    acados_size_t (*opts_calculate_size)(void *config, void *dims);
    void *(*opts_assign)(void *config, void *dims, void *raw_memory);
    void (*opts_initialize_default)(void *config, void *dims, void *opts);
    void (*opts_set)(void *config_, void *opts_, const char *field, void *value);
    void (*opts_update)(void *config, void *dims, void *opts);
    /* memory */
    acados_size_t (*memory_calculate_size)(void *config, void *dims, void *opts);
    void *(*memory_assign)(void *config, void *dims, void *opts, void *raw_memory);
    // get shooting node gap x_next(x_n, u_n) - x_{n+1}
    struct blasfeo_dvec *(*memory_get_fun_ptr)(void *memory_);
    struct blasfeo_dvec *(*memory_get_adj_ptr)(void *memory_);
    void (*memory_set_ux_ptr)(struct blasfeo_dvec *ux, void *memory_);
    void (*memory_set_tmp_ux_ptr)(struct blasfeo_dvec *tmp_ux, void *memory_);
    void (*memory_set_ux1_ptr)(struct blasfeo_dvec *ux1, void *memory_);
    void (*memory_set_tmp_ux1_ptr)(struct blasfeo_dvec *tmp_ux1, void *memory_);
    void (*memory_set_pi_ptr)(struct blasfeo_dvec *pi, void *memory_);
    void (*memory_set_tmp_pi_ptr)(struct blasfeo_dvec *tmp_pi, void *memory_);
    void (*memory_set_BAbt_ptr)(struct blasfeo_dmat *BAbt, void *memory_);
    void (*memory_set_RSQrq_ptr)(struct blasfeo_dmat *RSQrq, void *memory_);
    void (*memory_set_dzduxt_ptr)(struct blasfeo_dmat *mat, void *memory_);
    void (*memory_set_sim_guess_ptr)(struct blasfeo_dvec *vec, bool *bool_ptr, void *memory_);
    void (*memory_set_z_alg_ptr)(struct blasfeo_dvec *vec, void *memory_);
    void (*memory_get)(void *config, void *dims, void *mem, const char *field, void* value);
    /* workspace */
    acados_size_t (*workspace_calculate_size)(void *config, void *dims, void *opts);
    void (*initialize)(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);
    void (*update_qp_matrices)(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);
    void (*compute_fun)(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);
    int (*precompute)(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);
} ocp_nlp_dynamics_config;

//
acados_size_t ocp_nlp_dynamics_config_calculate_size();
//
ocp_nlp_dynamics_config *ocp_nlp_dynamics_config_assign(void *raw_memory);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_DYNAMICS_COMMON_H_
/// @}
/// @}
