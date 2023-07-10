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


/// \addtogroup ocp_nlp
/// @{
/// \addtogroup ocp_nlp_dynamics
/// @{

#ifndef ACADOS_OCP_NLP_OCP_NLP_DYNAMICS_CONT_H_
#define ACADOS_OCP_NLP_OCP_NLP_DYNAMICS_CONT_H_

#ifdef __cplusplus
extern "C" {
#endif



// blasfeo
#include "blasfeo/include/blasfeo_common.h"

// acados
#include "acados/ocp_nlp/ocp_nlp_dynamics_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/types.h"
#include "acados_c/sim_interface.h"



/************************************************
 * dims
 ************************************************/

typedef struct
{
    void *sim;
    int nx;   // number of states at the current stage
    int nz;   // number of algebraic states at the current stage
    int nu;   // number of inputs at the current stage
    int nx1;  // number of states at the next stage
    int nu1;  // number of inputes at the next stage
} ocp_nlp_dynamics_cont_dims;

//
acados_size_t ocp_nlp_dynamics_cont_dims_calculate_size(void *config);
//
void *ocp_nlp_dynamics_cont_dims_assign(void *config, void *raw_memory);
//
void ocp_nlp_dynamics_cont_dims_set(void *config_, void *dims_, const char *field, int* value);

/************************************************
 * options
 ************************************************/

typedef struct
{
    void *sim_solver;
    int compute_adj;
    int compute_hess;
} ocp_nlp_dynamics_cont_opts;

//
acados_size_t ocp_nlp_dynamics_cont_opts_calculate_size(void *config, void *dims);
//
void *ocp_nlp_dynamics_cont_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_nlp_dynamics_cont_opts_initialize_default(void *config, void *dims, void *opts);
//
void ocp_nlp_dynamics_cont_opts_update(void *config, void *dims, void *opts);
//
void ocp_nlp_dynamics_cont_opts_set(void *config, void *opts, const char *field, void* value);



/************************************************
 * memory
 ************************************************/

typedef struct
{
    struct blasfeo_dvec fun;
    struct blasfeo_dvec adj;
    struct blasfeo_dvec *ux;            // pointer to ux in nlp_out at current stage
    struct blasfeo_dvec *tmp_ux;        // pointer to ux in tmp_nlp_out at current stage
    struct blasfeo_dvec *ux1;           // pointer to ux in nlp_out at next stage
    struct blasfeo_dvec *tmp_ux1;       // pointer to ux in tmp_nlp_out at next stage
    struct blasfeo_dvec *pi;            // pointer to pi in nlp_out at current stage
    struct blasfeo_dvec *tmp_pi;        // pointer to pi in tmp_nlp_out at current stage
    struct blasfeo_dmat *BAbt;          // pointer to BAbt in qp_in
    struct blasfeo_dmat *RSQrq;         // pointer to RSQrq in qp_in
    struct blasfeo_dvec *z_alg;         // pointer to output z at t = 0
    bool *set_sim_guess;                 // indicate if initialization for integrator is set from outside
    struct blasfeo_dvec *sim_guess;     // initializations for integrator
    // struct blasfeo_dvec *z;             // pointer to (input) z in nlp_out at current stage
    struct blasfeo_dmat *dzduxt;        // pointer to dzdux transposed
    void *sim_solver;                   // sim solver memory
} ocp_nlp_dynamics_cont_memory;

//
acados_size_t ocp_nlp_dynamics_cont_memory_calculate_size(void *config, void *dims, void *opts);
//
void *ocp_nlp_dynamics_cont_memory_assign(void *config, void *dims, void *opts, void *raw_memory);
//
struct blasfeo_dvec *ocp_nlp_dynamics_cont_memory_get_fun_ptr(void *memory);
//
struct blasfeo_dvec *ocp_nlp_dynamics_cont_memory_get_adj_ptr(void *memory);
//
void ocp_nlp_dynamics_cont_memory_set_ux_ptr(struct blasfeo_dvec *ux, void *memory);
//
void ocp_nlp_dynamics_cont_memory_set_tmp_ux_ptr(struct blasfeo_dvec *tmp_ux, void *memory);
//
void ocp_nlp_dynamics_cont_memory_set_ux1_ptr(struct blasfeo_dvec *ux1, void *memory);
//
void ocp_nlp_dynamics_cont_memory_set_tmp_ux1_ptr(struct blasfeo_dvec *tmp_ux1, void *memory);
//
void ocp_nlp_dynamics_cont_memory_set_pi_ptr(struct blasfeo_dvec *pi, void *memory);
//
void ocp_nlp_dynamics_cont_memory_set_tmp_pi_ptr(struct blasfeo_dvec *tmp_pi, void *memory);
//
void ocp_nlp_dynamics_cont_memory_set_BAbt_ptr(struct blasfeo_dmat *BAbt, void *memory);



/************************************************
 * workspace
 ************************************************/

typedef struct
{
    struct blasfeo_dmat hess;
    sim_in *sim_in;
    sim_out *sim_out;
    void *sim_solver;  // sim solver workspace
} ocp_nlp_dynamics_cont_workspace;

acados_size_t ocp_nlp_dynamics_cont_workspace_calculate_size(void *config, void *dims, void *opts);



/************************************************
 * model
 ************************************************/

typedef struct
{
    void *sim_model;
    // double *state_transition; // TODO
    double T;  // simulation time
} ocp_nlp_dynamics_cont_model;

//
acados_size_t ocp_nlp_dynamics_cont_model_calculate_size(void *config, void *dims);
//
void *ocp_nlp_dynamics_cont_model_assign(void *config, void *dims, void *raw_memory);
//
void ocp_nlp_dynamics_cont_model_set(void *config_, void *dims_, void *model_, const char *field, void *value);



/************************************************
 * functions
 ************************************************/

//
void ocp_nlp_dynamics_cont_config_initialize_default(void *config);
//
void ocp_nlp_dynamics_cont_initialize(void *config_, void *dims, void *model_, void *opts, void *mem, void *work_);
//
void ocp_nlp_dynamics_cont_update_qp_matrices(void *config_, void *dims, void *model_, void *opts, void *mem, void *work_);
//
void ocp_nlp_dynamics_cont_compute_fun(void *config_, void *dims, void *model_, void *opts, void *mem, void *work_);
//
int ocp_nlp_dynamics_cont_precompute(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_DYNAMICS_CONT_H_
/// @}
/// @}
