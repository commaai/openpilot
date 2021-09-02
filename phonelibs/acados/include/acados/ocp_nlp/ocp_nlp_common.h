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



/// \defgroup ocp_nlp ocp_nlp
/// @{
/// @}

/// \defgroup ocp_nlp_solver ocp_nlp_solver
/// @{
/// @}

/// \ingroup ocp_nlp
/// @{

/// \ingroup ocp_nlp_solver
/// @{

/// \defgroup ocp_nlp_common ocp_nlp_common
/// @{

#ifndef ACADOS_OCP_NLP_OCP_NLP_COMMON_H_
#define ACADOS_OCP_NLP_OCP_NLP_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/ocp_nlp/ocp_nlp_constraints_common.h"
#include "acados/ocp_nlp/ocp_nlp_cost_common.h"
#include "acados/ocp_nlp/ocp_nlp_dynamics_common.h"
#include "acados/ocp_nlp/ocp_nlp_reg_common.h"
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_qp/ocp_qp_xcond_solver.h"
#include "acados/sim/sim_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/types.h"



/************************************************
 * config
 ************************************************/

typedef struct ocp_nlp_config
{
    int N;  // number of stages

    // solver-specific implementations of memory management functions
    acados_size_t (*opts_calculate_size)(void *config, void *dims);
    void *(*opts_assign)(void *config, void *dims, void *raw_memory);
    void (*opts_initialize_default)(void *config, void *dims, void *opts_);
    void (*opts_update)(void *config, void *dims, void *opts_);
    acados_size_t (*memory_calculate_size)(void *config, void *dims, void *opts_);
    void *(*memory_assign)(void *config, void *dims, void *opts_, void *raw_memory);
    acados_size_t (*workspace_calculate_size)(void *config, void *dims, void *opts_);
    void (*opts_set)(void *config_, void *opts_, const char *field, void* value);
    void (*opts_set_at_stage)(void *config_, void *opts_, size_t stage, const char *field, void* value);
    // evaluate solver // TODO rename into solve
    int (*evaluate)(void *config, void *dims, void *nlp_in, void *nlp_out, void *opts_, void *mem, void *work);
    void (*eval_param_sens)(void *config, void *dims, void *opts_, void *mem, void *work,
                            char *field, int stage, int index, void *sens_nlp_out);
    // prepare memory
    int (*precompute)(void *config, void *dims, void *nlp_in, void *nlp_out, void *opts_, void *mem, void *work);
    // initalize this struct with default values
    void (*config_initialize_default)(void *config);
    // general getter
    void (*get)(void *config_, void *dims, void *mem_, const char *field, void *return_value_);
    void (*opts_get)(void *config_, void *dims, void *opts_, const char *field, void *return_value_);
    void (*work_get)(void *config_, void *dims, void *work_, const char *field, void *return_value_);
    // config structs of submodules
    ocp_qp_xcond_solver_config *qp_solver; // TODO rename xcond_solver
    ocp_nlp_dynamics_config **dynamics;
    ocp_nlp_cost_config **cost;
    ocp_nlp_constraints_config **constraints;
    ocp_nlp_reg_config *regularize;

} ocp_nlp_config;

//
acados_size_t ocp_nlp_config_calculate_size(int N);
//
ocp_nlp_config *ocp_nlp_config_assign(int N, void *raw_memory);



/************************************************
 * dims
 ************************************************/

/// Structure to store dimensions/number of variables.
typedef struct ocp_nlp_dims
{
    void **cost;
    void **dynamics;
    void **constraints;
    ocp_qp_xcond_solver_dims *qp_solver;  // xcond solver instead ??
    ocp_nlp_reg_dims *regularize;

    int *nv;  // number of primal variables (states+controls+slacks)
    int *nx;  // number of differential states
    int *nu;  // number of inputs
    int *ni;  // number of two-sided inequality constraints: nb+ng+nh+ns
    int *nz;  // number of algebraic variables
    int *ns;  // number of slack variables
    int N;    // number of shooting nodes
} ocp_nlp_dims;

//
acados_size_t ocp_nlp_dims_calculate_size(void *config);
//
ocp_nlp_dims *ocp_nlp_dims_assign(void *config, void *raw_memory);

/// Sets the dimension of optimization variables
/// (states, constrols, algebraic variables, slack variables).
///
/// \param config_ The configuration struct.
/// \param dims_ The dimension struct.
/// \param field The type of optimization variables, either nx, nu, nz, or ns.
/// \param value_array Number of variables for each stage.
void ocp_nlp_dims_set_opt_vars(void *config_, void *dims_,
                               const char *field, const void* value_array);

/// Sets the dimensions of constraints functions for a stage
/// (bounds on states, bounds on controls, equality constraints,
/// inequality constraints).
///
/// \param config_ The configuration struct.
/// \param dims_ The dimension struct.
/// \param stage Stage number.
/// \param field The type of constraint/bound, either nbx, nbu, ng, or nh.
/// \param value_field Number of constraints/bounds for the given stage.
void ocp_nlp_dims_set_constraints(void *config_, void *dims_, int stage,
                                  const char *field, const void* value_field);

/// Sets the dimensions of the cost terms for a stage.
///
/// \param config_ The configuration struct.
/// \param dims_ The dimension struct.
/// \param stage Stage number.
/// \param field Type of cost term, can be eiter ny.
/// \param value_field Number of cost terms/residuals for the given stage.
void ocp_nlp_dims_set_cost(void *config_, void *dims_, int stage, const char *field,
                           const void* value_field);

/// Sets the dimensions of the dynamics for a stage.
///
/// \param config_ The configuration struct.
/// \param dims_ The dimension struct.
/// \param stage Stage number.
/// \param field TBD
/// \param value TBD
void ocp_nlp_dims_set_dynamics(void *config_, void *dims_, int stage, const char *field,
                               const void* value);

/************************************************
 * Inputs
 ************************************************/

/// Struct for storing the inputs of an OCP NLP solver
typedef struct ocp_nlp_in
{
    /// Length of sampling intervals/timesteps.
    double *Ts;

    /// Pointers to cost functions (TBC).
    void **cost;

    /// Pointers to dynamics functions (TBC).
    void **dynamics;

    /// Pointers to constraints functions (TBC).
    void **constraints;

} ocp_nlp_in;

//
acados_size_t ocp_nlp_in_calculate_size_self(int N);
//
acados_size_t ocp_nlp_in_calculate_size(ocp_nlp_config *config, ocp_nlp_dims *dims);
//
ocp_nlp_in *ocp_nlp_in_assign_self(int N, void *raw_memory);
//
ocp_nlp_in *ocp_nlp_in_assign(ocp_nlp_config *config, ocp_nlp_dims *dims, void *raw_memory);


/************************************************
 * out
 ************************************************/

typedef struct ocp_nlp_out
{
    struct blasfeo_dvec *ux;  // NOTE: this contains [u; x; s_l; s_u]! - rename to uxs?
    struct blasfeo_dvec *z;  // algebraic vairables
    struct blasfeo_dvec *pi;  // multipliers for dynamics
    struct blasfeo_dvec *lam;  // inequality mulitpliers
    struct blasfeo_dvec *t;  // slack variables corresponding to evaluation of all inequalities (at the solution)

    // NOTE: the inequalities are internally organized in the following order:
    // [ lbu lbx lg lh lphi ubu ubx ug uh uphi; lsbu lsbx lsg lsh lsphi usbu usbx usg ush usphi]

    int sqp_iter;
    int qp_iter;
    double inf_norm_res;
    double total_time;

} ocp_nlp_out;

//
acados_size_t ocp_nlp_out_calculate_size(ocp_nlp_config *config, ocp_nlp_dims *dims);
//
ocp_nlp_out *ocp_nlp_out_assign(ocp_nlp_config *config, ocp_nlp_dims *dims,
                                void *raw_memory);



/************************************************
 * options
 ************************************************/

/// Globalization types
typedef enum
{
    FIXED_STEP,
    MERIT_BACKTRACKING,
} ocp_nlp_globalization_t;

typedef struct ocp_nlp_opts
{
    ocp_qp_xcond_solver_opts *qp_solver_opts; // xcond solver opts instead ???
    void *regularize;
    void **dynamics;     // dynamics_opts
    void **cost;         // cost_opts
    void **constraints;  // constraints_opts
    double step_length;  // step length in case of FIXED_STEP
    double levenberg_marquardt;  // LM factor to be added to the hessian before regularization
    int reuse_workspace;
    int num_threads;

    // TODO: move to separate struct?
    ocp_nlp_globalization_t globalization;
    double alpha_min;
    double alpha_reduction;
} ocp_nlp_opts;

//
acados_size_t ocp_nlp_opts_calculate_size(void *config, void *dims);
//
void *ocp_nlp_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_nlp_opts_initialize_default(void *config, void *dims, void *opts);
//
void ocp_nlp_opts_update(void *config, void *dims, void *opts);
//
void ocp_nlp_opts_set(void *config_, void *opts_, const char *field, void* value);
//
void ocp_nlp_opts_set_at_stage(void *config, void *opts, int stage, const char *field, void *value);


/************************************************
 * residuals
 ************************************************/

typedef struct ocp_nlp_res
{
    struct blasfeo_dvec *res_stat;  // stationarity
    struct blasfeo_dvec *res_eq;  // dynamics
    struct blasfeo_dvec *res_ineq;  // inequality constraints
    struct blasfeo_dvec *res_comp;  // complementarity
    double inf_norm_res_stat;
    double inf_norm_res_eq;
    double inf_norm_res_ineq;
    double inf_norm_res_comp;
    acados_size_t memsize;
} ocp_nlp_res;

//
acados_size_t ocp_nlp_res_calculate_size(ocp_nlp_dims *dims);
//
ocp_nlp_res *ocp_nlp_res_assign(ocp_nlp_dims *dims, void *raw_memory);

/************************************************
 * memory
 ************************************************/

typedef struct ocp_nlp_memory
{
//    void *qp_solver_mem; // xcond solver mem instead ???
    ocp_qp_xcond_solver_memory *qp_solver_mem; // xcond solver mem instead ???
    void *regularize_mem;
    void **dynamics;     // dynamics memory
    void **cost;         // cost memory
    void **constraints;  // constraints memory

    // residuals
    ocp_nlp_res *nlp_res;

    // qp in & out
    ocp_qp_in *qp_in;
    ocp_qp_out *qp_out;
    // QP stuff not entering the qp_in struct
    struct blasfeo_dmat *dzduxt; // dzdux transposed
    struct blasfeo_dvec *z_alg; // z_alg, output algebraic variables

    struct blasfeo_dvec *cost_grad;
    struct blasfeo_dvec *ineq_fun;
    struct blasfeo_dvec *ineq_adj;
    struct blasfeo_dvec *dyn_fun;
    struct blasfeo_dvec *dyn_adj;

    double cost_value;

    bool *set_sim_guess; // indicate if there is new explicitly provided guess for integration variables
    struct blasfeo_dvec *sim_guess;

	int *sqp_iter; // pointer to iteration number

} ocp_nlp_memory;

//
acados_size_t ocp_nlp_memory_calculate_size(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_opts *opts);
//
ocp_nlp_memory *ocp_nlp_memory_assign(ocp_nlp_config *config, ocp_nlp_dims *dims,
                                      ocp_nlp_opts *opts, void *raw_memory);



/************************************************
 * workspace
 ************************************************/

typedef struct ocp_nlp_workspace
{

    void *qp_work;
    void **dynamics;     // dynamics_workspace
    void **cost;         // cost_workspace
    void **constraints;  // constraints_workspace

	ocp_nlp_out *tmp_nlp_out;
	ocp_nlp_out *weight_merit_fun;

} ocp_nlp_workspace;

//
acados_size_t ocp_nlp_workspace_calculate_size(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_opts *opts);
//
ocp_nlp_workspace *ocp_nlp_workspace_assign(ocp_nlp_config *config, ocp_nlp_dims *dims,
                                ocp_nlp_opts *opts, ocp_nlp_memory *mem, void *raw_memory);



/************************************************
 * function
 ************************************************/

//
void ocp_nlp_initialize_qp(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
            ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work);
//
void ocp_nlp_approximate_qp_matrices(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
             ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work);
//
void ocp_nlp_approximate_qp_vectors_sqp(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
                 ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work);
//
void ocp_nlp_embed_initial_value(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
                 ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work);
//
void ocp_nlp_update_variables_sqp(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
           ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work, double alpha);
//
double ocp_nlp_line_search(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
            ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work);
//
double ocp_nlp_evaluate_merit_fun(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
          ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work);
//
void ocp_nlp_initialize_t_slacks(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
            ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work);
//
void ocp_nlp_res_compute(ocp_nlp_dims *dims, ocp_nlp_in *in, ocp_nlp_out *out,
                         ocp_nlp_res *res, ocp_nlp_memory *mem);
//
void ocp_nlp_cost_compute(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in,
            ocp_nlp_out *out, ocp_nlp_opts *opts, ocp_nlp_memory *mem, ocp_nlp_workspace *work);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_COMMON_H_
/// @}
/// @}
/// @}
