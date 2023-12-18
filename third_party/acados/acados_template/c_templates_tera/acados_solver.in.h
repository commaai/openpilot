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

#ifndef ACADOS_SOLVER_{{ model.name }}_H_
#define ACADOS_SOLVER_{{ model.name }}_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define {{ model.name | upper }}_NX     {{ dims.nx }}
#define {{ model.name | upper }}_NZ     {{ dims.nz }}
#define {{ model.name | upper }}_NU     {{ dims.nu }}
#define {{ model.name | upper }}_NP     {{ dims.np }}
#define {{ model.name | upper }}_NBX    {{ dims.nbx }}
#define {{ model.name | upper }}_NBX0   {{ dims.nbx_0 }}
#define {{ model.name | upper }}_NBU    {{ dims.nbu }}
#define {{ model.name | upper }}_NSBX   {{ dims.nsbx }}
#define {{ model.name | upper }}_NSBU   {{ dims.nsbu }}
#define {{ model.name | upper }}_NSH    {{ dims.nsh }}
#define {{ model.name | upper }}_NSG    {{ dims.nsg }}
#define {{ model.name | upper }}_NSPHI  {{ dims.nsphi }}
#define {{ model.name | upper }}_NSHN   {{ dims.nsh_e }}
#define {{ model.name | upper }}_NSGN   {{ dims.nsg_e }}
#define {{ model.name | upper }}_NSPHIN {{ dims.nsphi_e }}
#define {{ model.name | upper }}_NSBXN  {{ dims.nsbx_e }}
#define {{ model.name | upper }}_NS     {{ dims.ns }}
#define {{ model.name | upper }}_NSN    {{ dims.ns_e }}
#define {{ model.name | upper }}_NG     {{ dims.ng }}
#define {{ model.name | upper }}_NBXN   {{ dims.nbx_e }}
#define {{ model.name | upper }}_NGN    {{ dims.ng_e }}
#define {{ model.name | upper }}_NY0    {{ dims.ny_0 }}
#define {{ model.name | upper }}_NY     {{ dims.ny }}
#define {{ model.name | upper }}_NYN    {{ dims.ny_e }}
#define {{ model.name | upper }}_N      {{ dims.N }}
#define {{ model.name | upper }}_NH     {{ dims.nh }}
#define {{ model.name | upper }}_NPHI   {{ dims.nphi }}
#define {{ model.name | upper }}_NHN    {{ dims.nh_e }}
#define {{ model.name | upper }}_NPHIN  {{ dims.nphi_e }}
#define {{ model.name | upper }}_NR     {{ dims.nr }}

#ifdef __cplusplus
extern "C" {
#endif

{%- if not solver_options.custom_update_filename %}
    {%- set custom_update_filename = "" %}
{% else %}
    {%- set custom_update_filename = solver_options.custom_update_filename %}
{%- endif %}

// ** capsule for solver data **
typedef struct {{ model.name }}_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics
{% if solver_options.integrator_type == "ERK" %}
    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;
{% if solver_options.hessian_approx == "EXACT" %}
    external_function_param_casadi *hess_vde_casadi;
{%- endif %}
{% elif solver_options.integrator_type == "IRK" %}
    external_function_param_{{ model.dyn_ext_fun_type }} *impl_dae_fun;
    external_function_param_{{ model.dyn_ext_fun_type }} *impl_dae_fun_jac_x_xdot_z;
    external_function_param_{{ model.dyn_ext_fun_type }} *impl_dae_jac_x_xdot_u_z;
{% if solver_options.hessian_approx == "EXACT" %}
    external_function_param_{{ model.dyn_ext_fun_type }} *impl_dae_hess;
{%- endif %}
{% elif solver_options.integrator_type == "LIFTED_IRK" %}
    external_function_param_{{ model.dyn_ext_fun_type }} *impl_dae_fun;
    external_function_param_{{ model.dyn_ext_fun_type }} *impl_dae_fun_jac_x_xdot_u;
{% elif solver_options.integrator_type == "GNSF" %}
    external_function_param_casadi *gnsf_phi_fun;
    external_function_param_casadi *gnsf_phi_fun_jac_y;
    external_function_param_casadi *gnsf_phi_jac_y_uhat;
    external_function_param_casadi *gnsf_f_lo_jac_x1_x1dot_u_z;
    external_function_param_casadi *gnsf_get_matrices_fun;
{% elif solver_options.integrator_type == "DISCRETE" %}
    external_function_param_{{ model.dyn_ext_fun_type }} *discr_dyn_phi_fun;
    external_function_param_{{ model.dyn_ext_fun_type }} *discr_dyn_phi_fun_jac_ut_xt;
{%- if solver_options.hessian_approx == "EXACT" %}
    external_function_param_{{ model.dyn_ext_fun_type }} *discr_dyn_phi_fun_jac_ut_xt_hess;
{%- endif %}
{%- endif %}


    // cost
{% if cost.cost_type == "NONLINEAR_LS" %}
    external_function_param_casadi *cost_y_fun;
    external_function_param_casadi *cost_y_fun_jac_ut_xt;
    external_function_param_casadi *cost_y_hess;
{% elif cost.cost_type == "CONVEX_OVER_NONLINEAR" %}
    external_function_param_casadi *conl_cost_fun;
    external_function_param_casadi *conl_cost_fun_jac_hess;
{%- elif cost.cost_type == "EXTERNAL" %}
    external_function_param_{{ cost.cost_ext_fun_type }} *ext_cost_fun;
    external_function_param_{{ cost.cost_ext_fun_type }} *ext_cost_fun_jac;
    external_function_param_{{ cost.cost_ext_fun_type }} *ext_cost_fun_jac_hess;
{% endif %}

{% if cost.cost_type_0 == "NONLINEAR_LS" %}
    external_function_param_casadi cost_y_0_fun;
    external_function_param_casadi cost_y_0_fun_jac_ut_xt;
    external_function_param_casadi cost_y_0_hess;
{% elif cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}
    external_function_param_casadi conl_cost_0_fun;
    external_function_param_casadi conl_cost_0_fun_jac_hess;
{% elif cost.cost_type_0 == "EXTERNAL" %}
    external_function_param_{{ cost.cost_ext_fun_type_0 }} ext_cost_0_fun;
    external_function_param_{{ cost.cost_ext_fun_type_0 }} ext_cost_0_fun_jac;
    external_function_param_{{ cost.cost_ext_fun_type_0 }} ext_cost_0_fun_jac_hess;
{%- endif %}

{% if cost.cost_type_e == "NONLINEAR_LS" %}
    external_function_param_casadi cost_y_e_fun;
    external_function_param_casadi cost_y_e_fun_jac_ut_xt;
    external_function_param_casadi cost_y_e_hess;
{% elif cost.cost_type_e == "CONVEX_OVER_NONLINEAR" %}
    external_function_param_casadi conl_cost_e_fun;
    external_function_param_casadi conl_cost_e_fun_jac_hess;
{% elif cost.cost_type_e == "EXTERNAL" %}
    external_function_param_{{ cost.cost_ext_fun_type_e }} ext_cost_e_fun;
    external_function_param_{{ cost.cost_ext_fun_type_e }} ext_cost_e_fun_jac;
    external_function_param_{{ cost.cost_ext_fun_type_e }} ext_cost_e_fun_jac_hess;
{%- endif %}

    // constraints
{%- if constraints.constr_type == "BGP" %}
    external_function_param_casadi *phi_constraint;
{% elif constraints.constr_type == "BGH" and dims.nh > 0 %}
    external_function_param_casadi *nl_constr_h_fun_jac;
    external_function_param_casadi *nl_constr_h_fun;
{%- if solver_options.hessian_approx == "EXACT" %}
    external_function_param_casadi *nl_constr_h_fun_jac_hess;
{%- endif %}
{%- endif %}


{% if constraints.constr_type_e == "BGP" %}
    external_function_param_casadi phi_e_constraint;
{% elif constraints.constr_type_e == "BGH" and dims.nh_e > 0 %}
    external_function_param_casadi nl_constr_h_e_fun_jac;
    external_function_param_casadi nl_constr_h_e_fun;
{%- if solver_options.hessian_approx == "EXACT" %}
    external_function_param_casadi nl_constr_h_e_fun_jac_hess;
{%- endif %}
{%- endif %}

{%- if custom_update_filename != "" %}
    void * custom_update_memory;
{%- endif %}

} {{ model.name }}_solver_capsule;

ACADOS_SYMBOL_EXPORT {{ model.name }}_solver_capsule * {{ model.name }}_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_free_capsule({{ model.name }}_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_create({{ model.name }}_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_reset({{ model.name }}_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of {{ model.name }}_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_create_with_discretization({{ model.name }}_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_update_time_steps({{ model.name }}_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_update_qp_solver_cond_N({{ model.name }}_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_update_params({{ model.name }}_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_update_params_sparse({{ model.name }}_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_solve({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_free({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void {{ model.name }}_acados_print_stats({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_custom_update({{ model.name }}_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *{{ model.name }}_acados_get_nlp_in({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *{{ model.name }}_acados_get_nlp_out({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *{{ model.name }}_acados_get_sens_out({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *{{ model.name }}_acados_get_nlp_solver({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *{{ model.name }}_acados_get_nlp_config({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *{{ model.name }}_acados_get_nlp_opts({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *{{ model.name }}_acados_get_nlp_dims({{ model.name }}_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *{{ model.name }}_acados_get_nlp_plan({{ model.name }}_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_{{ model.name }}_H_
