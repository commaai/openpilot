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

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "{{ model.name }}_model/{{ model.name }}_model.h"
#include "{{ model.name }}_constraints/{{ model.name }}_constraints.h"

{%- if cost.cost_type != "LINEAR_LS" or cost.cost_type_e != "LINEAR_LS" or cost.cost_type_0 != "LINEAR_LS" %}
#include "{{ model.name }}_cost/{{ model.name }}_cost.h"
{%- endif %}

{%- if not solver_options.custom_update_filename %}
    {%- set custom_update_filename = "" %}
{% else %}
    {%- set custom_update_filename = solver_options.custom_update_filename %}
{%- endif %}
{%- if not solver_options.custom_update_header_filename %}
    {%- set custom_update_header_filename = "" %}
{% else %}
    {%- set custom_update_header_filename = solver_options.custom_update_header_filename %}
{%- endif %}
{%- if custom_update_header_filename != "" %}
#include "{{ custom_update_header_filename }}"
{%- endif %}


#include "acados_solver_{{ model.name }}.h"

#define NX     {{ model.name | upper }}_NX
#define NZ     {{ model.name | upper }}_NZ
#define NU     {{ model.name | upper }}_NU
#define NP     {{ model.name | upper }}_NP
#define NBX    {{ model.name | upper }}_NBX
#define NBX0   {{ model.name | upper }}_NBX0
#define NBU    {{ model.name | upper }}_NBU
#define NSBX   {{ model.name | upper }}_NSBX
#define NSBU   {{ model.name | upper }}_NSBU
#define NSH    {{ model.name | upper }}_NSH
#define NSG    {{ model.name | upper }}_NSG
#define NSPHI  {{ model.name | upper }}_NSPHI
#define NSHN   {{ model.name | upper }}_NSHN
#define NSGN   {{ model.name | upper }}_NSGN
#define NSPHIN {{ model.name | upper }}_NSPHIN
#define NSBXN  {{ model.name | upper }}_NSBXN
#define NS     {{ model.name | upper }}_NS
#define NSN    {{ model.name | upper }}_NSN
#define NG     {{ model.name | upper }}_NG
#define NBXN   {{ model.name | upper }}_NBXN
#define NGN    {{ model.name | upper }}_NGN
#define NY0    {{ model.name | upper }}_NY0
#define NY     {{ model.name | upper }}_NY
#define NYN    {{ model.name | upper }}_NYN
// #define N      {{ model.name | upper }}_N
#define NH     {{ model.name | upper }}_NH
#define NPHI   {{ model.name | upper }}_NPHI
#define NHN    {{ model.name | upper }}_NHN
#define NPHIN  {{ model.name | upper }}_NPHIN
#define NR     {{ model.name | upper }}_NR


// ** solver data **

{{ model.name }}_solver_capsule * {{ model.name }}_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof({{ model.name }}_solver_capsule));
    {{ model.name }}_solver_capsule *capsule = ({{ model.name }}_solver_capsule *) capsule_mem;

    return capsule;
}


int {{ model.name }}_acados_free_capsule({{ model.name }}_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int {{ model.name }}_acados_create({{ model.name }}_solver_capsule* capsule)
{
    int N_shooting_intervals = {{ model.name | upper }}_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return {{ model.name }}_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int {{ model.name }}_acados_update_time_steps({{ model.name }}_solver_capsule* capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "{{ model.name }}_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;
}

/**
 * Internal function for {{ model.name }}_acados_create: step 1
 */
void {{ model.name }}_acados_create_1_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    {%- if solver_options.nlp_solver_type == "SQP" %}
    nlp_solver_plan->nlp_solver = SQP;
    {%- else %}
    nlp_solver_plan->nlp_solver = SQP_RTI;
    {%- endif %}

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = {{ solver_options.qp_solver }};

    nlp_solver_plan->nlp_cost[0] = {{ cost.cost_type_0 }};
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = {{ cost.cost_type }};

    nlp_solver_plan->nlp_cost[N] = {{ cost.cost_type_e }};

    for (int i = 0; i < N; i++)
    {
        {%- if solver_options.integrator_type == "DISCRETE" %}
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
        {%- else %}
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = {{ solver_options.integrator_type }};
        {%- endif %}
    }

    for (int i = 0; i < N; i++)
    {
        {%- if constraints.constr_type == "BGP" %}
        nlp_solver_plan->nlp_constraints[i] = BGP;
        {%- else -%}
        nlp_solver_plan->nlp_constraints[i] = BGH;
        {%- endif %}
    }

    {%- if constraints.constr_type_e == "BGP" %}
    nlp_solver_plan->nlp_constraints[N] = BGP;
    {%- else %}
    nlp_solver_plan->nlp_constraints[N] = BGH;
    {%- endif %}

    {%- if solver_options.regularize_method == "NO_REGULARIZE" %}
    nlp_solver_plan->regularization = NO_REGULARIZE;
    {%- elif solver_options.regularize_method == "MIRROR" %}
    nlp_solver_plan->regularization = MIRROR;
    {%- elif solver_options.regularize_method == "PROJECT" %}
    nlp_solver_plan->regularization = PROJECT;
    {%- elif solver_options.regularize_method == "PROJECT_REDUC_HESS" %}
    nlp_solver_plan->regularization = PROJECT_REDUC_HESS;
    {%- elif solver_options.regularize_method == "CONVEXIFY" %}
    nlp_solver_plan->regularization = CONVEXIFY;
    {%- endif %}
}


/**
 * Internal function for {{ model.name }}_acados_create: step 2
 */
ocp_nlp_dims* {{ model.name }}_acados_create_2_create_and_set_dimensions({{ model.name }}_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 17
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
    }

    // for initial state
    nbx[0]  = NBX0;
    nsbx[0] = 0;
    ns[0] = NS - NSBX;
    nbxe[0] = {{ dims.nbxe_0 }};
    ny[0] = NY0;

    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = {{ dims.nr_e }};

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }

{%- if cost.cost_type_0 == "NONLINEAR_LS" or cost.cost_type_0 == "LINEAR_LS" or cost.cost_type_0 == "CONVEX_OVER_NONLINEAR"%}
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
{%- endif %}

{%- if cost.cost_type == "NONLINEAR_LS" or cost.cost_type == "LINEAR_LS" or cost.cost_type == "CONVEX_OVER_NONLINEAR"%}
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
{%- endif %}

    for (int i = 0; i < N; i++)
    {
        {%- if constraints.constr_type == "BGH" and dims.nh > 0 %}
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
        {%- elif constraints.constr_type == "BGP" and dims.nphi > 0 %}
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nr", &nr[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nphi", &nphi[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsphi", &nsphi[i]);
        {%- endif %}
    }

{%- if constraints.constr_type_e == "BGH" %}
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
{%- elif constraints.constr_type_e == "BGP" %}
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nr", &nr[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nphi", &nphi[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsphi", &nsphi[N]);
{%- endif %}
{%- if cost.cost_type_e == "NONLINEAR_LS" or cost.cost_type_e == "LINEAR_LS" or cost.cost_type_e == "CONVEX_OVER_NONLINEAR"%}
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
{%- endif %}
    free(intNp1mem);

{%- if solver_options.integrator_type == "GNSF" -%}
    // GNSF specific dimensions
    int gnsf_nx1 = {{ dims.gnsf_nx1 }};
    int gnsf_nz1 = {{ dims.gnsf_nz1 }};
    int gnsf_nout = {{ dims.gnsf_nout }};
    int gnsf_ny = {{ dims.gnsf_ny }};
    int gnsf_nuhat = {{ dims.gnsf_nuhat }};

    for (int i = 0; i < N; i++)
    {
        if (nlp_solver_plan->sim_solver_plan[i].sim_solver == GNSF)
        {
            ocp_nlp_dims_set_dynamics(nlp_config, nlp_dims, i, "gnsf_nx1", &gnsf_nx1);
            ocp_nlp_dims_set_dynamics(nlp_config, nlp_dims, i, "gnsf_nz1", &gnsf_nz1);
            ocp_nlp_dims_set_dynamics(nlp_config, nlp_dims, i, "gnsf_nout", &gnsf_nout);
            ocp_nlp_dims_set_dynamics(nlp_config, nlp_dims, i, "gnsf_ny", &gnsf_ny);
            ocp_nlp_dims_set_dynamics(nlp_config, nlp_dims, i, "gnsf_nuhat", &gnsf_nuhat);
        }
    }
{%- endif %}
return nlp_dims;
}


/**
 * Internal function for {{ model.name }}_acados_create: step 3
 */
void {{ model.name }}_acados_create_3_create_and_set_functions({{ model.name }}_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_param_casadi_create(&capsule->__CAPSULE_FNC__ , {{ dims.np }}); \
    }while(false)

{% if constraints.constr_type == "BGP" %}
    // constraints.constr_type == "BGP"
    capsule->phi_constraint = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        // nonlinear part of convex-composite constraint
        MAP_CASADI_FNC(phi_constraint[i], {{ model.name }}_phi_constraint);
    }
{%- endif %}

{%- if constraints.constr_type_e == "BGP" %}
    // constraints.constr_type_e == "BGP"
    // nonlinear part of convex-composite constraint
    MAP_CASADI_FNC(phi_e_constraint, {{ model.name }}_phi_e_constraint);
{%- endif %}

{%- if constraints.constr_type == "BGH" and dims.nh > 0  %}
    // constraints.constr_type == "BGH" and dims.nh > 0
    capsule->nl_constr_h_fun_jac = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun_jac[i], {{ model.name }}_constr_h_fun_jac_uxt_zt);
    }
    capsule->nl_constr_h_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun[i], {{ model.name }}_constr_h_fun);
    }
    {% if solver_options.hessian_approx == "EXACT" %}
    capsule->nl_constr_h_fun_jac_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun_jac_hess[i], {{ model.name }}_constr_h_fun_jac_uxt_zt_hess);
    }
    {% endif %}
{% endif %}

{%- if constraints.constr_type_e == "BGH" and dims.nh_e > 0 %}
    MAP_CASADI_FNC(nl_constr_h_e_fun_jac, {{ model.name }}_constr_h_e_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_e_fun, {{ model.name }}_constr_h_e_fun);

    {%- if solver_options.hessian_approx == "EXACT" %}
    MAP_CASADI_FNC(nl_constr_h_e_fun_jac_hess, {{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess);
    {% endif %}
{%- endif %}

{% if solver_options.integrator_type == "ERK" %}
    // explicit ode
    capsule->forw_vde_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(forw_vde_casadi[i], {{ model.name }}_expl_vde_forw);
    }

    capsule->expl_ode_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(expl_ode_fun[i], {{ model.name }}_expl_ode_fun);
    }

    {%- if solver_options.hessian_approx == "EXACT" %}
    capsule->hess_vde_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(hess_vde_casadi[i], {{ model.name }}_expl_ode_hess);
    }
    {%- endif %}

{% elif solver_options.integrator_type == "IRK" %}
    // implicit dae
    capsule->impl_dae_fun = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++) {
    {%- if model.dyn_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(impl_dae_fun[i], {{ model.name }}_impl_dae_fun);
    {%- else %}
        capsule->impl_dae_fun[i].fun = &{{ model.dyn_impl_dae_fun }};
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->impl_dae_fun[i], {{ dims.np }});
    {%- endif %}
    }

    capsule->impl_dae_fun_jac_x_xdot_z = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++) {
    {%- if model.dyn_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(impl_dae_fun_jac_x_xdot_z[i], {{ model.name }}_impl_dae_fun_jac_x_xdot_z);
    {%- else %}
        capsule->impl_dae_fun_jac_x_xdot_z[i].fun = &{{ model.dyn_impl_dae_fun_jac }};
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->impl_dae_fun_jac_x_xdot_z[i], {{ dims.np }});
    {%- endif %}
    }

    capsule->impl_dae_jac_x_xdot_u_z = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++) {
    {%- if model.dyn_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(impl_dae_jac_x_xdot_u_z[i], {{ model.name }}_impl_dae_jac_x_xdot_u_z);
    {%- else %}
        capsule->impl_dae_jac_x_xdot_u_z[i].fun = &{{ model.dyn_impl_dae_jac }};
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->impl_dae_jac_x_xdot_u_z[i], {{ dims.np }});
    {%- endif %}
    }

    {%- if solver_options.hessian_approx == "EXACT" %}
    capsule->impl_dae_hess = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_hess[i], {{ model.name }}_impl_dae_hess);
    }
    {%- endif %}
{% elif solver_options.integrator_type == "LIFTED_IRK" %}
    // external functions (implicit model)
    capsule->impl_dae_fun = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_fun[i], {{ model.name }}_impl_dae_fun);
    }

    capsule->impl_dae_fun_jac_x_xdot_u = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_fun_jac_x_xdot_u[i], {{ model.name }}_impl_dae_fun_jac_x_xdot_u);
    }

{% elif solver_options.integrator_type == "GNSF" %}
    {% if model.gnsf.purely_linear != 1 %}
    capsule->gnsf_phi_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(gnsf_phi_fun[i], {{ model.name }}_gnsf_phi_fun);
    }

    capsule->gnsf_phi_fun_jac_y = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(gnsf_phi_fun_jac_y[i], {{ model.name }}_gnsf_phi_fun_jac_y);
    }

    capsule->gnsf_phi_jac_y_uhat = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(gnsf_phi_jac_y_uhat[i], {{ model.name }}_gnsf_phi_jac_y_uhat);
    }

    {% if model.gnsf.nontrivial_f_LO == 1 %}
    capsule->gnsf_f_lo_jac_x1_x1dot_u_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(gnsf_f_lo_jac_x1_x1dot_u_z[i], {{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz);
    }
    {%- endif %}
    {%- endif %}
    capsule->gnsf_get_matrices_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(gnsf_get_matrices_fun[i], {{ model.name }}_gnsf_get_matrices_fun);
    }
{% elif solver_options.integrator_type == "DISCRETE" %}
    // discrete dynamics
    capsule->discr_dyn_phi_fun = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++)
    {
        {%- if model.dyn_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(discr_dyn_phi_fun[i], {{ model.name }}_dyn_disc_phi_fun);
        {%- else %}
        capsule->discr_dyn_phi_fun[i].fun = &{{ model.dyn_disc_fun }};
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->discr_dyn_phi_fun[i], {{ dims.np }});
        {%- endif %}
    }

    capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++)
    {
        {%- if model.dyn_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], {{ model.name }}_dyn_disc_phi_fun_jac);
        {%- else %}
        capsule->discr_dyn_phi_fun_jac_ut_xt[i].fun = &{{ model.dyn_disc_fun_jac }};
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->discr_dyn_phi_fun_jac_ut_xt[i], {{ dims.np }});
        {%- endif %}
    }

  {%- if solver_options.hessian_approx == "EXACT" %}
    capsule->discr_dyn_phi_fun_jac_ut_xt_hess = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++)
    {
        {%- if model.dyn_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt_hess[i], {{ model.name }}_dyn_disc_phi_fun_jac_hess);
        {%- else %}
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i].fun = &{{ model.dyn_disc_fun_jac_hess }};
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i], {{ dims.np }});
        {%- endif %}
    }
  {%- endif %}
{%- endif %}


{%- if cost.cost_type_0 == "NONLINEAR_LS" %}
    // nonlinear least squares function
    MAP_CASADI_FNC(cost_y_0_fun, {{ model.name }}_cost_y_0_fun);
    MAP_CASADI_FNC(cost_y_0_fun_jac_ut_xt, {{ model.name }}_cost_y_0_fun_jac_ut_xt);
    MAP_CASADI_FNC(cost_y_0_hess, {{ model.name }}_cost_y_0_hess);

{%- elif cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}
    // convex-over-nonlinear cost
    MAP_CASADI_FNC(conl_cost_0_fun, {{ model.name }}_conl_cost_0_fun);
    MAP_CASADI_FNC(conl_cost_0_fun_jac_hess, {{ model.name }}_conl_cost_0_fun_jac_hess);

{%- elif cost.cost_type_0 == "EXTERNAL" %}
    // external cost
    {%- if cost.cost_ext_fun_type_0 == "casadi" %}
    MAP_CASADI_FNC(ext_cost_0_fun, {{ model.name }}_cost_ext_cost_0_fun);
    {%- else %}
    capsule->ext_cost_0_fun.fun = &{{ cost.cost_function_ext_cost_0 }};
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_create(&capsule->ext_cost_0_fun, {{ dims.np }});
    {%- endif %}

    // external cost
    {%- if cost.cost_ext_fun_type_0 == "casadi" %}
    MAP_CASADI_FNC(ext_cost_0_fun_jac, {{ model.name }}_cost_ext_cost_0_fun_jac);
    {%- else %}
    capsule->ext_cost_0_fun_jac.fun = &{{ cost.cost_function_ext_cost_0 }};
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_create(&capsule->ext_cost_0_fun_jac, {{ dims.np }});
    {%- endif %}

    // external cost
    {%- if cost.cost_ext_fun_type_0 == "casadi" %}
    MAP_CASADI_FNC(ext_cost_0_fun_jac_hess, {{ model.name }}_cost_ext_cost_0_fun_jac_hess);
    {%- else %}
    capsule->ext_cost_0_fun_jac_hess.fun = &{{ cost.cost_function_ext_cost_0 }};
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_create(&capsule->ext_cost_0_fun_jac_hess, {{ dims.np }});
    {%- endif %}
{%- endif %}

{%- if cost.cost_type == "NONLINEAR_LS" %}
    // nonlinear least squares cost
    capsule->cost_y_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(cost_y_fun[i], {{ model.name }}_cost_y_fun);
    }

    capsule->cost_y_fun_jac_ut_xt = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(cost_y_fun_jac_ut_xt[i], {{ model.name }}_cost_y_fun_jac_ut_xt);
    }

    capsule->cost_y_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(cost_y_hess[i], {{ model.name }}_cost_y_hess);
    }

{%- elif cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}
    // convex-over-nonlinear cost
    capsule->conl_cost_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(conl_cost_fun[i], {{ model.name }}_conl_cost_fun);
    }
    capsule->conl_cost_fun_jac_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(conl_cost_fun_jac_hess[i], {{ model.name }}_conl_cost_fun_jac_hess);
    }

{%- elif cost.cost_type == "EXTERNAL" %}
    // external cost
    capsule->ext_cost_fun = (external_function_param_{{ cost.cost_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ cost.cost_ext_fun_type }})*N);
    for (int i = 0; i < N-1; i++)
    {
        {%- if cost.cost_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(ext_cost_fun[i], {{ model.name }}_cost_ext_cost_fun);
        {%- else %}
        capsule->ext_cost_fun[i].fun = &{{ cost.cost_function_ext_cost }};
        external_function_param_{{ cost.cost_ext_fun_type }}_create(&capsule->ext_cost_fun[i], {{ dims.np }});
        {%- endif %}
    }

    capsule->ext_cost_fun_jac = (external_function_param_{{ cost.cost_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ cost.cost_ext_fun_type }})*N);
    for (int i = 0; i < N-1; i++)
    {
        {%- if cost.cost_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(ext_cost_fun_jac[i], {{ model.name }}_cost_ext_cost_fun_jac);
        {%- else %}
        capsule->ext_cost_fun_jac[i].fun = &{{ cost.cost_function_ext_cost }};
        external_function_param_{{ cost.cost_ext_fun_type }}_create(&capsule->ext_cost_fun_jac[i], {{ dims.np }});
        {%- endif %}
    }

    capsule->ext_cost_fun_jac_hess = (external_function_param_{{ cost.cost_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ cost.cost_ext_fun_type }})*N);
    for (int i = 0; i < N-1; i++)
    {
        {%- if cost.cost_ext_fun_type == "casadi" %}
        MAP_CASADI_FNC(ext_cost_fun_jac_hess[i], {{ model.name }}_cost_ext_cost_fun_jac_hess);
        {%- else %}
        capsule->ext_cost_fun_jac_hess[i].fun = &{{ cost.cost_function_ext_cost }};
        external_function_param_{{ cost.cost_ext_fun_type }}_create(&capsule->ext_cost_fun_jac_hess[i], {{ dims.np }});
        {%- endif %}
    }
{%- endif %}

{%- if cost.cost_type_e == "NONLINEAR_LS" %}
    // nonlinear least square function
    MAP_CASADI_FNC(cost_y_e_fun, {{ model.name }}_cost_y_e_fun);
    MAP_CASADI_FNC(cost_y_e_fun_jac_ut_xt, {{ model.name }}_cost_y_e_fun_jac_ut_xt);
    MAP_CASADI_FNC(cost_y_e_hess, {{ model.name }}_cost_y_e_hess);

{%- elif cost.cost_type_e == "CONVEX_OVER_NONLINEAR" %}
    // convex-over-nonlinear cost
    MAP_CASADI_FNC(conl_cost_e_fun, {{ model.name }}_conl_cost_e_fun);
    MAP_CASADI_FNC(conl_cost_e_fun_jac_hess, {{ model.name }}_conl_cost_e_fun_jac_hess);

{%- elif cost.cost_type_e == "EXTERNAL" %}
    // external cost - function
    {%- if cost.cost_ext_fun_type_e == "casadi" %}
    MAP_CASADI_FNC(ext_cost_e_fun, {{ model.name }}_cost_ext_cost_e_fun);
    {% else %}
    capsule->ext_cost_e_fun.fun = &{{ cost.cost_function_ext_cost_e }};
    external_function_param_{{ cost.cost_ext_fun_type_e }}_create(&capsule->ext_cost_e_fun, {{ dims.np }});
    {%- endif %}

    // external cost - jacobian
    {%- if cost.cost_ext_fun_type_e == "casadi" %}
    MAP_CASADI_FNC(ext_cost_e_fun_jac, {{ model.name }}_cost_ext_cost_e_fun_jac);
    {%- else %}
    capsule->ext_cost_e_fun_jac.fun = &{{ cost.cost_function_ext_cost_e }};
    external_function_param_{{ cost.cost_ext_fun_type_e }}_create(&capsule->ext_cost_e_fun_jac, {{ dims.np }});
    {%- endif %}

    // external cost - hessian
    {%- if cost.cost_ext_fun_type_e == "casadi" %}
    MAP_CASADI_FNC(ext_cost_e_fun_jac_hess, {{ model.name }}_cost_ext_cost_e_fun_jac_hess);
    {%- else %}
    capsule->ext_cost_e_fun_jac_hess.fun = &{{ cost.cost_function_ext_cost_e }};
    external_function_param_{{ cost.cost_ext_fun_type_e }}_create(&capsule->ext_cost_e_fun_jac_hess, {{ dims.np }});
    {%- endif %}
{%- endif %}

#undef MAP_CASADI_FNC
}


/**
 * Internal function for {{ model.name }}_acados_create: step 4
 */
void {{ model.name }}_acados_create_4_set_default_parameters({{ model.name }}_solver_capsule* capsule) {
{%- if dims.np > 0 %}
    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));
    {%- for item in parameter_values %}
        {%- if item != 0 %}
    p[{{ loop.index0 }}] = {{ item }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i <= N; i++) {
        {{ model.name }}_acados_update_params(capsule, i, p, NP);
    }
    free(p);
{%- else %}
    // no parameters defined
{%- endif %}{# if dims.np #}
}


/**
 * Internal function for {{ model.name }}_acados_create: step 5
 */
void {{ model.name }}_acados_create_5_set_nlp_in({{ model.name }}_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    /************************************************
    *  nlp_in
    ************************************************/
//    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
//    capsule->nlp_in = nlp_in;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    // set up time_steps
    {% set all_equal = true -%}
    {%- set val = solver_options.time_steps[0] %}
    {%- for j in range(start=1, end=dims.N) %}
        {%- if val != solver_options.time_steps[j] %}
            {%- set_global all_equal = false %}
            {%- break %}
        {%- endif %}
    {%- endfor %}

    if (new_time_steps) {
        {{ model.name }}_acados_update_time_steps(capsule, N, new_time_steps);
    } else {
    {%- if all_equal == true -%}
        // all time_steps are identical
        double time_step = {{ solver_options.time_steps[0] }};
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &time_step);
        }
    {%- else -%}
        // time_steps are different
        double* time_steps = malloc(N*sizeof(double));
        {%- for j in range(end=dims.N) %}
        time_steps[{{ j }}] = {{ solver_options.time_steps[j] }};
        {%- endfor %}
        {{ model.name }}_acados_update_time_steps(capsule, N, time_steps);
        free(time_steps);
    {%- endif %}
    }

    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
    {%- if solver_options.integrator_type == "ERK" %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "expl_vde_forw", &capsule->forw_vde_casadi[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "expl_ode_fun", &capsule->expl_ode_fun[i]);
        {%- if solver_options.hessian_approx == "EXACT" %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "expl_ode_hess", &capsule->hess_vde_casadi[i]);
        {%- endif %}
    {% elif solver_options.integrator_type == "IRK" %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "impl_dae_fun", &capsule->impl_dae_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_fun_jac_x_xdot_z", &capsule->impl_dae_fun_jac_x_xdot_z[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_jac_x_xdot_u", &capsule->impl_dae_jac_x_xdot_u_z[i]);
        {%- if solver_options.hessian_approx == "EXACT" %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "impl_dae_hess", &capsule->impl_dae_hess[i]);
        {%- endif %}
    {% elif solver_options.integrator_type == "LIFTED_IRK" %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "impl_dae_fun", &capsule->impl_dae_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_fun_jac_x_xdot_u", &capsule->impl_dae_fun_jac_x_xdot_u[i]);
    {% elif solver_options.integrator_type == "GNSF" %}
        {% if model.gnsf.purely_linear != 1 %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "phi_fun", &capsule->gnsf_phi_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "phi_fun_jac_y", &capsule->gnsf_phi_fun_jac_y[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "phi_jac_y_uhat", &capsule->gnsf_phi_jac_y_uhat[i]);
            {% if model.gnsf.nontrivial_f_LO == 1 %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "f_lo_jac_x1_x1dot_u_z",
                                   &capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i]);
            {%- endif %}
        {%- endif %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "gnsf_get_matrices_fun",
                                   &capsule->gnsf_get_matrices_fun[i]);
    {% elif solver_options.integrator_type == "DISCRETE" %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        {%- if solver_options.hessian_approx == "EXACT" %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac_hess",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i]);
        {%- endif %}
    {%- endif %}
    }

    /**** Cost ****/

{%- if dims.ny_0 == 0 %}
{%- elif cost.cost_type_0 == "NONLINEAR_LS" or cost.cost_type_0 == "LINEAR_LS" or cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny_0) %}
        {%- if cost.yref_0[j] != 0 %}
    yref_0[{{ j }}] = {{ cost.yref_0[j] }};
        {%- endif %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);
{%- endif %}


{%- if dims.ny == 0 %}
{%- elif cost.cost_type == "NONLINEAR_LS" or cost.cost_type == "LINEAR_LS" or cost.cost_type == "CONVEX_OVER_NONLINEAR" %}
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny) %}
        {%- if cost.yref[j] != 0 %}
    yref[{{ j }}] = {{ cost.yref[j] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
{%- endif %}


{%- if dims.ny_e == 0 %}
{%- elif cost.cost_type_e == "NONLINEAR_LS" or cost.cost_type_e == "LINEAR_LS" or cost.cost_type_e == "CONVEX_OVER_NONLINEAR" %}
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny_e) %}
        {%- if cost.yref_e[j] != 0 %}
    yref_e[{{ j }}] = {{ cost.yref_e[j] }};
        {%- endif %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);
{%- endif %}

{%- if dims.ny_0 == 0 %}
{%- elif cost.cost_type_0 == "NONLINEAR_LS" or cost.cost_type_0 == "LINEAR_LS" %}
   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny_0) %}
        {%- for k in range(end=dims.ny_0) %}
            {%- if cost.W_0[j][k] != 0 %}
    W_0[{{ j }}+(NY0) * {{ k }}] = {{ cost.W_0[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
{%- endif %}

{%- if dims.ny == 0 %}
{%- elif cost.cost_type == "NONLINEAR_LS" or cost.cost_type == "LINEAR_LS" %}
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny) %}
        {%- for k in range(end=dims.ny) %}
            {%- if cost.W[j][k] != 0 %}
    W[{{ j }}+(NY) * {{ k }}] = {{ cost.W[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
{%- endif %}

{%- if dims.ny_e == 0 %}
{%- elif cost.cost_type_e == "NONLINEAR_LS" or cost.cost_type_e == "LINEAR_LS" %}
    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny_e) %}
        {%- for k in range(end=dims.ny_e) %}
            {%- if cost.W_e[j][k] != 0 %}
    W_e[{{ j }}+(NYN) * {{ k }}] = {{ cost.W_e[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
{%- endif %}

{%- if dims.ny_0 == 0 %}
{%- elif cost.cost_type_0 == "LINEAR_LS" %}
    double* Vx_0 = calloc(NY0*NX, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny_0) %}
        {%- for k in range(end=dims.nx) %}
            {%- if cost.Vx_0[j][k] != 0 %}
    Vx_0[{{ j }}+(NY0) * {{ k }}] = {{ cost.Vx_0[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);

    {%- if dims.nu > 0 %}
    double* Vu_0 = calloc(NY0*NU, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny_0) %}
        {%- for k in range(end=dims.nu) %}
            {%- if cost.Vu_0[j][k] != 0 %}
    Vu_0[{{ j }}+(NY0) * {{ k }}] = {{ cost.Vu_0[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);
    {%- endif %}

    {%- if dims.nz > 0 %}
    double* Vz_0 = calloc(NY0*NZ, sizeof(double));
    // change only the non-zero elements:
    {% for j in range(end=dims.ny_0) %}
        {%- for k in range(end=dims.nz) %}
            {%- if cost.Vz_0[j][k] != 0 %}
    Vz_0[{{ j }}+(NY0) * {{ k }}] = {{ cost.Vz_0[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vz", Vz_0);
    free(Vz_0);
    {%- endif %}
{%- endif %}

{%- if dims.ny == 0 %}
{%- elif cost.cost_type == "LINEAR_LS" %}
    double* Vx = calloc(NY*NX, sizeof(double));
    // change only the non-zero elements:
    {%- for j in range(end=dims.ny) %}
        {%- for k in range(end=dims.nx) %}
            {%- if cost.Vx[j][k] != 0 %}
    Vx[{{ j }}+(NY) * {{ k }}] = {{ cost.Vx[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    {% if dims.nu > 0 %}
    double* Vu = calloc(NY*NU, sizeof(double));
    // change only the non-zero elements:
    {% for j in range(end=dims.ny) %}
        {%- for k in range(end=dims.nu) %}
            {%- if cost.Vu[j][k] != 0 %}
    Vu[{{ j }}+(NY) * {{ k }}] = {{ cost.Vu[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);
    {%- endif %}

    {%- if dims.nz > 0 %}
    double* Vz = calloc(NY*NZ, sizeof(double));
    // change only the non-zero elements:
    {% for j in range(end=dims.ny) %}
        {%- for k in range(end=dims.nz) %}
            {%- if cost.Vz[j][k] != 0 %}
    Vz[{{ j }}+(NY) * {{ k }}] = {{ cost.Vz[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vz", Vz);
    }
    free(Vz);
    {%- endif %}
{%- endif %}{# LINEAR LS #}

{%- if dims.ny_e == 0 %}
{%- elif cost.cost_type_e == "LINEAR_LS" %}
    double* Vx_e = calloc(NYN*NX, sizeof(double));
    // change only the non-zero elements:
    {% for j in range(end=dims.ny_e) %}
        {%- for k in range(end=dims.nx) %}
            {%- if cost.Vx_e[j][k] != 0 %}
    Vx_e[{{ j }}+(NYN) * {{ k }}] = {{ cost.Vx_e[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    free(Vx_e);
{%- endif %}

{%- if cost.cost_type_0 == "NONLINEAR_LS" %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun", &capsule->cost_y_0_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun_jac", &capsule->cost_y_0_fun_jac_ut_xt);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_hess", &capsule->cost_y_0_hess);
{%- elif cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "conl_cost_fun", &capsule->conl_cost_0_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "conl_cost_fun_jac_hess", &capsule->conl_cost_0_fun_jac_hess);
{%- elif cost.cost_type_0 == "EXTERNAL" %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun", &capsule->ext_cost_0_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac", &capsule->ext_cost_0_fun_jac);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac_hess", &capsule->ext_cost_0_fun_jac_hess);
{%- endif %}

{%- if cost.cost_type == "NONLINEAR_LS" %}
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun", &capsule->cost_y_fun[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun_jac", &capsule->cost_y_fun_jac_ut_xt[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "nls_y_hess", &capsule->cost_y_hess[i-1]);
    }
{%- elif cost.cost_type == "CONVEX_OVER_NONLINEAR" %}
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "conl_cost_fun", &capsule->conl_cost_fun[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "conl_cost_fun_jac_hess", &capsule->conl_cost_fun_jac_hess[i-1]);
    }
{%- elif cost.cost_type == "EXTERNAL" %}
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun", &capsule->ext_cost_fun[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac", &capsule->ext_cost_fun_jac[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac_hess", &capsule->ext_cost_fun_jac_hess[i-1]);
    }
{%- endif %}

{%- if cost.cost_type_e == "NONLINEAR_LS" %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &capsule->cost_y_e_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &capsule->cost_y_e_fun_jac_ut_xt);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_hess", &capsule->cost_y_e_hess);

{%- elif cost.cost_type_e == "CONVEX_OVER_NONLINEAR" %}

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "conl_cost_fun", &capsule->conl_cost_e_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "conl_cost_fun_jac_hess", &capsule->conl_cost_e_fun_jac_hess);

{%- elif cost.cost_type_e == "EXTERNAL" %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun", &capsule->ext_cost_e_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac", &capsule->ext_cost_e_fun_jac);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac_hess", &capsule->ext_cost_e_fun_jac_hess);
{%- endif %}


{%- if dims.ns > 0 %}
    // slacks
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    {%- for j in range(end=dims.ns) %}
        {%- if cost.Zl[j] != 0 %}
    Zl[{{ j }}] = {{ cost.Zl[j] }};
        {%- endif %}
    {%- endfor %}

    {%- for j in range(end=dims.ns) %}
        {%- if cost.Zu[j] != 0 %}
    Zu[{{ j }}] = {{ cost.Zu[j] }};
        {%- endif %}
    {%- endfor %}

    {%- for j in range(end=dims.ns) %}
        {%- if cost.zl[j] != 0 %}
    zl[{{ j }}] = {{ cost.zl[j] }};
        {%- endif %}
    {%- endfor %}

    {%- for j in range(end=dims.ns) %}
        {%- if cost.zu[j] != 0 %}
    zu[{{ j }}] = {{ cost.zu[j] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);
{%- endif %}

{% if dims.ns_e > 0 %}
    // slacks terminal
    double* zluemem = calloc(4*NSN, sizeof(double));
    double* Zl_e = zluemem+NSN*0;
    double* Zu_e = zluemem+NSN*1;
    double* zl_e = zluemem+NSN*2;
    double* zu_e = zluemem+NSN*3;

    // change only the non-zero elements:
    {% for j in range(end=dims.ns_e) %}
        {%- if cost.Zl_e[j] != 0 %}
    Zl_e[{{ j }}] = {{ cost.Zl_e[j] }};
        {%- endif %}
    {%- endfor %}

    {% for j in range(end=dims.ns_e) %}
        {%- if cost.Zu_e[j] != 0 %}
    Zu_e[{{ j }}] = {{ cost.Zu_e[j] }};
        {%- endif %}
    {%- endfor %}

    {% for j in range(end=dims.ns_e) %}
        {%- if cost.zl_e[j] != 0 %}
    zl_e[{{ j }}] = {{ cost.zl_e[j] }};
        {%- endif %}
    {%- endfor %}

    {% for j in range(end=dims.ns_e) %}
        {%- if cost.zu_e[j] != 0 %}
    zu_e[{{ j }}] = {{ cost.zu_e[j] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zl", Zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zu", Zu_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zl", zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zu", zu_e);
    free(zluemem);
{%- endif %}

    /**** Constraints ****/

    // bounds for initial stage
{%- if dims.nbx_0 > 0 %}
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    {%- for i in range(end=dims.nbx_0) %}
    idxbx0[{{ i }}] = {{ constraints.idxbx_0[i] }};
    {%- endfor %}

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:
    {%- for i in range(end=dims.nbx_0) %}
        {%- if constraints.lbx_0[i] != 0 %}
    lbx0[{{ i }}] = {{ constraints.lbx_0[i] }};
        {%- endif %}
        {%- if constraints.ubx_0[i] != 0 %}
    ubx0[{{ i }}] = {{ constraints.ubx_0[i] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
{%- endif %}

{%- if dims.nbxe_0 > 0 %}
    // idxbxe_0
    int* idxbxe_0 = malloc({{ dims.nbxe_0 }} * sizeof(int));
    {% for i in range(end=dims.nbxe_0) %}
    idxbxe_0[{{ i }}] = {{ constraints.idxbxe_0[i] }};
    {%- endfor %}
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);
{%- endif %}

    /* constraints that are the same for initial and intermediate */
{%- if dims.nsbx > 0 %}
{# TODO: introduce nsbx0 & move this block down!! #}
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxsbx", idxsbx);
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lsbx", lsbx);
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "usbx", usbx);

    // soft bounds on x
    int* idxsbx = malloc(NSBX * sizeof(int));
    {%- for i in range(end=dims.nsbx) %}
    idxsbx[{{ i }}] = {{ constraints.idxsbx[i] }};
    {%- endfor %}

    double* lusbx = calloc(2*NSBX, sizeof(double));
    double* lsbx = lusbx;
    double* usbx = lusbx + NSBX;
    {%- for i in range(end=dims.nsbx) %}
        {%- if constraints.lsbx[i] != 0 %}
    lsbx[{{ i }}] = {{ constraints.lsbx[i] }};
        {%- endif %}
        {%- if constraints.usbx[i] != 0 %}
    usbx[{{ i }}] = {{ constraints.usbx[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsbx", idxsbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsbx", lsbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usbx", usbx);
    }
    free(idxsbx);
    free(lusbx);
{%- endif %}


{%- if dims.nbu > 0 %}
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    {% for i in range(end=dims.nbu) %}
    idxbu[{{ i }}] = {{ constraints.idxbu[i] }};
    {%- endfor %}
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    {% for i in range(end=dims.nbu) %}
        {%- if constraints.lbu[i] != 0 %}
    lbu[{{ i }}] = {{ constraints.lbu[i] }};
        {%- endif %}
        {%- if constraints.ubu[i] != 0 %}
    ubu[{{ i }}] = {{ constraints.ubu[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);
{%- endif %}

{%- if dims.nsbu > 0 %}
    // set up soft bounds for u
    int* idxsbu = malloc(NSBU * sizeof(int));
    {% for i in range(end=dims.nsbu) %}
    idxsbu[{{ i }}] = {{ constraints.idxsbu[i] }};
    {%- endfor %}
    double* lusbu = calloc(2*NSBU, sizeof(double));
    double* lsbu = lusbu;
    double* usbu = lusbu + NSBU;
    {% for i in range(end=dims.nsbu) %}
        {%- if constraints.lsbu[i] != 0 %}
    lsbu[{{ i }}] = {{ constraints.lsbu[i] }};
        {%- endif %}
        {%- if constraints.usbu[i] != 0 %}
    usbu[{{ i }}] = {{ constraints.usbu[i] }};
        {%- endif %}
    {%- endfor %}
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsbu", idxsbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsbu", lsbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usbu", usbu);
    }
    free(idxsbu);
    free(lusbu);
{%- endif %}

{% if dims.nsg > 0 %}
    // set up soft bounds for general linear constraints
    int* idxsg = malloc(NSG * sizeof(int));
    {% for i in range(end=dims.nsg) %}
    idxsg[{{ i }}] = {{ constraints.idxsg[i] }};
    {%- endfor %}
    double* lusg = calloc(2*NSG, sizeof(double));
    double* lsg = lusg;
    double* usg = lusg + NSG;
    {% for i in range(end=dims.nsg) %}
        {%- if constraints.lsg[i] != 0 %}
    lsg[{{ i }}] = {{ constraints.lsg[i] }};
        {%- endif %}
        {%- if constraints.usg[i] != 0 %}
    usg[{{ i }}] = {{ constraints.usg[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsg", idxsg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsg", lsg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usg", usg);
    }
    free(idxsg);
    free(lusg);
{%- endif %}

{% if dims.nsh > 0 %}
    // set up soft bounds for nonlinear constraints
    int* idxsh = malloc(NSH * sizeof(int));
    {% for i in range(end=dims.nsh) %}
    idxsh[{{ i }}] = {{ constraints.idxsh[i] }};
    {%- endfor %}
    double* lush = calloc(2*NSH, sizeof(double));
    double* lsh = lush;
    double* ush = lush + NSH;
    {% for i in range(end=dims.nsh) %}
        {%- if constraints.lsh[i] != 0 %}
    lsh[{{ i }}] = {{ constraints.lsh[i] }};
        {%- endif %}
        {%- if constraints.ush[i] != 0 %}
    ush[{{ i }}] = {{ constraints.ush[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsh", idxsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsh", lsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ush", ush);
    }
    free(idxsh);
    free(lush);
{%- endif %}

{% if dims.nsphi > 0 %}
    // set up soft bounds for convex-over-nonlinear constraints
    int* idxsphi = malloc(NSPHI * sizeof(int));
    {% for i in range(end=dims.nsphi) %}
    idxsphi[{{ i }}] = {{ constraints.idxsphi[i] }};
    {%- endfor %}
    double* lusphi = calloc(2*NSPHI, sizeof(double));
    double* lsphi = lusphi;
    double* usphi = lusphi + NSPHI;
    {% for i in range(end=dims.nsphi) %}
        {%- if constraints.lsphi[i] != 0 %}
    lsphi[{{ i }}] = {{ constraints.lsphi[i] }};
        {%- endif %}
        {%- if constraints.usphi[i] != 0 %}
    usphi[{{ i }}] = {{ constraints.usphi[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsphi", idxsphi);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsphi", lsphi);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usphi", usphi);
    }
    free(idxsphi);
    free(lusphi);
{%- endif %}

{% if dims.nbx > 0 %}
    // x
    int* idxbx = malloc(NBX * sizeof(int));
    {% for i in range(end=dims.nbx) %}
    idxbx[{{ i }}] = {{ constraints.idxbx[i] }};
    {%- endfor %}
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    {% for i in range(end=dims.nbx) %}
        {%- if constraints.lbx[i] != 0 %}
    lbx[{{ i }}] = {{ constraints.lbx[i] }};
        {%- endif %}
        {%- if constraints.ubx[i] != 0 %}
    ubx[{{ i }}] = {{ constraints.ubx[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);
{%- endif %}

{% if dims.ng > 0 %}
    // set up general constraints for stage 0 to N-1
    double* D = calloc(NG*NU, sizeof(double));
    double* C = calloc(NG*NX, sizeof(double));
    double* lug = calloc(2*NG, sizeof(double));
    double* lg = lug;
    double* ug = lug + NG;

    {% for j in range(end=dims.ng) -%}
        {% for k in range(end=dims.nu) %}
            {%- if constraints.D[j][k] != 0 %}
    D[{{ j }}+NG * {{ k }}] = {{ constraints.D[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}

    {% for j in range(end=dims.ng) -%}
        {% for k in range(end=dims.nx) %}
            {%- if constraints.C[j][k] != 0 %}
    C[{{ j }}+NG * {{ k }}] = {{ constraints.C[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}

    {% for i in range(end=dims.ng) %}
        {%- if constraints.lg[i] != 0 %}
    lg[{{ i }}] = {{ constraints.lg[i] }};
        {%- endif %}
    {%- endfor %}

    {% for i in range(end=dims.ng) %}
        {%- if constraints.ug[i] != 0 %}
    ug[{{ i }}] = {{ constraints.ug[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "D", D);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "C", C);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lg", lg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ug", ug);
    }
    free(D);
    free(C);
    free(lug);
{%- endif %}

{% if dims.nh > 0 %}
    // set up nonlinear constraints for stage 0 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;

    {% for i in range(end=dims.nh) %}
        {%- if constraints.lh[i] != 0 %}
    lh[{{ i }}] = {{ constraints.lh[i] }};
        {%- endif %}
    {%- endfor %}

    {% for i in range(end=dims.nh) %}
        {%- if constraints.uh[i] != 0 %}
    uh[{{ i }}] = {{ constraints.uh[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        // nonlinear constraints for stages 0 to N-1
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i]);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i]);
        {% if solver_options.hessian_approx == "EXACT" %}
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i,
                                      "nl_constr_h_fun_jac_hess", &capsule->nl_constr_h_fun_jac_hess[i]);
        {% endif %}
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "uh", uh);
    }
    free(luh);
{%- endif %}

{% if dims.nphi > 0 and constraints.constr_type == "BGP" %}
    // set up convex-over-nonlinear constraints for stage 0 to N-1
    double* luphi = calloc(2*NPHI, sizeof(double));
    double* lphi = luphi;
    double* uphi = luphi + NPHI;
    {% for i in range(end=dims.nphi) %}
        {%- if constraints.lphi[i] != 0 %}
    lphi[{{ i }}] = {{ constraints.lphi[i] }};
        {%- endif %}
    {%- endfor %}

    {% for i in range(end=dims.nphi) %}
        {%- if constraints.uphi[i] != 0 %}
    uphi[{{ i }}] = {{ constraints.uphi[i] }};
        {%- endif %}
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i,
                                      "nl_constr_phi_o_r_fun_phi_jac_ux_z_phi_hess_r_jac_ux", &capsule->phi_constraint[i]);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lphi", lphi);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "uphi", uphi);
    }
    free(luphi);
{%- endif %}

    /* terminal constraints */
{% if dims.nbx_e > 0 %}
    // set up bounds for last stage
    // x
    int* idxbx_e = malloc(NBXN * sizeof(int));
    {% for i in range(end=dims.nbx_e) %}
    idxbx_e[{{ i }}] = {{ constraints.idxbx_e[i] }};
    {%- endfor %}
    double* lubx_e = calloc(2*NBXN, sizeof(double));
    double* lbx_e = lubx_e;
    double* ubx_e = lubx_e + NBXN;
    {% for i in range(end=dims.nbx_e) %}
        {%- if constraints.lbx_e[i] != 0 %}
    lbx_e[{{ i }}] = {{ constraints.lbx_e[i] }};
        {%- endif %}
        {%- if constraints.ubx_e[i] != 0 %}
    ubx_e[{{ i }}] = {{ constraints.ubx_e[i] }};
        {%- endif %}
    {%- endfor %}
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxbx", idxbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lbx", lbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ubx", ubx_e);
    free(idxbx_e);
    free(lubx_e);
{%- endif %}

{% if dims.nsg_e > 0 %}
    // set up soft bounds for general linear constraints
    int* idxsg_e = calloc(NSGN, sizeof(int));
    {% for i in range(end=dims.nsg_e) %}
    idxsg_e[{{ i }}] = {{ constraints.idxsg_e[i] }};
    {%- endfor %}
    double* lusg_e = calloc(2*NSGN, sizeof(double));
    double* lsg_e = lusg_e;
    double* usg_e = lusg_e + NSGN;
    {% for i in range(end=dims.nsg_e) %}
        {%- if constraints.lsg_e[i] != 0 %}
    lsg_e[{{ i }}] = {{ constraints.lsg_e[i] }};
        {%- endif %}
        {%- if constraints.usg_e[i] != 0 %}
    usg_e[{{ i }}] = {{ constraints.usg_e[i] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsg", idxsg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsg", lsg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "usg", usg_e);
    free(idxsg_e);
    free(lusg_e);
{%- endif %}

{% if dims.nsh_e > 0 %}
    // set up soft bounds for nonlinear constraints
    int* idxsh_e = malloc(NSHN * sizeof(int));
    {% for i in range(end=dims.nsh_e) %}
    idxsh_e[{{ i }}] = {{ constraints.idxsh_e[i] }};
    {%- endfor %}
    double* lush_e = calloc(2*NSHN, sizeof(double));
    double* lsh_e = lush_e;
    double* ush_e = lush_e + NSHN;
    {% for i in range(end=dims.nsh_e) %}
        {%- if constraints.lsh_e[i] != 0 %}
    lsh_e[{{ i }}] = {{ constraints.lsh_e[i] }};
        {%- endif %}
        {%- if constraints.ush_e[i] != 0 %}
    ush_e[{{ i }}] = {{ constraints.ush_e[i] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsh", idxsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsh", lsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ush", ush_e);
    free(idxsh_e);
    free(lush_e);
{%- endif %}

{% if dims.nsphi_e > 0 %}
    // set up soft bounds for convex-over-nonlinear constraints
    int* idxsphi_e = malloc(NSPHIN * sizeof(int));
    {% for i in range(end=dims.nsphi_e) %}
    idxsphi_e[{{ i }}] = {{ constraints.idxsphi_e[i] }};
    {%- endfor %}
    double* lusphi_e = calloc(2*NSPHIN, sizeof(double));
    double* lsphi_e = lusphi_e;
    double* usphi_e = lusphi_e + NSPHIN;
    {% for i in range(end=dims.nsphi_e) %}
        {%- if constraints.lsphi_e[i] != 0 %}
    lsphi_e[{{ i }}] = {{ constraints.lsphi_e[i] }};
        {%- endif %}
        {%- if constraints.usphi_e[i] != 0 %}
    usphi_e[{{ i }}] = {{ constraints.usphi_e[i] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsphi", idxsphi_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsphi", lsphi_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "usphi", usphi_e);
    free(idxsphi_e);
    free(lusphi_e);
{%- endif %}

{% if dims.nsbx_e > 0 %}
    // soft bounds on x
    int* idxsbx_e = malloc(NSBXN * sizeof(int));
    {% for i in range(end=dims.nsbx_e) %}
    idxsbx_e[{{ i }}] = {{ constraints.idxsbx_e[i] }};
    {%- endfor %}
    double* lusbx_e = calloc(2*NSBXN, sizeof(double));
    double* lsbx_e = lusbx_e;
    double* usbx_e = lusbx_e + NSBXN;
    {% for i in range(end=dims.nsbx_e) %}
        {%- if constraints.lsbx_e[i] != 0 %}
    lsbx_e[{{ i }}] = {{ constraints.lsbx_e[i] }};
        {%- endif %}
        {%- if constraints.usbx_e[i] != 0 %}
    usbx_e[{{ i }}] = {{ constraints.usbx_e[i] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsbx", idxsbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsbx", lsbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "usbx", usbx_e);
    free(idxsbx_e);
    free(lusbx_e);
{% endif %}

{% if dims.ng_e > 0 %}
    // set up general constraints for last stage
    double* C_e = calloc(NGN*NX, sizeof(double));
    double* lug_e = calloc(2*NGN, sizeof(double));
    double* lg_e = lug_e;
    double* ug_e = lug_e + NGN;

    {% for j in range(end=dims.ng_e) %}
        {%- for k in range(end=dims.nx) %}
            {%- if constraints.C_e[j][k] != 0 %}
    C_e[{{ j }}+NGN * {{ k }}] = {{ constraints.C_e[j][k] }};
            {%- endif %}
        {%- endfor %}
    {%- endfor %}

    {% for i in range(end=dims.ng_e) %}
        {%- if constraints.lg_e[i] != 0 %}
    lg_e[{{ i }}] = {{ constraints.lg_e[i] }};
        {%- endif %}
        {%- if constraints.ug_e[i] != 0 %}
    ug_e[{{ i }}] = {{ constraints.ug_e[i] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "C", C_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lg", lg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ug", ug_e);
    free(C_e);
    free(lug_e);
{%- endif %}

{% if dims.nh_e > 0 %}
    // set up nonlinear constraints for last stage
    double* luh_e = calloc(2*NHN, sizeof(double));
    double* lh_e = luh_e;
    double* uh_e = luh_e + NHN;
    {% for i in range(end=dims.nh_e) %}
        {%- if constraints.lh_e[i] != 0 %}
    lh_e[{{ i }}] = {{ constraints.lh_e[i] }};
        {%- endif %}
    {%- endfor %}

    {% for i in range(end=dims.nh_e) %}
        {%- if constraints.uh_e[i] != 0 %}
    uh_e[{{ i }}] = {{ constraints.uh_e[i] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac", &capsule->nl_constr_h_e_fun_jac);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun", &capsule->nl_constr_h_e_fun);
    {% if solver_options.hessian_approx == "EXACT" %}
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac_hess",
                                  &capsule->nl_constr_h_e_fun_jac_hess);
    {% endif %}
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lh", lh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "uh", uh_e);
    free(luh_e);
{%- endif %}

{% if dims.nphi_e > 0 and constraints.constr_type_e == "BGP" %}
    // set up convex-over-nonlinear constraints for last stage
    double* luphi_e = calloc(2*NPHIN, sizeof(double));
    double* lphi_e = luphi_e;
    double* uphi_e = luphi_e + NPHIN;
    {% for i in range(end=dims.nphi_e) %}
        {%- if constraints.lphi_e[i] != 0 %}
    lphi_e[{{ i }}] = {{ constraints.lphi_e[i] }};
        {%- endif %}
        {%- if constraints.uphi_e[i] != 0 %}
    uphi_e[{{ i }}] = {{ constraints.uphi_e[i] }};
        {%- endif %}
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lphi", lphi_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "uphi", uphi_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N,
                                  "nl_constr_phi_o_r_fun_phi_jac_ux_z_phi_hess_r_jac_ux", &capsule->phi_e_constraint);
    free(luphi_e);
{% endif %}
}


/**
 * Internal function for {{ model.name }}_acados_create: step 6
 */
void {{ model.name }}_acados_create_6_set_opts({{ model.name }}_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/

{% if solver_options.hessian_approx == "EXACT" %}
    int nlp_solver_exact_hessian = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess", &nlp_solver_exact_hessian);

    int exact_hess_dyn = {{ solver_options.exact_hess_dyn }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_dyn", &exact_hess_dyn);

    int exact_hess_cost = {{ solver_options.exact_hess_cost }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_cost", &exact_hess_cost);

    int exact_hess_constr = {{ solver_options.exact_hess_constr }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_constr", &exact_hess_constr);
{%- endif -%}

{%- if solver_options.globalization == "FIXED_STEP" %}
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization", "fixed_step");
{%- elif solver_options.globalization == "MERIT_BACKTRACKING" %}
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization", "merit_backtracking");

    double alpha_min = {{ solver_options.alpha_min }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "alpha_min", &alpha_min);

    double alpha_reduction = {{ solver_options.alpha_reduction }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "alpha_reduction", &alpha_reduction);

    int line_search_use_sufficient_descent = {{ solver_options.line_search_use_sufficient_descent }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "line_search_use_sufficient_descent", &line_search_use_sufficient_descent);

    int globalization_use_SOC = {{ solver_options.globalization_use_SOC }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_use_SOC", &globalization_use_SOC);

    double eps_sufficient_descent = {{ solver_options.eps_sufficient_descent }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "eps_sufficient_descent", &eps_sufficient_descent);
{%- endif -%}
    int full_step_dual = {{ solver_options.full_step_dual }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "full_step_dual", &full_step_dual);

{%- if dims.nz > 0 %}
    // TODO: these options are lower level -> should be encapsulated! maybe through hessian approx option.
    bool output_z_val = true;
    bool sens_algebraic_val = true;

    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_output_z", &output_z_val);
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_sens_algebraic", &sens_algebraic_val);
{%- endif %}

{%- if solver_options.integrator_type != "DISCRETE" %}

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = {{ solver_options.collocation_type }};
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    {%- set all_equal = true %}
    {%- set val = solver_options.sim_method_num_steps[0] %}
    {%- for j in range(start=1, end=dims.N) %}
        {%- if val != solver_options.sim_method_num_steps[j] %}
            {%- set_global all_equal = false %}
            {%- break %}
        {%- endif %}
    {%- endfor %}

    {%- if all_equal == true %}
    // all sim_method_num_steps are identical
    int sim_method_num_steps = {{ solver_options.sim_method_num_steps[0] }};
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);
    {%- else %}
    // sim_method_num_steps are different
    int* sim_method_num_steps = malloc(N*sizeof(int));
    {%- for j in range(end=dims.N) %}
    sim_method_num_steps[{{ j }}] = {{ solver_options.sim_method_num_steps[j] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps[i]);
    free(sim_method_num_steps);
    {%- endif %}

    // set up sim_method_num_stages
    {%- set all_equal = true %}
    {%- set val = solver_options.sim_method_num_stages[0] %}
    {%- for j in range(start=1, end=dims.N) %}
        {%- if val != solver_options.sim_method_num_stages[j] %}
            {%- set_global all_equal = false %}
            {%- break %}
        {%- endif %}
    {%- endfor %}

  {%- if all_equal == true %}
    // all sim_method_num_stages are identical
    int sim_method_num_stages = {{ solver_options.sim_method_num_stages[0] }};
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);
  {%- else %}
    int* sim_method_num_stages = malloc(N*sizeof(int));
    {%- for j in range(end=dims.N) %}
    sim_method_num_stages[{{ j }}] = {{ solver_options.sim_method_num_stages[j] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages[i]);
    free(sim_method_num_stages);
  {%- endif %}

    int newton_iter_val = {{ solver_options.sim_method_newton_iter }};
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);


    // set up sim_method_jac_reuse
    {%- set all_equal = true %}
    {%- set val = solver_options.sim_method_jac_reuse[0] %}
    {%- for j in range(start=1, end=dims.N) %}
        {%- if val != solver_options.sim_method_jac_reuse[j] %}
            {%- set_global all_equal = false %}
            {%- break %}
        {%- endif %}
    {%- endfor %}
  {%- if all_equal == true %}
    bool tmp_bool = (bool) {{ solver_options.sim_method_jac_reuse[0] }};
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);
  {%- else %}
    bool* sim_method_jac_reuse = malloc(N*sizeof(bool));
    {%- for j in range(end=dims.N) %}
    sim_method_jac_reuse[{{ j }}] = (bool){{ solver_options.sim_method_jac_reuse[j] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &sim_method_jac_reuse[i]);
    free(sim_method_jac_reuse);
  {%- endif %}

{%- endif %}

    double nlp_solver_step_length = {{ solver_options.nlp_solver_step_length }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "step_length", &nlp_solver_step_length);

    {%- if solver_options.nlp_solver_warm_start_first_qp %}
    int nlp_solver_warm_start_first_qp = {{ solver_options.nlp_solver_warm_start_first_qp }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "warm_start_first_qp", &nlp_solver_warm_start_first_qp);
    {%- endif %}

    double levenberg_marquardt = {{ solver_options.levenberg_marquardt }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
{%- if solver_options.qp_solver is starting_with("PARTIAL_CONDENSING") %}
    int qp_solver_cond_N;

    {% if solver_options.qp_solver_cond_N -%}
    const int qp_solver_cond_N_ori = {{ solver_options.qp_solver_cond_N }};
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    {%- else %}
    // NOTE: there is no condensing happening here!
    qp_solver_cond_N = N;
    {%- endif %}
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);
{%- endif %}

    int nlp_solver_ext_qp_res = {{ solver_options.nlp_solver_ext_qp_res }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

{%- if solver_options.qp_solver is containing("HPIPM") %}
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "{{ solver_options.hpipm_mode }}");
{%- endif %}

{% if solver_options.nlp_solver_type == "SQP" %}
    // set SQP specific options
    double nlp_solver_tol_stat = {{ solver_options.nlp_solver_tol_stat }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = {{ solver_options.nlp_solver_tol_eq }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = {{ solver_options.nlp_solver_tol_ineq }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = {{ solver_options.nlp_solver_tol_comp }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = {{ solver_options.nlp_solver_max_iter }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    int initialize_t_slacks = {{ solver_options.initialize_t_slacks }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "initialize_t_slacks", &initialize_t_slacks);
{%- endif %}

    int qp_solver_iter_max = {{ solver_options.qp_solver_iter_max }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);

{# NOTE: qp_solver tolerances must be set after NLP ones, since the setter for NLP tolerances sets the QP tolerances to the sam values. #}
    {%- if solver_options.qp_solver_tol_stat %}
    double qp_solver_tol_stat = {{ solver_options.qp_solver_tol_stat }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_tol_stat", &qp_solver_tol_stat);
    {%- endif -%}

    {%- if solver_options.qp_solver_tol_eq %}
    double qp_solver_tol_eq = {{ solver_options.qp_solver_tol_eq }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_tol_eq", &qp_solver_tol_eq);
    {%- endif -%}

    {%- if solver_options.qp_solver_tol_ineq %}
    double qp_solver_tol_ineq = {{ solver_options.qp_solver_tol_ineq }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_tol_ineq", &qp_solver_tol_ineq);
    {%- endif -%}

    {%- if solver_options.qp_solver_tol_comp %}
    double qp_solver_tol_comp = {{ solver_options.qp_solver_tol_comp }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_tol_comp", &qp_solver_tol_comp);
    {%- endif -%}

    {%- if solver_options.qp_solver_warm_start %}
    int qp_solver_warm_start = {{ solver_options.qp_solver_warm_start }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_warm_start", &qp_solver_warm_start);
    {%- endif -%}

    int print_level = {{ solver_options.print_level }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);

{%- if solver_options.qp_solver is containing('PARTIAL_CONDENSING') %}
    int qp_solver_cond_ric_alg = {{ solver_options.qp_solver_cond_ric_alg }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);
{% endif %}

{%- if solver_options.qp_solver  == 'PARTIAL_CONDENSING_HPIPM' %}
    int qp_solver_ric_alg = {{ solver_options.qp_solver_ric_alg }};
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);
{% endif %}

    int ext_cost_num_hess = {{ solver_options.ext_cost_num_hess }};
{%- if cost.cost_type == "EXTERNAL" %}
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "cost_numerical_hessian", &ext_cost_num_hess);
    }
{%- endif %}
{%- if cost.cost_type_e == "EXTERNAL" %}
    ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, N, "cost_numerical_hessian", &ext_cost_num_hess);
{%- endif %}
}


/**
 * Internal function for {{ model.name }}_acados_create: step 7
 */
void {{ model.name }}_acados_create_7_set_nlp_out({{ model.name }}_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;
{% if dims.nbx_0 == dims.nx %}
    // initialize with x0
    {% for item in constraints.lbx_0 %}
        {%- if item != 0 %}
    x0[{{ loop.index0 }}] = {{ item }};
        {%- endif %}
    {%- endfor %}
{% else %}
    // initialize with zeros
{%- endif %}

    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for {{ model.name }}_acados_create: step 8
 */
//void {{ model.name }}_acados_create_8_create_solver({{ model.name }}_solver_capsule* capsule)
//{
//    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);
//}

/**
 * Internal function for {{ model.name }}_acados_create: step 9
 */
int {{ model.name }}_acados_create_9_precompute({{ model.name }}_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int {{ model.name }}_acados_create_with_discretization({{ model.name }}_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != {{ model.name | upper }}_N && !new_time_steps) {
        fprintf(stderr, "{{ model.name }}_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, {{ model.name | upper }}_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    {{ model.name }}_acados_create_1_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 3) create and set dimensions
    capsule->nlp_dims = {{ model.name }}_acados_create_2_create_and_set_dimensions(capsule);
    {{ model.name }}_acados_create_3_create_and_set_functions(capsule);

    // 4) set default parameters in functions
    {{ model.name }}_acados_create_4_set_default_parameters(capsule);

    // 5) create and set nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);
    {{ model.name }}_acados_create_5_set_nlp_in(capsule, N, new_time_steps);

    // 6) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    {{ model.name }}_acados_create_6_set_opts(capsule);

    // 7) create and set nlp_out
    // 7.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 7.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    {{ model.name }}_acados_create_7_set_nlp_out(capsule);

    // 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);
    //{{ model.name }}_acados_create_8_create_solver(capsule);

    // 9) do precomputations
    int status = {{ model.name }}_acados_create_9_precompute(capsule);

    {%- if custom_update_filename != "" %}
    // Initialize custom update function
    custom_update_init_function(capsule);
    {%- endif %}

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int {{ model.name }}_acados_update_qp_solver_cond_N({{ model.name }}_solver_capsule* capsule, int qp_solver_cond_N)
{
{%- if solver_options.qp_solver is starting_with("PARTIAL_CONDENSING") %}
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from {{ model.name }}_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);

    // -> 9) do precomputations
    int status = {{ model.name }}_acados_create_9_precompute(capsule);
    return status;
{%- else %}
    printf("\nacados_update_qp_solver_cond_N() failed, since no partial condensing solver is used!\n\n");
    // Todo: what is an adequate behavior here?
    exit(1);
    return -1;
{%- endif %}
}


int {{ model.name }}_acados_reset({{ model.name }}_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros
{# TODO: use guess values / initial state value from json instead?! #}
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "t", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "pi", buffer);
        {%- if solver_options.integrator_type == "IRK" %}
            ocp_nlp_set(nlp_config, nlp_solver, i, "xdot_guess", buffer);
            ocp_nlp_set(nlp_config, nlp_solver, i, "z_guess", buffer);
        {% elif solver_options.integrator_type == "LIFTED_IRK" %}
            ocp_nlp_set(nlp_config, nlp_solver, i, "xdot_guess", buffer);
        {% elif solver_options.integrator_type == "GNSF" %}
            ocp_nlp_set(nlp_config, nlp_solver, i, "gnsf_phi_guess", buffer);
        {%- endif %}
        }
    }

{%- if solver_options.qp_solver == 'PARTIAL_CONDENSING_HPIPM' %}
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }
{%- endif %}

    free(buffer);
    return 0;
}




int {{ model.name }}_acados_update_params({{ model.name }}_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = {{ dims.np }};
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }

    const int N = capsule->nlp_solver_plan->N;
    if (stage < N && stage >= 0)
    {
    {%- if solver_options.integrator_type == "IRK" %}
        capsule->impl_dae_fun[stage].set_param(capsule->impl_dae_fun+stage, p);
        capsule->impl_dae_fun_jac_x_xdot_z[stage].set_param(capsule->impl_dae_fun_jac_x_xdot_z+stage, p);
        capsule->impl_dae_jac_x_xdot_u_z[stage].set_param(capsule->impl_dae_jac_x_xdot_u_z+stage, p);

        {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->impl_dae_hess[stage].set_param(capsule->impl_dae_hess+stage, p);
        {%- endif %}
    {% elif solver_options.integrator_type == "LIFTED_IRK" %}
        capsule->impl_dae_fun[stage].set_param(capsule->impl_dae_fun+stage, p);
        capsule->impl_dae_fun_jac_x_xdot_u[stage].set_param(capsule->impl_dae_fun_jac_x_xdot_u+stage, p);
    {% elif solver_options.integrator_type == "ERK" %}
        capsule->forw_vde_casadi[stage].set_param(capsule->forw_vde_casadi+stage, p);
        capsule->expl_ode_fun[stage].set_param(capsule->expl_ode_fun+stage, p);

        {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->hess_vde_casadi[stage].set_param(capsule->hess_vde_casadi+stage, p);
        {%- endif %}
    {% elif solver_options.integrator_type == "GNSF" %}
    {% if model.gnsf.purely_linear != 1 %}
        capsule->gnsf_phi_fun[stage].set_param(capsule->gnsf_phi_fun+stage, p);
        capsule->gnsf_phi_fun_jac_y[stage].set_param(capsule->gnsf_phi_fun_jac_y+stage, p);
        capsule->gnsf_phi_jac_y_uhat[stage].set_param(capsule->gnsf_phi_jac_y_uhat+stage, p);
        {% if model.gnsf.nontrivial_f_LO == 1 %}
            capsule->gnsf_f_lo_jac_x1_x1dot_u_z[stage].set_param(capsule->gnsf_f_lo_jac_x1_x1dot_u_z+stage, p);
        {%- endif %}
    {%- endif %}
    {% elif solver_options.integrator_type == "DISCRETE" %}
        capsule->discr_dyn_phi_fun[stage].set_param(capsule->discr_dyn_phi_fun+stage, p);
        capsule->discr_dyn_phi_fun_jac_ut_xt[stage].set_param(capsule->discr_dyn_phi_fun_jac_ut_xt+stage, p);
    {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[stage].set_param(capsule->discr_dyn_phi_fun_jac_ut_xt_hess+stage, p);
    {% endif %}
    {%- endif %}{# integrator_type #}

        // constraints
    {% if constraints.constr_type == "BGP" %}
        capsule->phi_constraint[stage].set_param(capsule->phi_constraint+stage, p);
    {% elif constraints.constr_type == "BGH" and dims.nh > 0 %}
        capsule->nl_constr_h_fun_jac[stage].set_param(capsule->nl_constr_h_fun_jac+stage, p);
        capsule->nl_constr_h_fun[stage].set_param(capsule->nl_constr_h_fun+stage, p);
    {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->nl_constr_h_fun_jac_hess[stage].set_param(capsule->nl_constr_h_fun_jac_hess+stage, p);
    {%- endif %}
    {%- endif %}

        // cost
        if (stage == 0)
        {
        {%- if cost.cost_type_0 == "NONLINEAR_LS" %}
            capsule->cost_y_0_fun.set_param(&capsule->cost_y_0_fun, p);
            capsule->cost_y_0_fun_jac_ut_xt.set_param(&capsule->cost_y_0_fun_jac_ut_xt, p);
            capsule->cost_y_0_hess.set_param(&capsule->cost_y_0_hess, p);
        {%- elif cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}
            capsule->conl_cost_0_fun.set_param(&capsule->conl_cost_0_fun, p);
            capsule->conl_cost_0_fun_jac_hess.set_param(&capsule->conl_cost_0_fun_jac_hess, p);
        {%- elif cost.cost_type_0 == "EXTERNAL" %}
            capsule->ext_cost_0_fun.set_param(&capsule->ext_cost_0_fun, p);
            capsule->ext_cost_0_fun_jac.set_param(&capsule->ext_cost_0_fun_jac, p);
            capsule->ext_cost_0_fun_jac_hess.set_param(&capsule->ext_cost_0_fun_jac_hess, p);
        {% endif %}
        }
        else // 0 < stage < N
        {
        {%- if cost.cost_type == "NONLINEAR_LS" %}
            capsule->cost_y_fun[stage-1].set_param(capsule->cost_y_fun+stage-1, p);
            capsule->cost_y_fun_jac_ut_xt[stage-1].set_param(capsule->cost_y_fun_jac_ut_xt+stage-1, p);
            capsule->cost_y_hess[stage-1].set_param(capsule->cost_y_hess+stage-1, p);
        {%- elif cost.cost_type == "CONVEX_OVER_NONLINEAR" %}
            capsule->conl_cost_fun[stage-1].set_param(capsule->conl_cost_fun+stage-1, p);
            capsule->conl_cost_fun_jac_hess[stage-1].set_param(capsule->conl_cost_fun_jac_hess+stage-1, p);
        {%- elif cost.cost_type == "EXTERNAL" %}
            capsule->ext_cost_fun[stage-1].set_param(capsule->ext_cost_fun+stage-1, p);
            capsule->ext_cost_fun_jac[stage-1].set_param(capsule->ext_cost_fun_jac+stage-1, p);
            capsule->ext_cost_fun_jac_hess[stage-1].set_param(capsule->ext_cost_fun_jac_hess+stage-1, p);
        {%- endif %}
        }
    }

    else // stage == N
    {
        // terminal shooting node has no dynamics
        // cost
    {%- if cost.cost_type_e == "NONLINEAR_LS" %}
        capsule->cost_y_e_fun.set_param(&capsule->cost_y_e_fun, p);
        capsule->cost_y_e_fun_jac_ut_xt.set_param(&capsule->cost_y_e_fun_jac_ut_xt, p);
        capsule->cost_y_e_hess.set_param(&capsule->cost_y_e_hess, p);
    {%- elif cost.cost_type_e == "CONVEX_OVER_NONLINEAR" %}
        capsule->conl_cost_e_fun.set_param(&capsule->conl_cost_e_fun, p);
        capsule->conl_cost_e_fun_jac_hess.set_param(&capsule->conl_cost_e_fun_jac_hess, p);
    {%- elif cost.cost_type_e == "EXTERNAL" %}
        capsule->ext_cost_e_fun.set_param(&capsule->ext_cost_e_fun, p);
        capsule->ext_cost_e_fun_jac.set_param(&capsule->ext_cost_e_fun_jac, p);
        capsule->ext_cost_e_fun_jac_hess.set_param(&capsule->ext_cost_e_fun_jac_hess, p);
    {% endif %}
        // constraints
    {% if constraints.constr_type_e == "BGP" %}
        capsule->phi_e_constraint.set_param(&capsule->phi_e_constraint, p);
    {% elif constraints.constr_type_e == "BGH" and dims.nh_e > 0 %}
        capsule->nl_constr_h_e_fun_jac.set_param(&capsule->nl_constr_h_e_fun_jac, p);
        capsule->nl_constr_h_e_fun.set_param(&capsule->nl_constr_h_e_fun, p);
    {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->nl_constr_h_e_fun_jac_hess.set_param(&capsule->nl_constr_h_e_fun_jac_hess, p);
    {%- endif %}
    {% endif %}
    }

    return solver_status;
}


int {{ model.name }}_acados_update_params_sparse({{ model.name }}_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    int solver_status = 0;

    int casadi_np = {{ dims.np }};
    if (casadi_np < n_update) {
        printf("{{ model.name }}_acados_update_params_sparse: trying to set %d parameters for external functions."
            " External function has %d parameters. Exiting.\n", n_update, casadi_np);
        exit(1);
    }
    // for (int i = 0; i < n_update; i++)
    // {
    //     if (idx[i] > casadi_np) {
    //         printf("{{ model.name }}_acados_update_params_sparse: attempt to set parameters with index %d, while"
    //             " external functions only has %d parameters. Exiting.\n", idx[i], casadi_np);
    //         exit(1);
    //     }
    //     printf("param %d value %e\n", idx[i], p[i]);
    // }

{%- if dims.np > 0 %}
    const int N = capsule->nlp_solver_plan->N;
    if (stage < N && stage >= 0)
    {
    {%- if solver_options.integrator_type == "IRK" %}
        capsule->impl_dae_fun[stage].set_param_sparse(capsule->impl_dae_fun+stage, n_update, idx, p);
        capsule->impl_dae_fun_jac_x_xdot_z[stage].set_param_sparse(capsule->impl_dae_fun_jac_x_xdot_z+stage, n_update, idx, p);
        capsule->impl_dae_jac_x_xdot_u_z[stage].set_param_sparse(capsule->impl_dae_jac_x_xdot_u_z+stage, n_update, idx, p);

        {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->impl_dae_hess[stage].set_param_sparse(capsule->impl_dae_hess+stage, n_update, idx, p);
        {%- endif %}
    {% elif solver_options.integrator_type == "LIFTED_IRK" %}
        capsule->impl_dae_fun[stage].set_param_sparse(capsule->impl_dae_fun+stage, n_update, idx, p);
        capsule->impl_dae_fun_jac_x_xdot_u[stage].set_param_sparse(capsule->impl_dae_fun_jac_x_xdot_u+stage, n_update, idx, p);
    {% elif solver_options.integrator_type == "ERK" %}
        capsule->forw_vde_casadi[stage].set_param_sparse(capsule->forw_vde_casadi+stage, n_update, idx, p);
        capsule->expl_ode_fun[stage].set_param_sparse(capsule->expl_ode_fun+stage, n_update, idx, p);

        {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->hess_vde_casadi[stage].set_param_sparse(capsule->hess_vde_casadi+stage, n_update, idx, p);
        {%- endif %}
    {% elif solver_options.integrator_type == "GNSF" %}
    {% if model.gnsf.purely_linear != 1 %}
        capsule->gnsf_phi_fun[stage].set_param_sparse(capsule->gnsf_phi_fun+stage, n_update, idx, p);
        capsule->gnsf_phi_fun_jac_y[stage].set_param_sparse(capsule->gnsf_phi_fun_jac_y+stage, n_update, idx, p);
        capsule->gnsf_phi_jac_y_uhat[stage].set_param_sparse(capsule->gnsf_phi_jac_y_uhat+stage, n_update, idx, p);
        {% if model.gnsf.nontrivial_f_LO == 1 %}
            capsule->gnsf_f_lo_jac_x1_x1dot_u_z[stage].set_param_sparse(capsule->gnsf_f_lo_jac_x1_x1dot_u_z+stage, n_update, idx, p);
        {%- endif %}
    {%- endif %}
    {% elif solver_options.integrator_type == "DISCRETE" %}
        capsule->discr_dyn_phi_fun[stage].set_param_sparse(capsule->discr_dyn_phi_fun+stage, n_update, idx, p);
        capsule->discr_dyn_phi_fun_jac_ut_xt[stage].set_param_sparse(capsule->discr_dyn_phi_fun_jac_ut_xt+stage, n_update, idx, p);
    {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[stage].set_param_sparse(capsule->discr_dyn_phi_fun_jac_ut_xt_hess+stage, n_update, idx, p);
    {% endif %}
    {%- endif %}{# integrator_type #}

        // constraints
    {% if constraints.constr_type == "BGP" %}
        capsule->phi_constraint[stage].set_param_sparse(capsule->phi_constraint+stage, n_update, idx, p);
    {% elif constraints.constr_type == "BGH" and dims.nh > 0 %}
        capsule->nl_constr_h_fun_jac[stage].set_param_sparse(capsule->nl_constr_h_fun_jac+stage, n_update, idx, p);
        capsule->nl_constr_h_fun[stage].set_param_sparse(capsule->nl_constr_h_fun+stage, n_update, idx, p);
    {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->nl_constr_h_fun_jac_hess[stage].set_param_sparse(capsule->nl_constr_h_fun_jac_hess+stage, n_update, idx, p);
    {%- endif %}
    {%- endif %}

        // cost
        if (stage == 0)
        {
        {%- if cost.cost_type_0 == "NONLINEAR_LS" %}
            capsule->cost_y_0_fun.set_param_sparse(&capsule->cost_y_0_fun, n_update, idx, p);
            capsule->cost_y_0_fun_jac_ut_xt.set_param_sparse(&capsule->cost_y_0_fun_jac_ut_xt, n_update, idx, p);
            capsule->cost_y_0_hess.set_param_sparse(&capsule->cost_y_0_hess, n_update, idx, p);
        {%- elif cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}
            capsule->conl_cost_0_fun.set_param_sparse(&capsule->conl_cost_0_fun, n_update, idx, p);
            capsule->conl_cost_0_fun_jac_hess.set_param_sparse(&capsule->conl_cost_0_fun_jac_hess, n_update, idx, p);
        {%- elif cost.cost_type_0 == "EXTERNAL" %}
            capsule->ext_cost_0_fun.set_param_sparse(&capsule->ext_cost_0_fun, n_update, idx, p);
            capsule->ext_cost_0_fun_jac.set_param_sparse(&capsule->ext_cost_0_fun_jac, n_update, idx, p);
            capsule->ext_cost_0_fun_jac_hess.set_param_sparse(&capsule->ext_cost_0_fun_jac_hess, n_update, idx, p);
        {% endif %}
        }
        else // 0 < stage < N
        {
        {%- if cost.cost_type == "NONLINEAR_LS" %}
            capsule->cost_y_fun[stage-1].set_param_sparse(capsule->cost_y_fun+stage-1, n_update, idx, p);
            capsule->cost_y_fun_jac_ut_xt[stage-1].set_param_sparse(capsule->cost_y_fun_jac_ut_xt+stage-1, n_update, idx, p);
            capsule->cost_y_hess[stage-1].set_param_sparse(capsule->cost_y_hess+stage-1, n_update, idx, p);
        {%- elif cost.cost_type == "CONVEX_OVER_NONLINEAR" %}
            capsule->conl_cost_fun[stage-1].set_param_sparse(capsule->conl_cost_fun+stage-1, n_update, idx, p);
            capsule->conl_cost_fun_jac_hess[stage-1].set_param_sparse(capsule->conl_cost_fun_jac_hess+stage-1, n_update, idx, p);
        {%- elif cost.cost_type == "EXTERNAL" %}
            capsule->ext_cost_fun[stage-1].set_param_sparse(capsule->ext_cost_fun+stage-1, n_update, idx, p);
            capsule->ext_cost_fun_jac[stage-1].set_param_sparse(capsule->ext_cost_fun_jac+stage-1, n_update, idx, p);
            capsule->ext_cost_fun_jac_hess[stage-1].set_param_sparse(capsule->ext_cost_fun_jac_hess+stage-1, n_update, idx, p);
        {%- endif %}
        }
    }

    else // stage == N
    {
        // terminal shooting node has no dynamics
        // cost
    {%- if cost.cost_type_e == "NONLINEAR_LS" %}
        capsule->cost_y_e_fun.set_param_sparse(&capsule->cost_y_e_fun, n_update, idx, p);
        capsule->cost_y_e_fun_jac_ut_xt.set_param_sparse(&capsule->cost_y_e_fun_jac_ut_xt, n_update, idx, p);
        capsule->cost_y_e_hess.set_param_sparse(&capsule->cost_y_e_hess, n_update, idx, p);
    {%- elif cost.cost_type_e == "CONVEX_OVER_NONLINEAR" %}
        capsule->conl_cost_e_fun.set_param_sparse(&capsule->conl_cost_e_fun, n_update, idx, p);
        capsule->conl_cost_e_fun_jac_hess.set_param_sparse(&capsule->conl_cost_e_fun_jac_hess, n_update, idx, p);
    {%- elif cost.cost_type_e == "EXTERNAL" %}
        capsule->ext_cost_e_fun.set_param_sparse(&capsule->ext_cost_e_fun, n_update, idx, p);
        capsule->ext_cost_e_fun_jac.set_param_sparse(&capsule->ext_cost_e_fun_jac, n_update, idx, p);
        capsule->ext_cost_e_fun_jac_hess.set_param_sparse(&capsule->ext_cost_e_fun_jac_hess, n_update, idx, p);
    {% endif %}
        // constraints
    {% if constraints.constr_type_e == "BGP" %}
        capsule->phi_e_constraint.set_param_sparse(&capsule->phi_e_constraint, n_update, idx, p);
    {% elif constraints.constr_type_e == "BGH" and dims.nh_e > 0 %}
        capsule->nl_constr_h_e_fun_jac.set_param_sparse(&capsule->nl_constr_h_e_fun_jac, n_update, idx, p);
        capsule->nl_constr_h_e_fun.set_param_sparse(&capsule->nl_constr_h_e_fun, n_update, idx, p);
    {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->nl_constr_h_e_fun_jac_hess.set_param_sparse(&capsule->nl_constr_h_e_fun_jac_hess, n_update, idx, p);
    {%- endif %}
    {% endif %}
    }
{% endif %}{# if dims.np #}

    return solver_status;
}

int {{ model.name }}_acados_solve({{ model.name }}_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


int {{ model.name }}_acados_free({{ model.name }}_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    {%- if custom_update_filename != "" %}
    custom_update_terminate_function(capsule);
    {%- endif %}
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
{%- if solver_options.integrator_type == "IRK" %}
    for (int i = 0; i < N; i++)
    {
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->impl_dae_fun[i]);
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->impl_dae_fun_jac_x_xdot_z[i]);
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->impl_dae_jac_x_xdot_u_z[i]);
    {%- if solver_options.hessian_approx == "EXACT" %}
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->impl_dae_hess[i]);
    {%- endif %}
    }
    free(capsule->impl_dae_fun);
    free(capsule->impl_dae_fun_jac_x_xdot_z);
    free(capsule->impl_dae_jac_x_xdot_u_z);
    {%- if solver_options.hessian_approx == "EXACT" %}
    free(capsule->impl_dae_hess);
    {%- endif %}

{%- elif solver_options.integrator_type == "LIFTED_IRK" %}
    for (int i = 0; i < N; i++)
    {
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->impl_dae_fun[i]);
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->impl_dae_fun_jac_x_xdot_u[i]);
    }
    free(capsule->impl_dae_fun);
    free(capsule->impl_dae_fun_jac_x_xdot_u);

{%- elif solver_options.integrator_type == "ERK" %}
    for (int i = 0; i < N; i++)
    {
        external_function_param_casadi_free(&capsule->forw_vde_casadi[i]);
        external_function_param_casadi_free(&capsule->expl_ode_fun[i]);
    {%- if solver_options.hessian_approx == "EXACT" %}
        external_function_param_casadi_free(&capsule->hess_vde_casadi[i]);
    {%- endif %}
    }
    free(capsule->forw_vde_casadi);
    free(capsule->expl_ode_fun);
    {%- if solver_options.hessian_approx == "EXACT" %}
    free(capsule->hess_vde_casadi);
    {%- endif %}

{%- elif solver_options.integrator_type == "GNSF" %}
    for (int i = 0; i < N; i++)
    {
        {% if model.gnsf.purely_linear != 1 %}
        external_function_param_casadi_free(&capsule->gnsf_phi_fun[i]);
        external_function_param_casadi_free(&capsule->gnsf_phi_fun_jac_y[i]);
        external_function_param_casadi_free(&capsule->gnsf_phi_jac_y_uhat[i]);
        {% if model.gnsf.nontrivial_f_LO == 1 %}
        external_function_param_casadi_free(&capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i]);
        {%- endif %}
        {%- endif %}
        external_function_param_casadi_free(&capsule->gnsf_get_matrices_fun[i]);
    }
  {% if model.gnsf.purely_linear != 1 %}
    free(capsule->gnsf_phi_fun);
    free(capsule->gnsf_phi_fun_jac_y);
    free(capsule->gnsf_phi_jac_y_uhat);
  {% if model.gnsf.nontrivial_f_LO == 1 %}
    free(capsule->gnsf_f_lo_jac_x1_x1dot_u_z);
  {%- endif %}
  {%- endif %}
    free(capsule->gnsf_get_matrices_fun);
{%- elif solver_options.integrator_type == "DISCRETE" %}
    for (int i = 0; i < N; i++)
    {
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->discr_dyn_phi_fun[i]);
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
    {%- if solver_options.hessian_approx == "EXACT" %}
        external_function_param_{{ model.dyn_ext_fun_type }}_free(&capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i]);
    {%- endif %}
    }
    free(capsule->discr_dyn_phi_fun);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt);
    {%- if solver_options.hessian_approx == "EXACT" %}
    free(capsule->discr_dyn_phi_fun_jac_ut_xt_hess);
    {%- endif %}

{%- endif %}

    // cost
{%- if cost.cost_type_0 == "NONLINEAR_LS" %}
    external_function_param_casadi_free(&capsule->cost_y_0_fun);
    external_function_param_casadi_free(&capsule->cost_y_0_fun_jac_ut_xt);
    external_function_param_casadi_free(&capsule->cost_y_0_hess);
{%- elif cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}
    external_function_param_casadi_free(&capsule->conl_cost_0_fun);
    external_function_param_casadi_free(&capsule->conl_cost_0_fun_jac_hess);
{%- elif cost.cost_type_0 == "EXTERNAL" %}
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_free(&capsule->ext_cost_0_fun);
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_free(&capsule->ext_cost_0_fun_jac);
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_free(&capsule->ext_cost_0_fun_jac_hess);
{%- endif %}
{%- if cost.cost_type == "NONLINEAR_LS" %}
    for (int i = 0; i < N - 1; i++)
    {
        external_function_param_casadi_free(&capsule->cost_y_fun[i]);
        external_function_param_casadi_free(&capsule->cost_y_fun_jac_ut_xt[i]);
        external_function_param_casadi_free(&capsule->cost_y_hess[i]);
    }
    free(capsule->cost_y_fun);
    free(capsule->cost_y_fun_jac_ut_xt);
    free(capsule->cost_y_hess);
{%- elif cost.cost_type == "CONVEX_OVER_NONLINEAR" %}
    for (int i = 0; i < N - 1; i++)
    {
        external_function_param_casadi_free(&capsule->conl_cost_fun[i]);
        external_function_param_casadi_free(&capsule->conl_cost_fun_jac_hess[i]);
    }
    free(capsule->conl_cost_fun);
    free(capsule->conl_cost_fun_jac_hess);
{%- elif cost.cost_type == "EXTERNAL" %}
    for (int i = 0; i < N - 1; i++)
    {
        external_function_param_{{ cost.cost_ext_fun_type }}_free(&capsule->ext_cost_fun[i]);
        external_function_param_{{ cost.cost_ext_fun_type }}_free(&capsule->ext_cost_fun_jac[i]);
        external_function_param_{{ cost.cost_ext_fun_type }}_free(&capsule->ext_cost_fun_jac_hess[i]);
    }
    free(capsule->ext_cost_fun);
    free(capsule->ext_cost_fun_jac);
    free(capsule->ext_cost_fun_jac_hess);
{%- endif %}
{%- if cost.cost_type_e == "NONLINEAR_LS" %}
    external_function_param_casadi_free(&capsule->cost_y_e_fun);
    external_function_param_casadi_free(&capsule->cost_y_e_fun_jac_ut_xt);
    external_function_param_casadi_free(&capsule->cost_y_e_hess);
{%- elif cost.cost_type_e == "CONVEX_OVER_NONLINEAR" %}
    external_function_param_casadi_free(&capsule->conl_cost_e_fun);
    external_function_param_casadi_free(&capsule->conl_cost_e_fun_jac_hess);
{%- elif cost.cost_type_e == "EXTERNAL" %}
    external_function_param_{{ cost.cost_ext_fun_type_e }}_free(&capsule->ext_cost_e_fun);
    external_function_param_{{ cost.cost_ext_fun_type_e }}_free(&capsule->ext_cost_e_fun_jac);
    external_function_param_{{ cost.cost_ext_fun_type_e }}_free(&capsule->ext_cost_e_fun_jac_hess);
{%- endif %}

    // constraints
{%- if constraints.constr_type == "BGH" and dims.nh > 0 %}
    for (int i = 0; i < N; i++)
    {
        external_function_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
  {%- if solver_options.hessian_approx == "EXACT" %}
    for (int i = 0; i < N; i++)
    {
        external_function_param_casadi_free(&capsule->nl_constr_h_fun_jac_hess[i]);
    }
  {%- endif %}
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);
  {%- if solver_options.hessian_approx == "EXACT" %}
    free(capsule->nl_constr_h_fun_jac_hess);
  {%- endif %}

{%- elif constraints.constr_type == "BGP" and dims.nphi > 0 %}
    for (int i = 0; i < N; i++)
    {
        external_function_param_casadi_free(&capsule->phi_constraint[i]);
    }
    free(capsule->phi_constraint);
{%- endif %}

{%- if constraints.constr_type_e == "BGH" and dims.nh_e > 0 %}
    external_function_param_casadi_free(&capsule->nl_constr_h_e_fun_jac);
    external_function_param_casadi_free(&capsule->nl_constr_h_e_fun);
{%- if solver_options.hessian_approx == "EXACT" %}
    external_function_param_casadi_free(&capsule->nl_constr_h_e_fun_jac_hess);
{%- endif %}
{%- elif constraints.constr_type_e == "BGP" and dims.nphi_e > 0 %}
    external_function_param_casadi_free(&capsule->phi_e_constraint);
{%- endif %}

    return 0;
}


void {{ model.name }}_acados_print_stats({{ model.name }}_solver_capsule* capsule)
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_m", &stat_m);

    {% set stat_n_max = 12 %}
    double stat[{{ solver_options.nlp_solver_max_iter * stat_n_max }}];
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "statistics", stat);

    int nrow = sqp_iter+1 < stat_m ? sqp_iter+1 : stat_m;

    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");

{%- if solver_options.nlp_solver_type == "SQP" %}

    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j == 5 || j == 6)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
{% else %}
    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
{%- endif %}
}

int {{ model.name }}_acados_custom_update({{ model.name }}_solver_capsule* capsule, double* data, int data_len)
{
{%- if custom_update_filename == "" %}
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;
{% else %}
    custom_update_function(capsule, data, data_len);
{%- endif %}
}



ocp_nlp_in *{{ model.name }}_acados_get_nlp_in({{ model.name }}_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *{{ model.name }}_acados_get_nlp_out({{ model.name }}_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *{{ model.name }}_acados_get_sens_out({{ model.name }}_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *{{ model.name }}_acados_get_nlp_solver({{ model.name }}_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *{{ model.name }}_acados_get_nlp_config({{ model.name }}_solver_capsule* capsule) { return capsule->nlp_config; }
void *{{ model.name }}_acados_get_nlp_opts({{ model.name }}_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *{{ model.name }}_acados_get_nlp_dims({{ model.name }}_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *{{ model.name }}_acados_get_nlp_plan({{ model.name }}_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
