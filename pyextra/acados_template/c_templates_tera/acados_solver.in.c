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

// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "{{ model.name }}_model/{{ model.name }}_model.h"
{% if constraints.constr_type == "BGP" and dims.nphi %}
#include "{{ model.name }}_constraints/{{ model.name }}_phi_constraint.h"
{% endif %}
{% if constraints.constr_type_e == "BGP" and dims.nphi_e > 0 %}
#include "{{ model.name }}_constraints/{{ model.name }}_phi_e_constraint.h"
{% endif %}
{% if constraints.constr_type == "BGH" and dims.nh > 0 %}
#include "{{ model.name }}_constraints/{{ model.name }}_h_constraint.h"
{% endif %}
{% if constraints.constr_type_e == "BGH" and dims.nh_e > 0 %}
#include "{{ model.name }}_constraints/{{ model.name }}_h_e_constraint.h"
{% endif %}
{%- if cost.cost_type == "NONLINEAR_LS" %}
#include "{{ model.name }}_cost/{{ model.name }}_cost_y_fun.h"
{%- elif cost.cost_type == "EXTERNAL" %}
#include "{{ model.name }}_cost/{{ model.name }}_external_cost.h"
{%- endif %}
{%- if cost.cost_type_0 == "NONLINEAR_LS" %}
#include "{{ model.name }}_cost/{{ model.name }}_cost_y_0_fun.h"
{%- elif cost.cost_type_0 == "EXTERNAL" %}
#include "{{ model.name }}_cost/{{ model.name }}_external_cost_0.h"
{%- endif %}
{%- if cost.cost_type_e == "NONLINEAR_LS" %}
#include "{{ model.name }}_cost/{{ model.name }}_cost_y_e_fun.h"
{%- elif cost.cost_type_e == "EXTERNAL" %}
#include "{{ model.name }}_cost/{{ model.name }}_external_cost_e.h"
{%- endif %}

#include "acados_solver_{{ model.name }}.h"

#define NX     {{ dims.nx }}
#define NZ     {{ dims.nz }}
#define NU     {{ dims.nu }}
#define NP     {{ dims.np }}
#define NBX    {{ dims.nbx }}
#define NBX0   {{ dims.nbx_0 }}
#define NBU    {{ dims.nbu }}
#define NSBX   {{ dims.nsbx }}
#define NSBU   {{ dims.nsbu }}
#define NSH    {{ dims.nsh }}
#define NSG    {{ dims.nsg }}
#define NSPHI  {{ dims.nsphi }}
#define NSHN   {{ dims.nsh_e }}
#define NSGN   {{ dims.nsg_e }}
#define NSPHIN {{ dims.nsphi_e }}
#define NSBXN  {{ dims.nsbx_e }}
#define NS     {{ dims.ns }}
#define NSN    {{ dims.ns_e }}
#define NG     {{ dims.ng }}
#define NBXN   {{ dims.nbx_e }}
#define NGN    {{ dims.ng_e }}
#define NY0    {{ dims.ny_0 }}
#define NY     {{ dims.ny }}
#define NYN    {{ dims.ny_e }}
#define N      {{ dims.N }}
#define NH     {{ dims.nh }}
#define NPHI   {{ dims.nphi }}
#define NHN    {{ dims.nh_e }}
#define NPHIN  {{ dims.nphi_e }}
#define NR     {{ dims.nr }}


// ** solver data **

nlp_solver_capsule * {{ model.name }}_acados_create_capsule()
{
    void* capsule_mem = malloc(sizeof(nlp_solver_capsule));
    nlp_solver_capsule *capsule = (nlp_solver_capsule *) capsule_mem;

    return capsule;
}


int {{ model.name }}_acados_free_capsule(nlp_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int {{ model.name }}_acados_create(nlp_solver_capsule * capsule)
{
    int status = 0;

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    /************************************************
    *  plan & config
    ************************************************/
    ocp_nlp_plan * nlp_solver_plan = ocp_nlp_plan_create(N);
    capsule->nlp_solver_plan = nlp_solver_plan;

    {%- if solver_options.nlp_solver_type == "SQP" %}
    nlp_solver_plan->nlp_solver = SQP;
    {% else %}
    nlp_solver_plan->nlp_solver = SQP_RTI;
    {%- endif %}

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = {{ solver_options.qp_solver }};

    nlp_solver_plan->nlp_cost[0] = {{ cost.cost_type_0 }};
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = {{ cost.cost_type }};

    nlp_solver_plan->nlp_cost[N] = {{ cost.cost_type_e }};

    for (int i = 0; i < N; i++)
    {
        {% if solver_options.integrator_type == "DISCRETE" %}
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
        {% else %}
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = {{ solver_options.integrator_type }};
        {%- endif %}
    }

    for (int i = 0; i < N; i++)
    {
        {% if constraints.constr_type == "BGP" %}
        nlp_solver_plan->nlp_constraints[i] = BGP;
        {%- else -%}
        nlp_solver_plan->nlp_constraints[i] = BGH;
        {%- endif %}
    }

    {%- if constraints.constr_type_e == "BGP" %}
    nlp_solver_plan->nlp_constraints[N] = BGP;
    {% else %}
    nlp_solver_plan->nlp_constraints[N] = BGH;
    {%- endif %}

{%- if solver_options.hessian_approx == "EXACT" %}
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
{%- endif %}
    ocp_nlp_config * nlp_config = ocp_nlp_config_create(*nlp_solver_plan);
    capsule->nlp_config = nlp_config;


    /************************************************
    *  dimensions
    ************************************************/
    int nx[N+1];
    int nu[N+1];
    int nbx[N+1];
    int nbu[N+1];
    int nsbx[N+1];
    int nsbu[N+1];
    int nsg[N+1];
    int nsh[N+1];
    int nsphi[N+1];
    int ns[N+1];
    int ng[N+1];
    int nh[N+1];
    int nphi[N+1];
    int nz[N+1];
    int ny[N+1];
    int nr[N+1];
    int nbxe[N+1];

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
        nsg[i] = NSG;
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
    capsule->nlp_dims = nlp_dims;

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

    {%- if cost.cost_type_0 == "NONLINEAR_LS" or cost.cost_type_0 == "LINEAR_LS" %}
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    {%- endif %}

    {%- if cost.cost_type == "NONLINEAR_LS" or cost.cost_type == "LINEAR_LS" %}
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
    {%- if cost.cost_type_e == "NONLINEAR_LS" or cost.cost_type_e == "LINEAR_LS" %}
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    {%- endif %}

{% if solver_options.integrator_type == "GNSF" -%}
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

    /************************************************
    *  external functions
    ************************************************/
    {%- if constraints.constr_type == "BGP" %}
    capsule->phi_constraint = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        // nonlinear part of convex-composite constraint
        capsule->phi_constraint[i].casadi_fun = &{{ model.name }}_phi_constraint;
        capsule->phi_constraint[i].casadi_n_in = &{{ model.name }}_phi_constraint_n_in;
        capsule->phi_constraint[i].casadi_n_out = &{{ model.name }}_phi_constraint_n_out;
        capsule->phi_constraint[i].casadi_sparsity_in = &{{ model.name }}_phi_constraint_sparsity_in;
        capsule->phi_constraint[i].casadi_sparsity_out = &{{ model.name }}_phi_constraint_sparsity_out;
        capsule->phi_constraint[i].casadi_work = &{{ model.name }}_phi_constraint_work;

        external_function_param_casadi_create(&capsule->phi_constraint[i], {{ dims.np }});
    }
    {%- endif %}

    {%- if constraints.constr_type_e == "BGP" %}
    // nonlinear part of convex-composite constraint
    capsule->phi_e_constraint.casadi_fun = &{{ model.name }}_phi_e_constraint;
    capsule->phi_e_constraint.casadi_n_in = &{{ model.name }}_phi_e_constraint_n_in;
    capsule->phi_e_constraint.casadi_n_out = &{{ model.name }}_phi_e_constraint_n_out;
    capsule->phi_e_constraint.casadi_sparsity_in = &{{ model.name }}_phi_e_constraint_sparsity_in;
    capsule->phi_e_constraint.casadi_sparsity_out = &{{ model.name }}_phi_e_constraint_sparsity_out;
    capsule->phi_e_constraint.casadi_work = &{{ model.name }}_phi_e_constraint_work;

    external_function_param_casadi_create(&capsule->phi_e_constraint, {{ dims.np }});
    {% endif %}

    {%- if constraints.constr_type == "BGH" and dims.nh > 0  %}
    capsule->nl_constr_h_fun_jac = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->nl_constr_h_fun_jac[i].casadi_fun = &{{ model.name }}_constr_h_fun_jac_uxt_zt;
        capsule->nl_constr_h_fun_jac[i].casadi_n_in = &{{ model.name }}_constr_h_fun_jac_uxt_zt_n_in;
        capsule->nl_constr_h_fun_jac[i].casadi_n_out = &{{ model.name }}_constr_h_fun_jac_uxt_zt_n_out;
        capsule->nl_constr_h_fun_jac[i].casadi_sparsity_in = &{{ model.name }}_constr_h_fun_jac_uxt_zt_sparsity_in;
        capsule->nl_constr_h_fun_jac[i].casadi_sparsity_out = &{{ model.name }}_constr_h_fun_jac_uxt_zt_sparsity_out;
        capsule->nl_constr_h_fun_jac[i].casadi_work = &{{ model.name }}_constr_h_fun_jac_uxt_zt_work;
        external_function_param_casadi_create(&capsule->nl_constr_h_fun_jac[i], {{ dims.np }});
    }
    capsule->nl_constr_h_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->nl_constr_h_fun[i].casadi_fun = &{{ model.name }}_constr_h_fun;
        capsule->nl_constr_h_fun[i].casadi_n_in = &{{ model.name }}_constr_h_fun_n_in;
        capsule->nl_constr_h_fun[i].casadi_n_out = &{{ model.name }}_constr_h_fun_n_out;
        capsule->nl_constr_h_fun[i].casadi_sparsity_in = &{{ model.name }}_constr_h_fun_sparsity_in;
        capsule->nl_constr_h_fun[i].casadi_sparsity_out = &{{ model.name }}_constr_h_fun_sparsity_out;
        capsule->nl_constr_h_fun[i].casadi_work = &{{ model.name }}_constr_h_fun_work;
        external_function_param_casadi_create(&capsule->nl_constr_h_fun[i], {{ dims.np }});
    }
    {% if solver_options.hessian_approx == "EXACT" %}
    capsule->nl_constr_h_fun_jac_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->nl_constr_h_fun_jac_hess[i].casadi_fun = &{{ model.name }}_constr_h_fun_jac_uxt_zt_hess;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_n_in = &{{ model.name }}_constr_h_fun_jac_uxt_zt_hess_n_in;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_n_out = &{{ model.name }}_constr_h_fun_jac_uxt_zt_hess_n_out;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_sparsity_in = &{{ model.name }}_constr_h_fun_jac_uxt_zt_hess_sparsity_in;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_sparsity_out = &{{ model.name }}_constr_h_fun_jac_uxt_zt_hess_sparsity_out;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_work = &{{ model.name }}_constr_h_fun_jac_uxt_zt_hess_work;

        external_function_param_casadi_create(&capsule->nl_constr_h_fun_jac_hess[i], {{ dims.np }});
    }
    {% endif %}
    {% endif %}

    {%- if constraints.constr_type_e == "BGH" and dims.nh_e > 0 %}
    capsule->nl_constr_h_e_fun_jac.casadi_fun = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt;
    capsule->nl_constr_h_e_fun_jac.casadi_n_in = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_n_in;
    capsule->nl_constr_h_e_fun_jac.casadi_n_out = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_n_out;
    capsule->nl_constr_h_e_fun_jac.casadi_sparsity_in = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_sparsity_in;
    capsule->nl_constr_h_e_fun_jac.casadi_sparsity_out = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_sparsity_out;
    capsule->nl_constr_h_e_fun_jac.casadi_work = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_work;
    external_function_param_casadi_create(&capsule->nl_constr_h_e_fun_jac, {{ dims.np }});

    capsule->nl_constr_h_e_fun.casadi_fun = &{{ model.name }}_constr_h_e_fun;
    capsule->nl_constr_h_e_fun.casadi_n_in = &{{ model.name }}_constr_h_e_fun_n_in;
    capsule->nl_constr_h_e_fun.casadi_n_out = &{{ model.name }}_constr_h_e_fun_n_out;
    capsule->nl_constr_h_e_fun.casadi_sparsity_in = &{{ model.name }}_constr_h_e_fun_sparsity_in;
    capsule->nl_constr_h_e_fun.casadi_sparsity_out = &{{ model.name }}_constr_h_e_fun_sparsity_out;
    capsule->nl_constr_h_e_fun.casadi_work = &{{ model.name }}_constr_h_e_fun_work;
    external_function_param_casadi_create(&capsule->nl_constr_h_e_fun, {{ dims.np }});

    {% if solver_options.hessian_approx == "EXACT" %}
    capsule->nl_constr_h_e_fun_jac_hess.casadi_fun = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_n_in = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_n_in;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_n_out = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_n_out;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_sparsity_in = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_sparsity_in;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_sparsity_out = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_sparsity_out;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_work = &{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_work;
    external_function_param_casadi_create(&capsule->nl_constr_h_e_fun_jac_hess, {{ dims.np }});
    {% endif %}
    {%- endif %}

{% if solver_options.integrator_type == "ERK" %}
    // explicit ode
    capsule->forw_vde_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->forw_vde_casadi[i].casadi_fun = &{{ model.name }}_expl_vde_forw;
        capsule->forw_vde_casadi[i].casadi_n_in = &{{ model.name }}_expl_vde_forw_n_in;
        capsule->forw_vde_casadi[i].casadi_n_out = &{{ model.name }}_expl_vde_forw_n_out;
        capsule->forw_vde_casadi[i].casadi_sparsity_in = &{{ model.name }}_expl_vde_forw_sparsity_in;
        capsule->forw_vde_casadi[i].casadi_sparsity_out = &{{ model.name }}_expl_vde_forw_sparsity_out;
        capsule->forw_vde_casadi[i].casadi_work = &{{ model.name }}_expl_vde_forw_work;
        external_function_param_casadi_create(&capsule->forw_vde_casadi[i], {{ dims.np }});
    }

    capsule->expl_ode_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->expl_ode_fun[i].casadi_fun = &{{ model.name }}_expl_ode_fun;
        capsule->expl_ode_fun[i].casadi_n_in = &{{ model.name }}_expl_ode_fun_n_in;
        capsule->expl_ode_fun[i].casadi_n_out = &{{ model.name }}_expl_ode_fun_n_out;
        capsule->expl_ode_fun[i].casadi_sparsity_in = &{{ model.name }}_expl_ode_fun_sparsity_in;
        capsule->expl_ode_fun[i].casadi_sparsity_out = &{{ model.name }}_expl_ode_fun_sparsity_out;
        capsule->expl_ode_fun[i].casadi_work = &{{ model.name }}_expl_ode_fun_work;
        external_function_param_casadi_create(&capsule->expl_ode_fun[i], {{ dims.np }});
    }

    {%- if solver_options.hessian_approx == "EXACT" %}
    capsule->hess_vde_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->hess_vde_casadi[i].casadi_fun = &{{ model.name }}_expl_ode_hess;
        capsule->hess_vde_casadi[i].casadi_n_in = &{{ model.name }}_expl_ode_hess_n_in;
        capsule->hess_vde_casadi[i].casadi_n_out = &{{ model.name }}_expl_ode_hess_n_out;
        capsule->hess_vde_casadi[i].casadi_sparsity_in = &{{ model.name }}_expl_ode_hess_sparsity_in;
        capsule->hess_vde_casadi[i].casadi_sparsity_out = &{{ model.name }}_expl_ode_hess_sparsity_out;
        capsule->hess_vde_casadi[i].casadi_work = &{{ model.name }}_expl_ode_hess_work;
        external_function_param_casadi_create(&capsule->hess_vde_casadi[i], {{ dims.np }});
    }
    {%- endif %}

{% elif solver_options.integrator_type == "IRK" %}
    // implicit dae
    capsule->impl_dae_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->impl_dae_fun[i].casadi_fun = &{{ model.name }}_impl_dae_fun;
        capsule->impl_dae_fun[i].casadi_work = &{{ model.name }}_impl_dae_fun_work;
        capsule->impl_dae_fun[i].casadi_sparsity_in = &{{ model.name }}_impl_dae_fun_sparsity_in;
        capsule->impl_dae_fun[i].casadi_sparsity_out = &{{ model.name }}_impl_dae_fun_sparsity_out;
        capsule->impl_dae_fun[i].casadi_n_in = &{{ model.name }}_impl_dae_fun_n_in;
        capsule->impl_dae_fun[i].casadi_n_out = &{{ model.name }}_impl_dae_fun_n_out;
        external_function_param_casadi_create(&capsule->impl_dae_fun[i], {{ dims.np }});
    }

    capsule->impl_dae_fun_jac_x_xdot_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->impl_dae_fun_jac_x_xdot_z[i].casadi_fun = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z;
        capsule->impl_dae_fun_jac_x_xdot_z[i].casadi_work = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_work;
        capsule->impl_dae_fun_jac_x_xdot_z[i].casadi_sparsity_in = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_sparsity_in;
        capsule->impl_dae_fun_jac_x_xdot_z[i].casadi_sparsity_out = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_sparsity_out;
        capsule->impl_dae_fun_jac_x_xdot_z[i].casadi_n_in = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_n_in;
        capsule->impl_dae_fun_jac_x_xdot_z[i].casadi_n_out = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_n_out;
        external_function_param_casadi_create(&capsule->impl_dae_fun_jac_x_xdot_z[i], {{ dims.np }});
    }

    capsule->impl_dae_jac_x_xdot_u_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->impl_dae_jac_x_xdot_u_z[i].casadi_fun = &{{ model.name }}_impl_dae_jac_x_xdot_u_z;
        capsule->impl_dae_jac_x_xdot_u_z[i].casadi_work = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_work;
        capsule->impl_dae_jac_x_xdot_u_z[i].casadi_sparsity_in = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_sparsity_in;
        capsule->impl_dae_jac_x_xdot_u_z[i].casadi_sparsity_out = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_sparsity_out;
        capsule->impl_dae_jac_x_xdot_u_z[i].casadi_n_in = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_n_in;
        capsule->impl_dae_jac_x_xdot_u_z[i].casadi_n_out = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_n_out;
        external_function_param_casadi_create(&capsule->impl_dae_jac_x_xdot_u_z[i], {{ dims.np }});
    }

    {%- if solver_options.hessian_approx == "EXACT" %}
    capsule->impl_dae_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->impl_dae_hess[i].casadi_fun = &{{ model.name }}_impl_dae_hess;
        capsule->impl_dae_hess[i].casadi_work = &{{ model.name }}_impl_dae_hess_work;
        capsule->impl_dae_hess[i].casadi_sparsity_in = &{{ model.name }}_impl_dae_hess_sparsity_in;
        capsule->impl_dae_hess[i].casadi_sparsity_out = &{{ model.name }}_impl_dae_hess_sparsity_out;
        capsule->impl_dae_hess[i].casadi_n_in = &{{ model.name }}_impl_dae_hess_n_in;
        capsule->impl_dae_hess[i].casadi_n_out = &{{ model.name }}_impl_dae_hess_n_out;
        external_function_param_casadi_create(&capsule->impl_dae_hess[i], {{ dims.np }});
    }
    {%- endif %}

{% elif solver_options.integrator_type == "GNSF" %}
    capsule->gnsf_phi_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->gnsf_phi_fun[i].casadi_fun = &{{ model.name }}_gnsf_phi_fun;
        capsule->gnsf_phi_fun[i].casadi_work = &{{ model.name }}_gnsf_phi_fun_work;
        capsule->gnsf_phi_fun[i].casadi_sparsity_in = &{{ model.name }}_gnsf_phi_fun_sparsity_in;
        capsule->gnsf_phi_fun[i].casadi_sparsity_out = &{{ model.name }}_gnsf_phi_fun_sparsity_out;
        capsule->gnsf_phi_fun[i].casadi_n_in = &{{ model.name }}_gnsf_phi_fun_n_in;
        capsule->gnsf_phi_fun[i].casadi_n_out = &{{ model.name }}_gnsf_phi_fun_n_out;
        external_function_param_casadi_create(&capsule->gnsf_phi_fun[i], {{ dims.np }});
    }

    capsule->gnsf_phi_fun_jac_y = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->gnsf_phi_fun_jac_y[i].casadi_fun = &{{ model.name }}_gnsf_phi_fun_jac_y;
        capsule->gnsf_phi_fun_jac_y[i].casadi_work = &{{ model.name }}_gnsf_phi_fun_jac_y_work;
        capsule->gnsf_phi_fun_jac_y[i].casadi_sparsity_in = &{{ model.name }}_gnsf_phi_fun_jac_y_sparsity_in;
        capsule->gnsf_phi_fun_jac_y[i].casadi_sparsity_out = &{{ model.name }}_gnsf_phi_fun_jac_y_sparsity_out;
        capsule->gnsf_phi_fun_jac_y[i].casadi_n_in = &{{ model.name }}_gnsf_phi_fun_jac_y_n_in;
        capsule->gnsf_phi_fun_jac_y[i].casadi_n_out = &{{ model.name }}_gnsf_phi_fun_jac_y_n_out;
        external_function_param_casadi_create(&capsule->gnsf_phi_fun_jac_y[i], {{ dims.np }});
    }

    capsule->gnsf_phi_jac_y_uhat = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->gnsf_phi_jac_y_uhat[i].casadi_fun = &{{ model.name }}_gnsf_phi_jac_y_uhat;
        capsule->gnsf_phi_jac_y_uhat[i].casadi_work = &{{ model.name }}_gnsf_phi_jac_y_uhat_work;
        capsule->gnsf_phi_jac_y_uhat[i].casadi_sparsity_in = &{{ model.name }}_gnsf_phi_jac_y_uhat_sparsity_in;
        capsule->gnsf_phi_jac_y_uhat[i].casadi_sparsity_out = &{{ model.name }}_gnsf_phi_jac_y_uhat_sparsity_out;
        capsule->gnsf_phi_jac_y_uhat[i].casadi_n_in = &{{ model.name }}_gnsf_phi_jac_y_uhat_n_in;
        capsule->gnsf_phi_jac_y_uhat[i].casadi_n_out = &{{ model.name }}_gnsf_phi_jac_y_uhat_n_out;
        external_function_param_casadi_create(&capsule->gnsf_phi_jac_y_uhat[i], {{ dims.np }});
    }

    capsule->gnsf_f_lo_jac_x1_x1dot_u_z = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i].casadi_fun = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz;
        capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i].casadi_work = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_work;
        capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i].casadi_sparsity_in = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_sparsity_in;
        capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i].casadi_sparsity_out = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_sparsity_out;
        capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i].casadi_n_in = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_n_in;
        capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i].casadi_n_out = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_n_out;
        external_function_param_casadi_create(&capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i], {{ dims.np }});
    }

    capsule->gnsf_get_matrices_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->gnsf_get_matrices_fun[i].casadi_fun = &{{ model.name }}_gnsf_get_matrices_fun;
        capsule->gnsf_get_matrices_fun[i].casadi_work = &{{ model.name }}_gnsf_get_matrices_fun_work;
        capsule->gnsf_get_matrices_fun[i].casadi_sparsity_in = &{{ model.name }}_gnsf_get_matrices_fun_sparsity_in;
        capsule->gnsf_get_matrices_fun[i].casadi_sparsity_out = &{{ model.name }}_gnsf_get_matrices_fun_sparsity_out;
        capsule->gnsf_get_matrices_fun[i].casadi_n_in = &{{ model.name }}_gnsf_get_matrices_fun_n_in;
        capsule->gnsf_get_matrices_fun[i].casadi_n_out = &{{ model.name }}_gnsf_get_matrices_fun_n_out;
        external_function_param_casadi_create(&capsule->gnsf_get_matrices_fun[i], {{ dims.np }});
    }
{% elif solver_options.integrator_type == "DISCRETE" %}
    // discrete dynamics
    capsule->discr_dyn_phi_fun = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++)
    {
        {%- if model.dyn_ext_fun_type == "casadi" %}
        capsule->discr_dyn_phi_fun[i].casadi_fun = &{{ model.name }}_dyn_disc_phi_fun;
        capsule->discr_dyn_phi_fun[i].casadi_n_in = &{{ model.name }}_dyn_disc_phi_fun_n_in;
        capsule->discr_dyn_phi_fun[i].casadi_n_out = &{{ model.name }}_dyn_disc_phi_fun_n_out;
        capsule->discr_dyn_phi_fun[i].casadi_sparsity_in = &{{ model.name }}_dyn_disc_phi_fun_sparsity_in;
        capsule->discr_dyn_phi_fun[i].casadi_sparsity_out = &{{ model.name }}_dyn_disc_phi_fun_sparsity_out;
        capsule->discr_dyn_phi_fun[i].casadi_work = &{{ model.name }}_dyn_disc_phi_fun_work;
        {%- else %}
        capsule->discr_dyn_phi_fun[i].fun = &{{ model.dyn_disc_fun }};
        {%- endif %}
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->discr_dyn_phi_fun[i], {{ dims.np }});
    }

    capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++)
    {
        {%- if model.dyn_ext_fun_type == "casadi" %}
        capsule->discr_dyn_phi_fun_jac_ut_xt[i].casadi_fun = &{{ model.name }}_dyn_disc_phi_fun_jac;
        capsule->discr_dyn_phi_fun_jac_ut_xt[i].casadi_n_in = &{{ model.name }}_dyn_disc_phi_fun_jac_n_in;
        capsule->discr_dyn_phi_fun_jac_ut_xt[i].casadi_n_out = &{{ model.name }}_dyn_disc_phi_fun_jac_n_out;
        capsule->discr_dyn_phi_fun_jac_ut_xt[i].casadi_sparsity_in = &{{ model.name }}_dyn_disc_phi_fun_jac_sparsity_in;
        capsule->discr_dyn_phi_fun_jac_ut_xt[i].casadi_sparsity_out = &{{ model.name }}_dyn_disc_phi_fun_jac_sparsity_out;
        capsule->discr_dyn_phi_fun_jac_ut_xt[i].casadi_work = &{{ model.name }}_dyn_disc_phi_fun_jac_work;
        {%- else %}
        capsule->discr_dyn_phi_fun_jac_ut_xt[i].fun = &{{ model.dyn_disc_fun_jac }};
        {%- endif %}
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->discr_dyn_phi_fun_jac_ut_xt[i], {{ dims.np }});
    }

  {%- if solver_options.hessian_approx == "EXACT" %}
    capsule->discr_dyn_phi_fun_jac_ut_xt_hess = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }})*N);
    for (int i = 0; i < N; i++)
    {
        {%- if model.dyn_ext_fun_type == "casadi" %}
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i].casadi_fun = &{{ model.name }}_dyn_disc_phi_fun_jac_hess;
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i].casadi_n_in = &{{ model.name }}_dyn_disc_phi_fun_jac_hess_n_in;
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i].casadi_n_out = &{{ model.name }}_dyn_disc_phi_fun_jac_hess_n_out;
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i].casadi_sparsity_in = &{{ model.name }}_dyn_disc_phi_fun_jac_hess_sparsity_in;
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i].casadi_sparsity_out = &{{ model.name }}_dyn_disc_phi_fun_jac_hess_sparsity_out;
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i].casadi_work = &{{ model.name }}_dyn_disc_phi_fun_jac_hess_work;
        {%- else %}
        capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i].fun = &{{ model.dyn_disc_fun_jac_hess }};
        {%- endif %}
        external_function_param_{{ model.dyn_ext_fun_type }}_create(&capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i], {{ dims.np }});
    }
  {%- endif %}
{%- endif %}


{%- if cost.cost_type_0 == "NONLINEAR_LS" %}
    // nonlinear least square function
    capsule->cost_y_0_fun.casadi_fun = &{{ model.name }}_cost_y_0_fun;
    capsule->cost_y_0_fun.casadi_n_in = &{{ model.name }}_cost_y_0_fun_n_in;
    capsule->cost_y_0_fun.casadi_n_out = &{{ model.name }}_cost_y_0_fun_n_out;
    capsule->cost_y_0_fun.casadi_sparsity_in = &{{ model.name }}_cost_y_0_fun_sparsity_in;
    capsule->cost_y_0_fun.casadi_sparsity_out = &{{ model.name }}_cost_y_0_fun_sparsity_out;
    capsule->cost_y_0_fun.casadi_work = &{{ model.name }}_cost_y_0_fun_work;
    external_function_param_casadi_create(&capsule->cost_y_0_fun, {{ dims.np }});

    capsule->cost_y_0_fun_jac_ut_xt.casadi_fun = &{{ model.name }}_cost_y_0_fun_jac_ut_xt;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_n_in = &{{ model.name }}_cost_y_0_fun_jac_ut_xt_n_in;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_n_out = &{{ model.name }}_cost_y_0_fun_jac_ut_xt_n_out;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_sparsity_in = &{{ model.name }}_cost_y_0_fun_jac_ut_xt_sparsity_in;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_sparsity_out = &{{ model.name }}_cost_y_0_fun_jac_ut_xt_sparsity_out;
    capsule->cost_y_0_fun_jac_ut_xt.casadi_work = &{{ model.name }}_cost_y_0_fun_jac_ut_xt_work;
    external_function_param_casadi_create(&capsule->cost_y_0_fun_jac_ut_xt, {{ dims.np }});

    capsule->cost_y_0_hess.casadi_fun = &{{ model.name }}_cost_y_0_hess;
    capsule->cost_y_0_hess.casadi_n_in = &{{ model.name }}_cost_y_0_hess_n_in;
    capsule->cost_y_0_hess.casadi_n_out = &{{ model.name }}_cost_y_0_hess_n_out;
    capsule->cost_y_0_hess.casadi_sparsity_in = &{{ model.name }}_cost_y_0_hess_sparsity_in;
    capsule->cost_y_0_hess.casadi_sparsity_out = &{{ model.name }}_cost_y_0_hess_sparsity_out;
    capsule->cost_y_0_hess.casadi_work = &{{ model.name }}_cost_y_0_hess_work;
    external_function_param_casadi_create(&capsule->cost_y_0_hess, {{ dims.np }});

{%- elif cost.cost_type_0 == "EXTERNAL" %}
    // external cost
    {% if cost.cost_ext_fun_type_0 == "casadi" %}
    capsule->ext_cost_0_fun.casadi_fun = &{{ model.name }}_cost_ext_cost_0_fun;
    capsule->ext_cost_0_fun.casadi_n_in = &{{ model.name }}_cost_ext_cost_0_fun_n_in;
    capsule->ext_cost_0_fun.casadi_n_out = &{{ model.name }}_cost_ext_cost_0_fun_n_out;
    capsule->ext_cost_0_fun.casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_0_fun_sparsity_in;
    capsule->ext_cost_0_fun.casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_0_fun_sparsity_out;
    capsule->ext_cost_0_fun.casadi_work = &{{ model.name }}_cost_ext_cost_0_fun_work;
    {% else %}
    capsule->ext_cost_0_fun.fun = &{{ cost.cost_function_ext_cost_0 }};
    {% endif %}
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_create(&capsule->ext_cost_0_fun, {{ dims.np }});

    // external cost
    {% if cost.cost_ext_fun_type_0 == "casadi" %}
    capsule->ext_cost_0_fun_jac.casadi_fun = &{{ model.name }}_cost_ext_cost_0_fun_jac;
    capsule->ext_cost_0_fun_jac.casadi_n_in = &{{ model.name }}_cost_ext_cost_0_fun_jac_n_in;
    capsule->ext_cost_0_fun_jac.casadi_n_out = &{{ model.name }}_cost_ext_cost_0_fun_jac_n_out;
    capsule->ext_cost_0_fun_jac.casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_0_fun_jac_sparsity_in;
    capsule->ext_cost_0_fun_jac.casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_0_fun_jac_sparsity_out;
    capsule->ext_cost_0_fun_jac.casadi_work = &{{ model.name }}_cost_ext_cost_0_fun_jac_work;
    {% else %}
    capsule->ext_cost_0_fun_jac.fun = &{{ cost.cost_function_ext_cost_0 }};
    {% endif %}
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_create(&capsule->ext_cost_0_fun_jac, {{ dims.np }});

    // external cost
    {% if cost.cost_ext_fun_type_0 == "casadi" %}
    capsule->ext_cost_0_fun_jac_hess.casadi_fun = &{{ model.name }}_cost_ext_cost_0_fun_jac_hess;
    capsule->ext_cost_0_fun_jac_hess.casadi_n_in = &{{ model.name }}_cost_ext_cost_0_fun_jac_hess_n_in;
    capsule->ext_cost_0_fun_jac_hess.casadi_n_out = &{{ model.name }}_cost_ext_cost_0_fun_jac_hess_n_out;
    capsule->ext_cost_0_fun_jac_hess.casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_0_fun_jac_hess_sparsity_in;
    capsule->ext_cost_0_fun_jac_hess.casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_0_fun_jac_hess_sparsity_out;
    capsule->ext_cost_0_fun_jac_hess.casadi_work = &{{ model.name }}_cost_ext_cost_0_fun_jac_hess_work;
    {% else %}
    capsule->ext_cost_0_fun_jac_hess.fun = &{{ cost.cost_function_ext_cost_0 }};
    {% endif %}
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_create(&capsule->ext_cost_0_fun_jac_hess, {{ dims.np }});
{%- endif %}

{%- if cost.cost_type == "NONLINEAR_LS" %}
    // nonlinear least squares cost
    capsule->cost_y_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        capsule->cost_y_fun[i].casadi_fun = &{{ model.name }}_cost_y_fun;
        capsule->cost_y_fun[i].casadi_n_in = &{{ model.name }}_cost_y_fun_n_in;
        capsule->cost_y_fun[i].casadi_n_out = &{{ model.name }}_cost_y_fun_n_out;
        capsule->cost_y_fun[i].casadi_sparsity_in = &{{ model.name }}_cost_y_fun_sparsity_in;
        capsule->cost_y_fun[i].casadi_sparsity_out = &{{ model.name }}_cost_y_fun_sparsity_out;
        capsule->cost_y_fun[i].casadi_work = &{{ model.name }}_cost_y_fun_work;

        external_function_param_casadi_create(&capsule->cost_y_fun[i], {{ dims.np }});
    }

    capsule->cost_y_fun_jac_ut_xt = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        capsule->cost_y_fun_jac_ut_xt[i].casadi_fun = &{{ model.name }}_cost_y_fun_jac_ut_xt;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_n_in = &{{ model.name }}_cost_y_fun_jac_ut_xt_n_in;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_n_out = &{{ model.name }}_cost_y_fun_jac_ut_xt_n_out;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_sparsity_in = &{{ model.name }}_cost_y_fun_jac_ut_xt_sparsity_in;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_sparsity_out = &{{ model.name }}_cost_y_fun_jac_ut_xt_sparsity_out;
        capsule->cost_y_fun_jac_ut_xt[i].casadi_work = &{{ model.name }}_cost_y_fun_jac_ut_xt_work;

        external_function_param_casadi_create(&capsule->cost_y_fun_jac_ut_xt[i], {{ dims.np }});
    }

    capsule->cost_y_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N-1; i++)
    {
        capsule->cost_y_hess[i].casadi_fun = &{{ model.name }}_cost_y_hess;
        capsule->cost_y_hess[i].casadi_n_in = &{{ model.name }}_cost_y_hess_n_in;
        capsule->cost_y_hess[i].casadi_n_out = &{{ model.name }}_cost_y_hess_n_out;
        capsule->cost_y_hess[i].casadi_sparsity_in = &{{ model.name }}_cost_y_hess_sparsity_in;
        capsule->cost_y_hess[i].casadi_sparsity_out = &{{ model.name }}_cost_y_hess_sparsity_out;
        capsule->cost_y_hess[i].casadi_work = &{{ model.name }}_cost_y_hess_work;

        external_function_param_casadi_create(&capsule->cost_y_hess[i], {{ dims.np }});
    }
{%- elif cost.cost_type == "EXTERNAL" %}
    // external cost
    capsule->ext_cost_fun = (external_function_param_{{ cost.cost_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ cost.cost_ext_fun_type }})*N);
    for (int i = 0; i < N-1; i++)
    {
        {% if cost.cost_ext_fun_type == "casadi" %}
        capsule->ext_cost_fun[i].casadi_fun = &{{ model.name }}_cost_ext_cost_fun;
        capsule->ext_cost_fun[i].casadi_n_in = &{{ model.name }}_cost_ext_cost_fun_n_in;
        capsule->ext_cost_fun[i].casadi_n_out = &{{ model.name }}_cost_ext_cost_fun_n_out;
        capsule->ext_cost_fun[i].casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_fun_sparsity_in;
        capsule->ext_cost_fun[i].casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_fun_sparsity_out;
        capsule->ext_cost_fun[i].casadi_work = &{{ model.name }}_cost_ext_cost_fun_work;
        {% else %}
        capsule->ext_cost_fun[i].fun = &{{ cost.cost_function_ext_cost }};
        {% endif %}
        external_function_param_{{ cost.cost_ext_fun_type }}_create(&capsule->ext_cost_fun[i], {{ dims.np }});
    }

    capsule->ext_cost_fun_jac = (external_function_param_{{ cost.cost_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ cost.cost_ext_fun_type }})*N);
    for (int i = 0; i < N-1; i++)
    {
        {% if cost.cost_ext_fun_type == "casadi" %}
        capsule->ext_cost_fun_jac[i].casadi_fun = &{{ model.name }}_cost_ext_cost_fun_jac;
        capsule->ext_cost_fun_jac[i].casadi_n_in = &{{ model.name }}_cost_ext_cost_fun_jac_n_in;
        capsule->ext_cost_fun_jac[i].casadi_n_out = &{{ model.name }}_cost_ext_cost_fun_jac_n_out;
        capsule->ext_cost_fun_jac[i].casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_fun_jac_sparsity_in;
        capsule->ext_cost_fun_jac[i].casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_fun_jac_sparsity_out;
        capsule->ext_cost_fun_jac[i].casadi_work = &{{ model.name }}_cost_ext_cost_fun_jac_work;
        {% else %}
        capsule->ext_cost_fun_jac[i].fun = &{{ cost.cost_function_ext_cost }};
        {% endif %}
        external_function_param_{{ cost.cost_ext_fun_type }}_create(&capsule->ext_cost_fun_jac[i], {{ dims.np }});
    }

    capsule->ext_cost_fun_jac_hess = (external_function_param_{{ cost.cost_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ cost.cost_ext_fun_type }})*N);
    for (int i = 0; i < N-1; i++)
    {
        {% if cost.cost_ext_fun_type == "casadi" %}
        capsule->ext_cost_fun_jac_hess[i].casadi_fun = &{{ model.name }}_cost_ext_cost_fun_jac_hess;
        capsule->ext_cost_fun_jac_hess[i].casadi_n_in = &{{ model.name }}_cost_ext_cost_fun_jac_hess_n_in;
        capsule->ext_cost_fun_jac_hess[i].casadi_n_out = &{{ model.name }}_cost_ext_cost_fun_jac_hess_n_out;
        capsule->ext_cost_fun_jac_hess[i].casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_fun_jac_hess_sparsity_in;
        capsule->ext_cost_fun_jac_hess[i].casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_fun_jac_hess_sparsity_out;
        capsule->ext_cost_fun_jac_hess[i].casadi_work = &{{ model.name }}_cost_ext_cost_fun_jac_hess_work;
        {% else %}
        capsule->ext_cost_fun_jac_hess[i].fun = &{{ cost.cost_function_ext_cost }};
        {% endif %}
        external_function_param_{{ cost.cost_ext_fun_type }}_create(&capsule->ext_cost_fun_jac_hess[i], {{ dims.np }});
    }
{%- endif %}

{%- if cost.cost_type_e == "NONLINEAR_LS" %}
    // nonlinear least square function
    capsule->cost_y_e_fun.casadi_fun = &{{ model.name }}_cost_y_e_fun;
    capsule->cost_y_e_fun.casadi_n_in = &{{ model.name }}_cost_y_e_fun_n_in;
    capsule->cost_y_e_fun.casadi_n_out = &{{ model.name }}_cost_y_e_fun_n_out;
    capsule->cost_y_e_fun.casadi_sparsity_in = &{{ model.name }}_cost_y_e_fun_sparsity_in;
    capsule->cost_y_e_fun.casadi_sparsity_out = &{{ model.name }}_cost_y_e_fun_sparsity_out;
    capsule->cost_y_e_fun.casadi_work = &{{ model.name }}_cost_y_e_fun_work;
    external_function_param_casadi_create(&capsule->cost_y_e_fun, {{ dims.np }});

    capsule->cost_y_e_fun_jac_ut_xt.casadi_fun = &{{ model.name }}_cost_y_e_fun_jac_ut_xt;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_n_in = &{{ model.name }}_cost_y_e_fun_jac_ut_xt_n_in;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_n_out = &{{ model.name }}_cost_y_e_fun_jac_ut_xt_n_out;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_sparsity_in = &{{ model.name }}_cost_y_e_fun_jac_ut_xt_sparsity_in;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_sparsity_out = &{{ model.name }}_cost_y_e_fun_jac_ut_xt_sparsity_out;
    capsule->cost_y_e_fun_jac_ut_xt.casadi_work = &{{ model.name }}_cost_y_e_fun_jac_ut_xt_work;
    external_function_param_casadi_create(&capsule->cost_y_e_fun_jac_ut_xt, {{ dims.np }});

    capsule->cost_y_e_hess.casadi_fun = &{{ model.name }}_cost_y_e_hess;
    capsule->cost_y_e_hess.casadi_n_in = &{{ model.name }}_cost_y_e_hess_n_in;
    capsule->cost_y_e_hess.casadi_n_out = &{{ model.name }}_cost_y_e_hess_n_out;
    capsule->cost_y_e_hess.casadi_sparsity_in = &{{ model.name }}_cost_y_e_hess_sparsity_in;
    capsule->cost_y_e_hess.casadi_sparsity_out = &{{ model.name }}_cost_y_e_hess_sparsity_out;
    capsule->cost_y_e_hess.casadi_work = &{{ model.name }}_cost_y_e_hess_work;
    external_function_param_casadi_create(&capsule->cost_y_e_hess, {{ dims.np }});

{%- elif cost.cost_type_e == "EXTERNAL" %}
    // external cost
    {% if cost.cost_ext_fun_type_e == "casadi" %}
    capsule->ext_cost_e_fun.casadi_fun = &{{ model.name }}_cost_ext_cost_e_fun;
    capsule->ext_cost_e_fun.casadi_n_in = &{{ model.name }}_cost_ext_cost_e_fun_n_in;
    capsule->ext_cost_e_fun.casadi_n_out = &{{ model.name }}_cost_ext_cost_e_fun_n_out;
    capsule->ext_cost_e_fun.casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_e_fun_sparsity_in;
    capsule->ext_cost_e_fun.casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_e_fun_sparsity_out;
    capsule->ext_cost_e_fun.casadi_work = &{{ model.name }}_cost_ext_cost_e_fun_work;
    {% else %}
    capsule->ext_cost_e_fun.fun = &{{ cost.cost_function_ext_cost_e }};
    {% endif %}
    external_function_param_{{ cost.cost_ext_fun_type_e }}_create(&capsule->ext_cost_e_fun, {{ dims.np }});

    // external cost
    {% if cost.cost_ext_fun_type_e == "casadi" %}
    capsule->ext_cost_e_fun_jac.casadi_fun = &{{ model.name }}_cost_ext_cost_e_fun_jac;
    capsule->ext_cost_e_fun_jac.casadi_n_in = &{{ model.name }}_cost_ext_cost_e_fun_jac_n_in;
    capsule->ext_cost_e_fun_jac.casadi_n_out = &{{ model.name }}_cost_ext_cost_e_fun_jac_n_out;
    capsule->ext_cost_e_fun_jac.casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_e_fun_jac_sparsity_in;
    capsule->ext_cost_e_fun_jac.casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_e_fun_jac_sparsity_out;
    capsule->ext_cost_e_fun_jac.casadi_work = &{{ model.name }}_cost_ext_cost_e_fun_jac_work;
    {% else %}
    capsule->ext_cost_e_fun_jac.fun = &{{ cost.cost_function_ext_cost_e }};
    {% endif %}
    external_function_param_{{ cost.cost_ext_fun_type_e }}_create(&capsule->ext_cost_e_fun_jac, {{ dims.np }});

    // external cost
    {% if cost.cost_ext_fun_type_e == "casadi" %}
    capsule->ext_cost_e_fun_jac_hess.casadi_fun = &{{ model.name }}_cost_ext_cost_e_fun_jac_hess;
    capsule->ext_cost_e_fun_jac_hess.casadi_n_in = &{{ model.name }}_cost_ext_cost_e_fun_jac_hess_n_in;
    capsule->ext_cost_e_fun_jac_hess.casadi_n_out = &{{ model.name }}_cost_ext_cost_e_fun_jac_hess_n_out;
    capsule->ext_cost_e_fun_jac_hess.casadi_sparsity_in = &{{ model.name }}_cost_ext_cost_e_fun_jac_hess_sparsity_in;
    capsule->ext_cost_e_fun_jac_hess.casadi_sparsity_out = &{{ model.name }}_cost_ext_cost_e_fun_jac_hess_sparsity_out;
    capsule->ext_cost_e_fun_jac_hess.casadi_work = &{{ model.name }}_cost_ext_cost_e_fun_jac_hess_work;
    {% else %}
    capsule->ext_cost_e_fun_jac_hess.fun = &{{ cost.cost_function_ext_cost_e }};
    {% endif %}
    external_function_param_{{ cost.cost_ext_fun_type_e }}_create(&capsule->ext_cost_e_fun_jac_hess, {{ dims.np }});
{%- endif %}

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
    capsule->nlp_in = nlp_in;

    double time_steps[N];
    {%- for j in range(end=dims.N) %}
    time_steps[{{ j }}] = {{ solver_options.time_steps[j] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &time_steps[i]);
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
    {% elif solver_options.integrator_type == "GNSF" %}
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "phi_fun", &capsule->gnsf_phi_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "phi_fun_jac_y", &capsule->gnsf_phi_fun_jac_y[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "phi_jac_y_uhat", &capsule->gnsf_phi_jac_y_uhat[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "f_lo_jac_x1_x1dot_u_z",
                                   &capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i]);
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
{%- if cost.cost_type_0 == "NONLINEAR_LS" or cost.cost_type_0 == "LINEAR_LS" %}
{% if dims.ny_0 > 0 %}
    double W_0[NY0*NY0];
    {% for j in range(end=dims.ny_0) %}
        {%- for k in range(end=dims.ny_0) %}
    W_0[{{ j }}+(NY0) * {{ k }}] = {{ cost.W_0[j][k] }};
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);

    double yref_0[NY0];
    {% for j in range(end=dims.ny_0) %}
    yref_0[{{ j }}] = {{ cost.yref_0[j] }};
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
{% endif %}
{% endif %}

{%- if cost.cost_type == "NONLINEAR_LS" or cost.cost_type == "LINEAR_LS" %}
{% if dims.ny > 0 %}
    double W[NY*NY];
    {% for j in range(end=dims.ny) %}
        {%- for k in range(end=dims.ny) %}
    W[{{ j }}+(NY) * {{ k }}] = {{ cost.W[j][k] }};
        {%- endfor %}
    {%- endfor %}

    double yref[NY];
    {% for j in range(end=dims.ny) %}
    yref[{{ j }}] = {{ cost.yref[j] }};
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
{% endif %}
{% endif %}

{%- if cost.cost_type_0 == "LINEAR_LS" %}
    double Vx_0[NY0*NX];
    {% for j in range(end=dims.ny_0) %}
        {%- for k in range(end=dims.nx) %}
    Vx_0[{{ j }}+(NY0) * {{ k }}] = {{ cost.Vx_0[j][k] }};
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);

{% if dims.ny_0 > 0 and dims.nu > 0 %}
    double Vu_0[NY0*NU];
    {% for j in range(end=dims.ny_0) %}
        {%- for k in range(end=dims.nu) %}
    Vu_0[{{ j }}+(NY0) * {{ k }}] = {{ cost.Vu_0[j][k] }};
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
{% endif %}
{% if dims.ny_0 > 0 and dims.nz > 0 %}
    double Vz_0[NY0*NZ];
    {% for j in range(end=dims.ny_0) %}
        {%- for k in range(end=dims.nz) %}
    Vz_0[{{ j }}+(NY0) * {{ k }}] = {{ cost.Vz_0[j][k] }};
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vz", Vz_0);
{%- endif %}
{%- endif %}{# LINEAR LS #}


{%- if cost.cost_type == "LINEAR_LS" %}
    double Vx[NY*NX];
    {% for j in range(end=dims.ny) %}
        {%- for k in range(end=dims.nx) %}
    Vx[{{ j }}+(NY) * {{ k }}] = {{ cost.Vx[j][k] }};
        {%- endfor %}
    {%- endfor %}
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }

{% if dims.ny > 0 and dims.nu > 0 %}
    double Vu[NY*NU];
    {% for j in range(end=dims.ny) %}
        {%- for k in range(end=dims.nu) %}
    Vu[{{ j }}+(NY) * {{ k }}] = {{ cost.Vu[j][k] }};
        {%- endfor %}
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
{% endif %}

{% if dims.ny > 0 and dims.nz > 0 %}
    double Vz[NY*NZ];
    {% for j in range(end=dims.ny) %}
        {%- for k in range(end=dims.nz) %}
    Vz[{{ j }}+(NY) * {{ k }}] = {{ cost.Vz[j][k] }};
        {%- endfor %}
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vz", Vz);
    }
{%- endif %}
{%- endif %}{# LINEAR LS #}


{%- if cost.cost_type_0 == "NONLINEAR_LS" %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun", &capsule->cost_y_0_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun_jac", &capsule->cost_y_0_fun_jac_ut_xt);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "nls_y_hess", &capsule->cost_y_0_hess);
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
{%- elif cost.cost_type == "EXTERNAL" %}
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun", &capsule->ext_cost_fun[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac", &capsule->ext_cost_fun_jac[i-1]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac_hess", &capsule->ext_cost_fun_jac_hess[i-1]);
    }
{%- endif %}


{% if dims.ns > 0 %}
    double Zl[NS];
    double Zu[NS];
    double zl[NS];
    double zu[NS];
    {% for j in range(end=dims.ns) %}
    Zl[{{ j }}] = {{ cost.Zl[j] }};
    {%- endfor %}

    {% for j in range(end=dims.ns) %}
    Zu[{{ j }}] = {{ cost.Zu[j] }};
    {%- endfor %}

    {% for j in range(end=dims.ns) %}
    zl[{{ j }}] = {{ cost.zl[j] }};
    {%- endfor %}

    {% for j in range(end=dims.ns) %}
    zu[{{ j }}] = {{ cost.zu[j] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
{% endif %}

    // terminal cost
{% if cost.cost_type_e == "LINEAR_LS" or cost.cost_type_e == "NONLINEAR_LS" %}
{% if dims.ny_e > 0 %}
    double yref_e[NYN];
    {% for j in range(end=dims.ny_e) %}
    yref_e[{{ j }}] = {{ cost.yref_e[j] }};
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);

    double W_e[NYN*NYN];
    {% for j in range(end=dims.ny_e) %}
        {%- for k in range(end=dims.ny_e) %}
    W_e[{{ j }}+(NYN) * {{ k }}] = {{ cost.W_e[j][k] }};
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);

    {%- if cost.cost_type_e == "LINEAR_LS" %}
    double Vx_e[NYN*NX];
    {% for j in range(end=dims.ny_e) %}
        {%- for k in range(end=dims.nx) %}
    Vx_e[{{ j }}+(NYN) * {{ k }}] = {{ cost.Vx_e[j][k] }};
        {%- endfor %}
    {%- endfor %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    {%- endif %}

    {%- if cost.cost_type_e == "NONLINEAR_LS" %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &capsule->cost_y_e_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &capsule->cost_y_e_fun_jac_ut_xt);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "nls_y_hess", &capsule->cost_y_e_hess);
    {%- endif %}
{%- endif %}{# ny_e > 0 #}

{%- elif cost.cost_type_e == "EXTERNAL" %}
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun", &capsule->ext_cost_e_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac", &capsule->ext_cost_e_fun_jac);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac_hess", &capsule->ext_cost_e_fun_jac_hess);
{%- endif %}

{% if dims.ns_e > 0 %}
    double Zl_e[NSN];
    double Zu_e[NSN];
    double zl_e[NSN];
    double zu_e[NSN];

    {% for j in range(end=dims.ns_e) %}
    Zl_e[{{ j }}] = {{ cost.Zl_e[j] }};
    {%- endfor %}

    {% for j in range(end=dims.ns_e) %}
    Zu_e[{{ j }}] = {{ cost.Zu_e[j] }};
    {%- endfor %}

    {% for j in range(end=dims.ns_e) %}
    zl_e[{{ j }}] = {{ cost.zl_e[j] }};
    {%- endfor %}

    {% for j in range(end=dims.ns_e) %}
    zu_e[{{ j }}] = {{ cost.zu_e[j] }};
    {%- endfor %}

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zl", Zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zu", Zu_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zl", zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zu", zu_e);
{%- endif %}

    /**** Constraints ****/

    // bounds for initial stage
{% if dims.nbx_0 > 0 %}
    // x0
    int idxbx0[{{ dims.nbx_0 }}];
    {% for i in range(end=dims.nbx_0) %}
    idxbx0[{{ i }}] = {{ constraints.idxbx_0[i] }};
    {%- endfor %}

    double lbx0[{{ dims.nbx_0 }}];
    double ubx0[{{ dims.nbx_0 }}];
    {% for i in range(end=dims.nbx_0) %}
    lbx0[{{ i }}] = {{ constraints.lbx_0[i] }};
    ubx0[{{ i }}] = {{ constraints.ubx_0[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);
{% endif %}
{% if dims.nbxe_0 > 0 %}
    // idxbxe_0
    int idxbxe_0[{{ dims.nbxe_0 }}];
    {% for i in range(end=dims.nbxe_0) %}
    idxbxe_0[{{ i }}] = {{ constraints.idxbxe_0[i] }};
    {%- endfor %}
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbxe", idxbxe_0);
{% endif %}

    /* constraints that are the same for initial and intermediate */
{%- if dims.nsbx > 0 %}
{# TODO: introduce nsbx0 & move this block down!! #}
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxsbx", idxsbx);
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lsbx", lsbx);
    // ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "usbx", usbx);

    // soft bounds on x
    int idxsbx[NSBX];
    {% for i in range(end=dims.nsbx) %}
    idxsbx[{{ i }}] = {{ constraints.idxsbx[i] }};
    {%- endfor %}
    double lsbx[NSBX];
    double usbx[NSBX];
    {% for i in range(end=dims.nsbx) %}
    lsbx[{{ i }}] = {{ constraints.lsbx[i] }};
    usbx[{{ i }}] = {{ constraints.usbx[i] }};
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {       
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsbx", idxsbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsbx", lsbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usbx", usbx);
    }
{%- endif %}


{% if dims.nbu > 0 %}
    // u
    int idxbu[NBU];
    {% for i in range(end=dims.nbu) %}
    idxbu[{{ i }}] = {{ constraints.idxbu[i] }};
    {%- endfor %}
    double lbu[NBU];
    double ubu[NBU];
    {% for i in range(end=dims.nbu) %}
    lbu[{{ i }}] = {{ constraints.lbu[i] }};
    ubu[{{ i }}] = {{ constraints.ubu[i] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }
{% endif %}

{% if dims.nsbu > 0 %}
    // set up soft bounds for u
    int idxsbu[NSBU];
    {% for i in range(end=dims.nsbu) %}
    idxsbu[{{ i }}] = {{ constraints.idxsbu[i] }};
    {%- endfor %}
    double lsbu[NSBU];
    double usbu[NSBU];
    {% for i in range(end=dims.nsbu) %}
    lsbu[{{ i }}] = {{ constraints.lsbu[i] }};
    usbu[{{ i }}] = {{ constraints.usbu[i] }};
    {%- endfor %}
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsbu", idxsbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsbu", lsbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usbu", usbu);
    }
{% endif %}

{% if dims.nsg > 0 %}
    // set up soft bounds for general linear constraints
    int idxsg[NSG];
    {% for i in range(end=dims.nsg) %}
    idxsg[{{ i }}] = {{ constraints.idxsg[i] }};
    {%- endfor %}
    double lsg[NSG];
    double usg[NSG];
    {% for i in range(end=dims.nsg) %}
    lsg[{{ i }}] = {{ constraints.lsg[i] }};
    usg[{{ i }}] = {{ constraints.usg[i] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsg", idxsg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsg", lsg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usg", usg);
    }
{% endif %}

{% if dims.nsh > 0 %}
    // set up soft bounds for nonlinear constraints
    int idxsh[NSH];
    {% for i in range(end=dims.nsh) %}
    idxsh[{{ i }}] = {{ constraints.idxsh[i] }};
    {%- endfor %}
    double lsh[NSH];
    double ush[NSH];
    {% for i in range(end=dims.nsh) %}
    lsh[{{ i }}] = {{ constraints.lsh[i] }};
    ush[{{ i }}] = {{ constraints.ush[i] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsh", idxsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsh", lsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ush", ush);
    }
{% endif %}

{% if dims.nsphi > 0 %}
    // set up soft bounds for convex-over-nonlinear constraints
    int idxsphi[NSPHI];
    {% for i in range(end=dims.nsphi) %}
    idxsphi[{{ i }}] = {{ constraints.idxsphi[i] }};
    {%- endfor %}
    double lsphi[NSPHI];
    double usphi[NSPHI];
    {% for i in range(end=dims.nsphi) %}
    lsphi[{{ i }}] = {{ constraints.lsphi[i] }};
    usphi[{{ i }}] = {{ constraints.usphi[i] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsphi", idxsphi);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsphi", lsphi);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "usphi", usphi);
    }
{% endif %}

{% if dims.nbx > 0 %}
    // x
    int idxbx[NBX];
    {% for i in range(end=dims.nbx) %}
    idxbx[{{ i }}] = {{ constraints.idxbx[i] }};
    {%- endfor %}
    double lbx[NBX];
    double ubx[NBX];
    {% for i in range(end=dims.nbx) %}
    lbx[{{ i }}] = {{ constraints.lbx[i] }};
    ubx[{{ i }}] = {{ constraints.ubx[i] }};
    {%- endfor %}

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
    }
{% endif %}

{% if dims.ng > 0 %}
    // set up general constraints for stage 0 to N-1 
    double D[NG*NU];
    double C[NG*NX];
    double lg[NG];
    double ug[NG];

    {% for j in range(end=dims.ng) %}
        {%- for k in range(end=dims.nu) %}
    D[{{ j }}+NG * {{ k }}] = {{ constraints.D[j][k] }};
        {%- endfor %}
    {%- endfor %}

    {% for j in range(end=dims.ng) %}
        {%- for k in range(end=dims.nx) %}
    C[{{ j }}+NG * {{ k }}] = {{ constraints.C[j][k] }};
        {%- endfor %}
    {%- endfor %}

    {% for i in range(end=dims.ng) %}
    lg[{{ i }}] = {{ constraints.lg[i] }};
    {%- endfor %}

    {% for i in range(end=dims.ng) %}
    ug[{{ i }}] = {{ constraints.ug[i] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "D", D);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "C", C);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lg", lg);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ug", ug);
    }
{% endif %}

{% if dims.nh > 0 %}
    // set up nonlinear constraints for stage 0 to N-1 
    double lh[NH];
    double uh[NH];

    {% for i in range(end=dims.nh) %}
    lh[{{ i }}] = {{ constraints.lh[i] }};
    {%- endfor %}

    {% for i in range(end=dims.nh) %}
    uh[{{ i }}] = {{ constraints.uh[i] }};
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
{% endif %}

{% if dims.nphi > 0 and constraints.constr_type == "BGP" %}
    // set up convex-over-nonlinear constraints for stage 0 to N-1 
    double lphi[NPHI];
    double uphi[NPHI];

    {% for i in range(end=dims.nphi) %}
    lphi[{{ i }}] = {{ constraints.lphi[i] }};
    {%- endfor %}

    {% for i in range(end=dims.nphi) %}
    uphi[{{ i }}] = {{ constraints.uphi[i] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i,
                                      "nl_constr_phi_o_r_fun_phi_jac_ux_z_phi_hess_r_jac_ux", &capsule->phi_constraint[i]);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lphi", lphi);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "uphi", uphi);
    }
{% endif %}

    /* terminal constraints */
{% if dims.nbx_e > 0 %}
    // set up bounds for last stage
    // x
    int idxbx_e[NBXN];
    {% for i in range(end=dims.nbx_e) %}
    idxbx_e[{{ i }}] = {{ constraints.idxbx_e[i] }};
    {%- endfor %}
    double lbx_e[NBXN];
    double ubx_e[NBXN];
    {% for i in range(end=dims.nbx_e) %}
    lbx_e[{{ i }}] = {{ constraints.lbx_e[i] }};
    ubx_e[{{ i }}] = {{ constraints.ubx_e[i] }};
    {%- endfor %}
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxbx", idxbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lbx", lbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ubx", ubx_e);
{%- endif %}

{% if dims.nsg_e > 0 %}
    // set up soft bounds for general linear constraints
    int idxsg_e[NSGN];
    {% for i in range(end=dims.nsg_e) %}
    idxsg_e[{{ i }}] = {{ constraints.idxsg_e[i] }};
    {%- endfor %}
    double lsg_e[NSGN];
    double usg_e[NSGN];
    {% for i in range(end=dims.nsg_e) %}
    lsg_e[{{ i }}] = {{ constraints.lsg_e[i] }};
    usg_e[{{ i }}] = {{ constraints.usg_e[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsg", idxsg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsg", lsg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "usg", usg_e);
{%- endif %}

{% if dims.nsh_e > 0 %}
    // set up soft bounds for nonlinear constraints
    int idxsh_e[NSHN];
    {% for i in range(end=dims.nsh_e) %}
    idxsh_e[{{ i }}] = {{ constraints.idxsh_e[i] }};
    {%- endfor %}
    double lsh_e[NSHN];
    double ush_e[NSHN];
    {% for i in range(end=dims.nsh_e) %}
    lsh_e[{{ i }}] = {{ constraints.lsh_e[i] }};
    ush_e[{{ i }}] = {{ constraints.ush_e[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsh", idxsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsh", lsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ush", ush_e);
{%- endif %}

{% if dims.nsphi_e > 0 %}
    // set up soft bounds for convex-over-nonlinear constraints
    int idxsphi_e[NSPHIN];
    {% for i in range(end=dims.nsphi_e) %}
    idxsphi_e[{{ i }}] = {{ constraints.idxsphi_e[i] }};
    {%- endfor %}
    double lsphi_e[NSPHIN];
    double usphi_e[NSPHIN];
    {% for i in range(end=dims.nsphi_e) %}
    lsphi_e[{{ i }}] = {{ constraints.lsphi_e[i] }};
    usphi_e[{{ i }}] = {{ constraints.usphi_e[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsphi", idxsphi_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsphi", lsphi_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "usphi", usphi_e);
{%- endif %}

{% if dims.nsbx_e > 0 %}
    // soft bounds on x
    int idxsbx_e[NSBXN];
    {% for i in range(end=dims.nsbx_e) %}
    idxsbx_e[{{ i }}] = {{ constraints.idxsbx_e[i] }};
    {%- endfor %}
    double lsbx_e[NSBXN];
    double usbx_e[NSBXN];
    {% for i in range(end=dims.nsbx_e) %}
    lsbx_e[{{ i }}] = {{ constraints.lsbx_e[i] }};
    usbx_e[{{ i }}] = {{ constraints.usbx_e[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsbx", idxsbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsbx", lsbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "usbx", usbx_e);
{% endif %}

{% if dims.ng_e > 0 %}
    // set up general constraints for last stage 
    double C_e[NGN*NX];
    double lg_e[NGN];
    double ug_e[NGN];

    {% for j in range(end=dims.ng) %}
        {%- for k in range(end=dims.nx) %}
    C_e[{{ j }}+NG * {{ k }}] = {{ constraints.C_e[j][k] }};
        {%- endfor %}
    {%- endfor %}

    {% for i in range(end=dims.ng_e) %}
    lg_e[{{ i }}] = {{ constraints.lg_e[i] }};
    ug_e[{{ i }}] = {{ constraints.ug_e[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "C", C_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lg", lg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ug", ug_e);
{%- endif %}

{% if dims.nh_e > 0 %}
    // set up nonlinear constraints for last stage 
    double lh_e[NHN];
    double uh_e[NHN];

    {% for i in range(end=dims.nh_e) %}
    lh_e[{{ i }}] = {{ constraints.lh_e[i] }};
    {%- endfor %}

    {% for i in range(end=dims.nh_e) %}
    uh_e[{{ i }}] = {{ constraints.uh_e[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac", &capsule->nl_constr_h_e_fun_jac);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun", &capsule->nl_constr_h_e_fun);
    {% if solver_options.hessian_approx == "EXACT" %}
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac_hess",
                                  &capsule->nl_constr_h_e_fun_jac_hess);
    {% endif %}
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lh", lh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "uh", uh_e);
{%- endif %}

{% if dims.nphi_e > 0 and constraints.constr_type_e == "BGP" %}
    // set up convex-over-nonlinear constraints for last stage 
    double lphi_e[NPHIN];
    double uphi_e[NPHIN];

    {% for i in range(end=dims.nphi_e) %}
    lphi_e[{{ i }}] = {{ constraints.lphi_e[i] }};
    {%- endfor %}

    {% for i in range(end=dims.nphi_e) %}
    uphi_e[{{ i }}] = {{ constraints.uphi_e[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lphi", lphi_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "uphi", uphi_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N,
                                  "nl_constr_phi_o_r_fun_phi_jac_ux_z_phi_hess_r_jac_ux", &capsule->phi_e_constraint);
{% endif %}


    /************************************************
    *  opts
    ************************************************/

    capsule->nlp_opts = ocp_nlp_solver_opts_create(nlp_config, nlp_dims);

{% if solver_options.hessian_approx == "EXACT" %}
    bool nlp_solver_exact_hessian = true;
    // TODO: this if should not be needed! however, calling the setter with false leads to weird behavior. Investigate!
    if (nlp_solver_exact_hessian)
    {
        ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "exact_hess", &nlp_solver_exact_hessian);
    }
    int exact_hess_dyn = {{ solver_options.exact_hess_dyn }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "exact_hess_dyn", &exact_hess_dyn);

    int exact_hess_cost = {{ solver_options.exact_hess_cost }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "exact_hess_cost", &exact_hess_cost);

    int exact_hess_constr = {{ solver_options.exact_hess_constr }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "exact_hess_constr", &exact_hess_constr);
{%- endif -%}

{%- if solver_options.globalization == "FIXED_STEP" %}
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization", "fixed_step");
{%- elif solver_options.globalization == "MERIT_BACKTRACKING" %}
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization", "merit_backtracking");

    double alpha_min = {{ solver_options.alpha_min }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "alpha_min", &alpha_min);

    double alpha_reduction = {{ solver_options.alpha_reduction }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "alpha_reduction", &alpha_reduction);
{%- endif -%}

{%- if dims.nz > 0 %}
    // TODO: these options are lower level -> should be encapsulated! maybe through hessian approx option.
    bool output_z_val = true;
    bool sens_algebraic_val = true;

    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_output_z", &output_z_val);
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_sens_algebraic", &sens_algebraic_val);
{%- endif %}

{%- if solver_options.integrator_type != "DISCRETE" %}

    int sim_method_num_steps[N];
    {%- for j in range(end=dims.N) %}
    sim_method_num_steps[{{ j }}] = {{ solver_options.sim_method_num_steps[j] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps[i]);

    int sim_method_num_stages[N];
    {%- for j in range(end=dims.N) %}
    sim_method_num_stages[{{ j }}] = {{ solver_options.sim_method_num_stages[j] }};
    {%- endfor %}

    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages[i]);

    int newton_iter_val = {{ solver_options.sim_method_newton_iter }};
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    bool tmp_bool = {{ solver_options.sim_method_jac_reuse }};
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

{%- endif %}

    double nlp_solver_step_length = {{ solver_options.nlp_solver_step_length }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "step_length", &nlp_solver_step_length);

    {%- if solver_options.nlp_solver_warm_start_first_qp %}
    int nlp_solver_warm_start_first_qp = {{ solver_options.nlp_solver_warm_start_first_qp }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "warm_start_first_qp", &nlp_solver_warm_start_first_qp);
    {%- endif %}

    double levenberg_marquardt = {{ solver_options.levenberg_marquardt }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
{%- if solver_options.qp_solver is starting_with("PARTIAL_CONDENSING") %}
    int qp_solver_cond_N;

    {%- if solver_options.qp_solver_cond_N %}
    qp_solver_cond_N = {{ solver_options.qp_solver_cond_N }};
    {% else %}
    // NOTE: there is no condensing happening here!
    qp_solver_cond_N = N;
    {%- endif %}
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);
{% endif %}

    int qp_solver_iter_max = {{ solver_options.qp_solver_iter_max }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_iter_max", &qp_solver_iter_max);

    {%- if solver_options.qp_solver_tol_stat %}
    double qp_solver_tol_stat = {{ solver_options.qp_solver_tol_stat }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_tol_stat", &qp_solver_tol_stat);
    {%- endif -%}

    {%- if solver_options.qp_solver_tol_eq %}
    double qp_solver_tol_eq = {{ solver_options.qp_solver_tol_eq }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_tol_eq", &qp_solver_tol_eq);
    {%- endif -%}

    {%- if solver_options.qp_solver_tol_ineq %}
    double qp_solver_tol_ineq = {{ solver_options.qp_solver_tol_ineq }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_tol_ineq", &qp_solver_tol_ineq);
    {%- endif -%}

    {%- if solver_options.qp_solver_tol_comp %}
    double qp_solver_tol_comp = {{ solver_options.qp_solver_tol_comp }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_tol_comp", &qp_solver_tol_comp);
    {%- endif -%}

    {%- if solver_options.qp_solver_warm_start %}
    int qp_solver_warm_start = {{ solver_options.qp_solver_warm_start }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_warm_start", &qp_solver_warm_start);
    {%- endif -%}
    
{% if solver_options.nlp_solver_type == "SQP" %}
    // set SQP specific options
    double nlp_solver_tol_stat = {{ solver_options.nlp_solver_tol_stat }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = {{ solver_options.nlp_solver_tol_eq }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = {{ solver_options.nlp_solver_tol_ineq }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = {{ solver_options.nlp_solver_tol_comp }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = {{ solver_options.nlp_solver_max_iter }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "max_iter", &nlp_solver_max_iter);

    int initialize_t_slacks = {{ solver_options.initialize_t_slacks }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "initialize_t_slacks", &initialize_t_slacks);
{%- endif %}

    int print_level = {{ solver_options.print_level }};
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "print_level", &print_level);


    int ext_cost_num_hess = {{ solver_options.ext_cost_num_hess }};
{%- if cost.cost_type == "EXTERNAL" %}
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "cost_numerical_hessian", &ext_cost_num_hess);
    }
{%- endif %}
{%- if cost.cost_type_e == "EXTERNAL" %}
    ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, N, "cost_numerical_hessian", &ext_cost_num_hess);
{%- endif %}


    /* out */
    ocp_nlp_out * nlp_out = ocp_nlp_out_create(nlp_config, nlp_dims);
    capsule->nlp_out = nlp_out;

    // initialize primal solution
    double x0[{{ dims.nx }}];
{% if dims.nbx_0 == dims.nx %}
    // initialize with x0
    {% for item in constraints.lbx_0 %}
    x0[{{ loop.index0 }}] = {{ item }};
    {%- endfor %}
{% else %}
    // initialize with zeros
    {% for i in range(end=dims.nx) %}
    x0[{{ i }}] = 0.0;
    {%- endfor %}
{%- endif %}

    double u0[NU];
    {% for i in range(end=dims.nu) %}
    u0[{{ i }}] = 0.0;
    {%- endfor %}

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    
    capsule->nlp_solver = ocp_nlp_solver_create(nlp_config, nlp_dims, capsule->nlp_opts);


{% if dims.np > 0 %}
    // initialize parameters to nominal value
    double p[{{ dims.np }}];
    {% for i in range(end=dims.np) %}
    p[{{ i }}] = {{ parameter_values[i] }};
    {%- endfor %}

    for (int i = 0; i <= N; i++)
    {
        {{ model.name }}_acados_update_params(capsule, i, p, NP);
    }
{%- endif %}{# if dims.np #}

    status = ocp_nlp_precompute(capsule->nlp_solver, nlp_in, nlp_out);

    if (status != ACADOS_SUCCESS)
    {
        printf("\nocp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int {{ model.name }}_acados_update_params(nlp_solver_capsule * capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = {{ dims.np }};
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }

{%- if dims.np > 0 %}
    if (stage < {{ dims.N }} && stage >= 0)
    {
    {%- if solver_options.integrator_type == "IRK" %}
        capsule->impl_dae_fun[stage].set_param(capsule->impl_dae_fun+stage, p);
        capsule->impl_dae_fun_jac_x_xdot_z[stage].set_param(capsule->impl_dae_fun_jac_x_xdot_z+stage, p);
        capsule->impl_dae_jac_x_xdot_u_z[stage].set_param(capsule->impl_dae_jac_x_xdot_u_z+stage, p);

        {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->impl_dae_hess[stage].set_param(capsule->impl_dae_hess+stage, p);
        {%- endif %}
    {% elif solver_options.integrator_type == "ERK" %}
        capsule->forw_vde_casadi[stage].set_param(capsule->forw_vde_casadi+stage, p);
        capsule->expl_ode_fun[stage].set_param(capsule->expl_ode_fun+stage, p);

        {%- if solver_options.hessian_approx == "EXACT" %}
        capsule->hess_vde_casadi[stage].set_param(capsule->hess_vde_casadi+stage, p);
        {%- endif %}
    {% elif solver_options.integrator_type == "GNSF" %}
        capsule->gnsf_phi_fun[stage].set_param(capsule->gnsf_phi_fun+stage, p);
        capsule->gnsf_phi_fun_jac_y[stage].set_param(capsule->gnsf_phi_fun_jac_y+stage, p);
        capsule->gnsf_phi_jac_y_uhat[stage].set_param(capsule->gnsf_phi_jac_y_uhat+stage, p);

        capsule->gnsf_f_lo_jac_x1_x1dot_u_z[stage].set_param(capsule->gnsf_f_lo_jac_x1_x1dot_u_z+stage, p);
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
{% endif %}{# if dims.np #}

    return solver_status;
}



int {{ model.name }}_acados_solve(nlp_solver_capsule * capsule)
{
    // solve NLP 
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


int {{ model.name }}_acados_free(nlp_solver_capsule * capsule)
{
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
{%- if solver_options.integrator_type == "IRK" %}
    for (int i = 0; i < {{ dims.N }}; i++)
    {
        external_function_param_casadi_free(&capsule->impl_dae_fun[i]);
        external_function_param_casadi_free(&capsule->impl_dae_fun_jac_x_xdot_z[i]);
        external_function_param_casadi_free(&capsule->impl_dae_jac_x_xdot_u_z[i]);
    {%- if solver_options.hessian_approx == "EXACT" %}
        external_function_param_casadi_free(&capsule->impl_dae_hess[i]);
    {%- endif %}
    }
    free(capsule->impl_dae_fun);
    free(capsule->impl_dae_fun_jac_x_xdot_z);
    free(capsule->impl_dae_jac_x_xdot_u_z);
    {%- if solver_options.hessian_approx == "EXACT" %}
    free(capsule->impl_dae_hess);
    {%- endif %}

{%- elif solver_options.integrator_type == "ERK" %}
    for (int i = 0; i < {{ dims.N }}; i++)
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
    for (int i = 0; i < {{ dims.N }}; i++)
    {
        external_function_param_casadi_free(&capsule->gnsf_phi_fun[i]);
        external_function_param_casadi_free(&capsule->gnsf_phi_fun_jac_y[i]);
        external_function_param_casadi_free(&capsule->gnsf_phi_jac_y_uhat[i]);
        external_function_param_casadi_free(&capsule->gnsf_f_lo_jac_x1_x1dot_u_z[i]);
        external_function_param_casadi_free(&capsule->gnsf_get_matrices_fun[i]);
    }
    free(capsule->gnsf_phi_fun);
    free(capsule->gnsf_phi_fun_jac_y);
    free(capsule->gnsf_phi_jac_y_uhat);
    free(capsule->gnsf_f_lo_jac_x1_x1dot_u_z);
    free(capsule->gnsf_get_matrices_fun);
{%- elif solver_options.integrator_type == "DISCRETE" %}
    for (int i = 0; i < {{ dims.N }}; i++)
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
{%- elif cost.cost_type_0 == "EXTERNAL" %}
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_free(&capsule->ext_cost_0_fun);
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_free(&capsule->ext_cost_0_fun_jac);
    external_function_param_{{ cost.cost_ext_fun_type_0 }}_free(&capsule->ext_cost_0_fun_jac_hess);
{%- endif %}
{%- if cost.cost_type == "NONLINEAR_LS" %}
    for (int i = 0; i < {{ dims.N }} - 1; i++)
    {
        external_function_param_casadi_free(&capsule->cost_y_fun[i]);
        external_function_param_casadi_free(&capsule->cost_y_fun_jac_ut_xt[i]);
        external_function_param_casadi_free(&capsule->cost_y_hess[i]);
    }
    free(capsule->cost_y_fun);
    free(capsule->cost_y_fun_jac_ut_xt);
    free(capsule->cost_y_hess);
{%- elif cost.cost_type == "EXTERNAL" %}
    for (int i = 0; i < {{ dims.N }} - 1; i++)
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
{%- elif cost.cost_type_e == "EXTERNAL" %}
    external_function_param_{{ cost.cost_ext_fun_type_e }}_free(&capsule->ext_cost_e_fun);
    external_function_param_{{ cost.cost_ext_fun_type_e }}_free(&capsule->ext_cost_e_fun_jac);
    external_function_param_{{ cost.cost_ext_fun_type_e }}_free(&capsule->ext_cost_e_fun_jac_hess);
{%- endif %}

    // constraints
{%- if constraints.constr_type == "BGH" and dims.nh > 0 %}
    for (int i = 0; i < {{ dims.N }}; i++)
    {
        external_function_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
  {%- if solver_options.hessian_approx == "EXACT" %}
    for (int i = 0; i < {{ dims.N }}; i++)
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
    for (int i = 0; i < {{ dims.N }}; i++)
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

ocp_nlp_in *{{ model.name }}_acados_get_nlp_in(nlp_solver_capsule * capsule) { return capsule->nlp_in; }
ocp_nlp_out *{{ model.name }}_acados_get_nlp_out(nlp_solver_capsule * capsule) { return capsule->nlp_out; }
ocp_nlp_solver *{{ model.name }}_acados_get_nlp_solver(nlp_solver_capsule * capsule) { return capsule->nlp_solver; }
ocp_nlp_config *{{ model.name }}_acados_get_nlp_config(nlp_solver_capsule * capsule) { return capsule->nlp_config; }
void *{{ model.name }}_acados_get_nlp_opts(nlp_solver_capsule * capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *{{ model.name }}_acados_get_nlp_dims(nlp_solver_capsule * capsule) { return capsule->nlp_dims; }
ocp_nlp_plan *{{ model.name }}_acados_get_nlp_plan(nlp_solver_capsule * capsule) { return capsule->nlp_solver_plan; }


void {{ model.name }}_acados_print_stats(nlp_solver_capsule * capsule)
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_m", &stat_m);

    {% set stat_n_max = 10 %}
    double stat[{{ solver_options.nlp_solver_max_iter * stat_n_max }}];
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "statistics", stat);

    int nrow = sqp_iter+1 < stat_m ? sqp_iter+1 : stat_m;

    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j > 4)
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
}

