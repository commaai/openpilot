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

{%- if solver_options.hessian_approx %}
	{%- set hessian_approx = solver_options.hessian_approx %}
{%- elif solver_options.sens_hess %}
	{%- set hessian_approx = "EXACT" %}
{%- else %}
	{%- set hessian_approx = "GAUSS_NEWTON" %}
{%- endif %}
// standard
#include <stdio.h>
#include <stdlib.h>

// acados
#include "acados_c/external_function_interface.h"
#include "acados_c/sim_interface.h"
#include "acados_c/external_function_interface.h"

#include "acados/sim/sim_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/print.h"


// example specific
#include "{{ model.name }}_model/{{ model.name }}_model.h"
#include "acados_sim_solver_{{ model.name }}.h"


// ** solver data **

sim_solver_capsule * {{ model.name }}_acados_sim_solver_create_capsule()
{
    void* capsule_mem = malloc(sizeof(sim_solver_capsule));
    sim_solver_capsule *capsule = (sim_solver_capsule *) capsule_mem;

    return capsule;
}


int {{ model.name }}_acados_sim_solver_free_capsule(sim_solver_capsule * capsule)
{
    free(capsule);
    return 0;
}


int {{ model.name }}_acados_sim_create(sim_solver_capsule * capsule)
{
    // initialize
    const int nx = {{ model.name | upper }}_NX;
    const int nu = {{ model.name | upper }}_NU;
    const int nz = {{ model.name | upper }}_NZ;
    const int np = {{ model.name | upper }}_NP;
    bool tmp_bool;

    {#// double Tsim = {{ solver_options.tf / dims.N }};#}
    double Tsim = {{ solver_options.Tsim }};

    {% if solver_options.integrator_type == "IRK" %}
    capsule->sim_impl_dae_fun = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
    capsule->sim_impl_dae_fun_jac_x_xdot_z = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
    capsule->sim_impl_dae_jac_x_xdot_u_z = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));

  {%- if model.dyn_ext_fun_type == "casadi" %}
    // external functions (implicit model)
    capsule->sim_impl_dae_fun->casadi_fun = &{{ model.name }}_impl_dae_fun;
    capsule->sim_impl_dae_fun->casadi_work = &{{ model.name }}_impl_dae_fun_work;
    capsule->sim_impl_dae_fun->casadi_sparsity_in = &{{ model.name }}_impl_dae_fun_sparsity_in;
    capsule->sim_impl_dae_fun->casadi_sparsity_out = &{{ model.name }}_impl_dae_fun_sparsity_out;
    capsule->sim_impl_dae_fun->casadi_n_in = &{{ model.name }}_impl_dae_fun_n_in;
    capsule->sim_impl_dae_fun->casadi_n_out = &{{ model.name }}_impl_dae_fun_n_out;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_impl_dae_fun, np);

    capsule->sim_impl_dae_fun_jac_x_xdot_z->casadi_fun = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z;
    capsule->sim_impl_dae_fun_jac_x_xdot_z->casadi_work = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_work;
    capsule->sim_impl_dae_fun_jac_x_xdot_z->casadi_sparsity_in = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_sparsity_in;
    capsule->sim_impl_dae_fun_jac_x_xdot_z->casadi_sparsity_out = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_sparsity_out;
    capsule->sim_impl_dae_fun_jac_x_xdot_z->casadi_n_in = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_n_in;
    capsule->sim_impl_dae_fun_jac_x_xdot_z->casadi_n_out = &{{ model.name }}_impl_dae_fun_jac_x_xdot_z_n_out;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_impl_dae_fun_jac_x_xdot_z, np);

    // external_function_param_{{ model.dyn_ext_fun_type }} impl_dae_jac_x_xdot_u_z;
    capsule->sim_impl_dae_jac_x_xdot_u_z->casadi_fun = &{{ model.name }}_impl_dae_jac_x_xdot_u_z;
    capsule->sim_impl_dae_jac_x_xdot_u_z->casadi_work = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_work;
    capsule->sim_impl_dae_jac_x_xdot_u_z->casadi_sparsity_in = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_sparsity_in;
    capsule->sim_impl_dae_jac_x_xdot_u_z->casadi_sparsity_out = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_sparsity_out;
    capsule->sim_impl_dae_jac_x_xdot_u_z->casadi_n_in = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_n_in;
    capsule->sim_impl_dae_jac_x_xdot_u_z->casadi_n_out = &{{ model.name }}_impl_dae_jac_x_xdot_u_z_n_out;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_impl_dae_jac_x_xdot_u_z, np);
  {%- else %}
    capsule->sim_impl_dae_fun->fun = &{{ model.dyn_impl_dae_fun }};
    capsule->sim_impl_dae_fun_jac_x_xdot_z->fun = &{{ model.dyn_impl_dae_fun_jac }};
    capsule->sim_impl_dae_jac_x_xdot_u_z->fun = &{{ model.dyn_impl_dae_jac }};
  {%- endif %}

{%- if hessian_approx == "EXACT" %}
    capsule->sim_impl_dae_hess = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
    // external_function_param_{{ model.dyn_ext_fun_type }} impl_dae_jac_x_xdot_u_z;
    capsule->sim_impl_dae_hess->casadi_fun = &{{ model.name }}_impl_dae_hess;
    capsule->sim_impl_dae_hess->casadi_work = &{{ model.name }}_impl_dae_hess_work;
    capsule->sim_impl_dae_hess->casadi_sparsity_in = &{{ model.name }}_impl_dae_hess_sparsity_in;
    capsule->sim_impl_dae_hess->casadi_sparsity_out = &{{ model.name }}_impl_dae_hess_sparsity_out;
    capsule->sim_impl_dae_hess->casadi_n_in = &{{ model.name }}_impl_dae_hess_n_in;
    capsule->sim_impl_dae_hess->casadi_n_out = &{{ model.name }}_impl_dae_hess_n_out;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_impl_dae_hess, np);
{%- endif %}

    {% elif solver_options.integrator_type == "ERK" %}
    // explicit ode
    capsule->sim_forw_vde_casadi = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
    capsule->sim_vde_adj_casadi = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
    capsule->sim_expl_ode_fun_casadi = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));

    capsule->sim_forw_vde_casadi->casadi_fun = &{{ model.name }}_expl_vde_forw;
    capsule->sim_forw_vde_casadi->casadi_n_in = &{{ model.name }}_expl_vde_forw_n_in;
    capsule->sim_forw_vde_casadi->casadi_n_out = &{{ model.name }}_expl_vde_forw_n_out;
    capsule->sim_forw_vde_casadi->casadi_sparsity_in = &{{ model.name }}_expl_vde_forw_sparsity_in;
    capsule->sim_forw_vde_casadi->casadi_sparsity_out = &{{ model.name }}_expl_vde_forw_sparsity_out;
    capsule->sim_forw_vde_casadi->casadi_work = &{{ model.name }}_expl_vde_forw_work;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_forw_vde_casadi, np);

    capsule->sim_vde_adj_casadi->casadi_fun = &{{ model.name }}_expl_vde_adj;
    capsule->sim_vde_adj_casadi->casadi_n_in = &{{ model.name }}_expl_vde_adj_n_in;
    capsule->sim_vde_adj_casadi->casadi_n_out = &{{ model.name }}_expl_vde_adj_n_out;
    capsule->sim_vde_adj_casadi->casadi_sparsity_in = &{{ model.name }}_expl_vde_adj_sparsity_in;
    capsule->sim_vde_adj_casadi->casadi_sparsity_out = &{{ model.name }}_expl_vde_adj_sparsity_out;
    capsule->sim_vde_adj_casadi->casadi_work = &{{ model.name }}_expl_vde_adj_work;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_vde_adj_casadi, np);

    capsule->sim_expl_ode_fun_casadi->casadi_fun = &{{ model.name }}_expl_ode_fun;
    capsule->sim_expl_ode_fun_casadi->casadi_n_in = &{{ model.name }}_expl_ode_fun_n_in;
    capsule->sim_expl_ode_fun_casadi->casadi_n_out = &{{ model.name }}_expl_ode_fun_n_out;
    capsule->sim_expl_ode_fun_casadi->casadi_sparsity_in = &{{ model.name }}_expl_ode_fun_sparsity_in;
    capsule->sim_expl_ode_fun_casadi->casadi_sparsity_out = &{{ model.name }}_expl_ode_fun_sparsity_out;
    capsule->sim_expl_ode_fun_casadi->casadi_work = &{{ model.name }}_expl_ode_fun_work;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_expl_ode_fun_casadi, np);

{%- if hessian_approx == "EXACT" %}
    capsule->sim_expl_ode_hess = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
    // external_function_param_{{ model.dyn_ext_fun_type }} impl_dae_jac_x_xdot_u_z;
    capsule->sim_expl_ode_hess->casadi_fun = &{{ model.name }}_expl_ode_hess;
    capsule->sim_expl_ode_hess->casadi_work = &{{ model.name }}_expl_ode_hess_work;
    capsule->sim_expl_ode_hess->casadi_sparsity_in = &{{ model.name }}_expl_ode_hess_sparsity_in;
    capsule->sim_expl_ode_hess->casadi_sparsity_out = &{{ model.name }}_expl_ode_hess_sparsity_out;
    capsule->sim_expl_ode_hess->casadi_n_in = &{{ model.name }}_expl_ode_hess_n_in;
    capsule->sim_expl_ode_hess->casadi_n_out = &{{ model.name }}_expl_ode_hess_n_out;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_expl_ode_hess, np);
{%- endif %}

    {% elif solver_options.integrator_type == "GNSF" -%}
  {% if model.gnsf.purely_linear != 1 %}
    capsule->sim_gnsf_phi_fun = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
    capsule->sim_gnsf_phi_fun_jac_y = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
    capsule->sim_gnsf_phi_jac_y_uhat = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
  {% if model.gnsf.nontrivial_f_LO == 1 %}
    capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));
  {%- endif %}
  {%- endif %}
    capsule->sim_gnsf_get_matrices_fun = (external_function_param_{{ model.dyn_ext_fun_type }} *) malloc(sizeof(external_function_param_{{ model.dyn_ext_fun_type }}));

  {% if model.gnsf.purely_linear != 1 %}
    capsule->sim_gnsf_phi_fun->casadi_fun = &{{ model.name }}_gnsf_phi_fun;
    capsule->sim_gnsf_phi_fun->casadi_n_in = &{{ model.name }}_gnsf_phi_fun_n_in;
    capsule->sim_gnsf_phi_fun->casadi_n_out = &{{ model.name }}_gnsf_phi_fun_n_out;
    capsule->sim_gnsf_phi_fun->casadi_sparsity_in = &{{ model.name }}_gnsf_phi_fun_sparsity_in;
    capsule->sim_gnsf_phi_fun->casadi_sparsity_out = &{{ model.name }}_gnsf_phi_fun_sparsity_out;
    capsule->sim_gnsf_phi_fun->casadi_work = &{{ model.name }}_gnsf_phi_fun_work;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_gnsf_phi_fun, np);

    capsule->sim_gnsf_phi_fun_jac_y->casadi_fun = &{{ model.name }}_gnsf_phi_fun_jac_y;
    capsule->sim_gnsf_phi_fun_jac_y->casadi_n_in = &{{ model.name }}_gnsf_phi_fun_jac_y_n_in;
    capsule->sim_gnsf_phi_fun_jac_y->casadi_n_out = &{{ model.name }}_gnsf_phi_fun_jac_y_n_out;
    capsule->sim_gnsf_phi_fun_jac_y->casadi_sparsity_in = &{{ model.name }}_gnsf_phi_fun_jac_y_sparsity_in;
    capsule->sim_gnsf_phi_fun_jac_y->casadi_sparsity_out = &{{ model.name }}_gnsf_phi_fun_jac_y_sparsity_out;
    capsule->sim_gnsf_phi_fun_jac_y->casadi_work = &{{ model.name }}_gnsf_phi_fun_jac_y_work;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_gnsf_phi_fun_jac_y, np);

    capsule->sim_gnsf_phi_jac_y_uhat->casadi_fun = &{{ model.name }}_gnsf_phi_jac_y_uhat;
    capsule->sim_gnsf_phi_jac_y_uhat->casadi_n_in = &{{ model.name }}_gnsf_phi_jac_y_uhat_n_in;
    capsule->sim_gnsf_phi_jac_y_uhat->casadi_n_out = &{{ model.name }}_gnsf_phi_jac_y_uhat_n_out;
    capsule->sim_gnsf_phi_jac_y_uhat->casadi_sparsity_in = &{{ model.name }}_gnsf_phi_jac_y_uhat_sparsity_in;
    capsule->sim_gnsf_phi_jac_y_uhat->casadi_sparsity_out = &{{ model.name }}_gnsf_phi_jac_y_uhat_sparsity_out;
    capsule->sim_gnsf_phi_jac_y_uhat->casadi_work = &{{ model.name }}_gnsf_phi_jac_y_uhat_work;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_gnsf_phi_jac_y_uhat, np);

  {% if model.gnsf.nontrivial_f_LO == 1 %}
    capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z->casadi_fun = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz;
    capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z->casadi_n_in = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_n_in;
    capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z->casadi_n_out = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_n_out;
    capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z->casadi_sparsity_in = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_sparsity_in;
    capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z->casadi_sparsity_out = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_sparsity_out;
    capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z->casadi_work = &{{ model.name }}_gnsf_f_lo_fun_jac_x1k1uz_work;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z, np);
  {%- endif %}
  {%- endif %}

    capsule->sim_gnsf_get_matrices_fun->casadi_fun = &{{ model.name }}_gnsf_get_matrices_fun;
    capsule->sim_gnsf_get_matrices_fun->casadi_n_in = &{{ model.name }}_gnsf_get_matrices_fun_n_in;
    capsule->sim_gnsf_get_matrices_fun->casadi_n_out = &{{ model.name }}_gnsf_get_matrices_fun_n_out;
    capsule->sim_gnsf_get_matrices_fun->casadi_sparsity_in = &{{ model.name }}_gnsf_get_matrices_fun_sparsity_in;
    capsule->sim_gnsf_get_matrices_fun->casadi_sparsity_out = &{{ model.name }}_gnsf_get_matrices_fun_sparsity_out;
    capsule->sim_gnsf_get_matrices_fun->casadi_work = &{{ model.name }}_gnsf_get_matrices_fun_work;
    external_function_param_{{ model.dyn_ext_fun_type }}_create(capsule->sim_gnsf_get_matrices_fun, np);
    {% endif %}

    // sim plan & config
    sim_solver_plan_t plan;
    plan.sim_solver = {{ solver_options.integrator_type }};

    // create correct config based on plan
    sim_config * {{ model.name }}_sim_config = sim_config_create(plan);
    capsule->acados_sim_config = {{ model.name }}_sim_config;

    // sim dims
    void *{{ model.name }}_sim_dims = sim_dims_create({{ model.name }}_sim_config);
    capsule->acados_sim_dims = {{ model.name }}_sim_dims;
    sim_dims_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims, "nx", &nx);
    sim_dims_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims, "nu", &nu);
    sim_dims_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims, "nz", &nz);
{% if solver_options.integrator_type == "GNSF" %}
    int gnsf_nx1 = {{ dims.gnsf_nx1 }};
    int gnsf_nz1 = {{ dims.gnsf_nz1 }};
    int gnsf_nout = {{ dims.gnsf_nout }};
    int gnsf_ny = {{ dims.gnsf_ny }};
    int gnsf_nuhat = {{ dims.gnsf_nuhat }};

    sim_dims_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims, "nx1", &gnsf_nx1);
    sim_dims_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims, "nz1", &gnsf_nz1);
    sim_dims_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims, "nout", &gnsf_nout);
    sim_dims_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims, "ny", &gnsf_ny);
    sim_dims_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims, "nuhat", &gnsf_nuhat);
{% endif %}

    // sim opts
    sim_opts *{{ model.name }}_sim_opts = sim_opts_create({{ model.name }}_sim_config, {{ model.name }}_sim_dims);
    capsule->acados_sim_opts = {{ model.name }}_sim_opts;
    int tmp_int = {{ solver_options.sim_method_newton_iter }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "newton_iter", &tmp_int);
    double tmp_double = {{ solver_options.sim_method_newton_tol }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "newton_tol", &tmp_double);
    sim_collocation_type collocation_type = {{ solver_options.collocation_type }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "collocation_type", &collocation_type);

{% if problem_class == "SIM" %}
    tmp_int = {{ solver_options.sim_method_num_stages }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "num_stages", &tmp_int);
    tmp_int = {{ solver_options.sim_method_num_steps }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "num_steps", &tmp_int);

    // options that are not available to AcadosOcpSolver
    //  (in OCP they will be determined by other options, like exact_hessian)
    tmp_bool = {{ solver_options.sens_forw }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "sens_forw", &tmp_bool);
    tmp_bool = {{ solver_options.sens_adj }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "sens_adj", &tmp_bool);
    tmp_bool = {{ solver_options.sens_algebraic }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "sens_algebraic", &tmp_bool);
    tmp_bool = {{ solver_options.sens_hess }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "sens_hess", &tmp_bool);
    tmp_bool = {{ solver_options.output_z }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "output_z", &tmp_bool);

{% else %} {# num_stages and num_steps of first shooting interval are used #}
    tmp_int = {{ solver_options.sim_method_num_stages[0] }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "num_stages", &tmp_int);
    tmp_int = {{ solver_options.sim_method_num_steps[0] }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "num_steps", &tmp_int);
    tmp_bool = {{ solver_options.sim_method_jac_reuse[0] }};
    sim_opts_set({{ model.name }}_sim_config, {{ model.name }}_sim_opts, "jac_reuse", &tmp_bool);
{% endif %}

    // sim in / out
    sim_in *{{ model.name }}_sim_in = sim_in_create({{ model.name }}_sim_config, {{ model.name }}_sim_dims);
    capsule->acados_sim_in = {{ model.name }}_sim_in;
    sim_out *{{ model.name }}_sim_out = sim_out_create({{ model.name }}_sim_config, {{ model.name }}_sim_dims);
    capsule->acados_sim_out = {{ model.name }}_sim_out;

    sim_in_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims,
               {{ model.name }}_sim_in, "T", &Tsim);

    // model functions
{%- if solver_options.integrator_type == "IRK" %}
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "impl_ode_fun", capsule->sim_impl_dae_fun);
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "impl_ode_fun_jac_x_xdot", capsule->sim_impl_dae_fun_jac_x_xdot_z);
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "impl_ode_jac_x_xdot_u", capsule->sim_impl_dae_jac_x_xdot_u_z);
{%- if hessian_approx == "EXACT" %}
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                "impl_dae_hess", capsule->sim_impl_dae_hess);
{%- endif %}

{%- elif solver_options.integrator_type == "ERK" %}
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "expl_vde_forw", capsule->sim_forw_vde_casadi);
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "expl_vde_adj", capsule->sim_vde_adj_casadi);
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "expl_ode_fun", capsule->sim_expl_ode_fun_casadi);
{%- if hessian_approx == "EXACT" %}
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                "expl_ode_hess", capsule->sim_expl_ode_hess);
{%- endif %}
{%- elif solver_options.integrator_type == "GNSF" %}
  {% if model.gnsf.purely_linear != 1 %}
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "phi_fun", capsule->sim_gnsf_phi_fun);
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "phi_fun_jac_y", capsule->sim_gnsf_phi_fun_jac_y);
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "phi_jac_y_uhat", capsule->sim_gnsf_phi_jac_y_uhat);
  {% if model.gnsf.nontrivial_f_LO == 1 %}
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "f_lo_jac_x1_x1dot_u_z", capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z);
  {%- endif %}
  {%- endif %}
    {{ model.name }}_sim_config->model_set({{ model.name }}_sim_in->model,
                 "gnsf_get_matrices_fun", capsule->sim_gnsf_get_matrices_fun);
{%- endif %}

    // sim solver
    sim_solver *{{ model.name }}_sim_solver = sim_solver_create({{ model.name }}_sim_config,
                                               {{ model.name }}_sim_dims, {{ model.name }}_sim_opts);
    capsule->acados_sim_solver = {{ model.name }}_sim_solver;

{% if dims.np > 0 %}
    /* initialize parameter values */
    double* p = calloc(np, sizeof(double));
    {% for item in parameter_values %}
        {%- if item != 0 %}
    p[{{ loop.index0 }}] = {{ item }};
        {%- endif %}
    {%- endfor %}

    {{ model.name }}_acados_sim_update_params(capsule, p, np);
    free(p);
{% endif %}{# if dims.np #}

    /* initialize input */
    // x
    double x0[{{ dims.nx }}];
    for (int ii = 0; ii < {{ dims.nx }}; ii++)
        x0[ii] = 0.0;

    sim_in_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims,
               {{ model.name }}_sim_in, "x", x0);


    // u
    double u0[{{ dims.nu }}];
    for (int ii = 0; ii < {{ dims.nu }}; ii++)
        u0[ii] = 0.0;

    sim_in_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims,
               {{ model.name }}_sim_in, "u", u0);

    // S_forw
    double S_forw[{{ dims.nx * (dims.nx + dims.nu) }}];
    for (int ii = 0; ii < {{ dims.nx * (dims.nx + dims.nu) }}; ii++)
        S_forw[ii] = 0.0;
    for (int ii = 0; ii < {{ dims.nx }}; ii++)
        S_forw[ii + ii * {{ dims.nx }} ] = 1.0;


    sim_in_set({{ model.name }}_sim_config, {{ model.name }}_sim_dims,
               {{ model.name }}_sim_in, "S_forw", S_forw);

    int status = sim_precompute({{ model.name }}_sim_solver, {{ model.name }}_sim_in, {{ model.name }}_sim_out);

    return status;
}


int {{ model.name }}_acados_sim_solve(sim_solver_capsule *capsule)
{
    // integrate dynamics using acados sim_solver
    int status = sim_solve(capsule->acados_sim_solver,
                           capsule->acados_sim_in, capsule->acados_sim_out);
    if (status != 0)
        printf("error in {{ model.name }}_acados_sim_solve()! Exiting.\n");

    return status;
}


int {{ model.name }}_acados_sim_free(sim_solver_capsule *capsule)
{
    // free memory
    sim_solver_destroy(capsule->acados_sim_solver);
    sim_in_destroy(capsule->acados_sim_in);
    sim_out_destroy(capsule->acados_sim_out);
    sim_opts_destroy(capsule->acados_sim_opts);
    sim_dims_destroy(capsule->acados_sim_dims);
    sim_config_destroy(capsule->acados_sim_config);

    // free external function
{%- if solver_options.integrator_type == "IRK" %}
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_impl_dae_fun);
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_impl_dae_fun_jac_x_xdot_z);
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_impl_dae_jac_x_xdot_u_z);
{%- if hessian_approx == "EXACT" %}
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_impl_dae_hess);
{%- endif %}
{%- elif solver_options.integrator_type == "ERK" %}
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_forw_vde_casadi);
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_vde_adj_casadi);
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_expl_ode_fun_casadi);
{%- if hessian_approx == "EXACT" %}
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_expl_ode_hess);
{%- endif %}
{%- elif solver_options.integrator_type == "GNSF" %}
  {% if model.gnsf.purely_linear != 1 %}
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_gnsf_phi_fun);
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_gnsf_phi_fun_jac_y);
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_gnsf_phi_jac_y_uhat);
  {% if model.gnsf.nontrivial_f_LO == 1 %}
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z);
  {%- endif %}
  {%- endif %}
    external_function_param_{{ model.dyn_ext_fun_type }}_free(capsule->sim_gnsf_get_matrices_fun);
{% endif %}

    return 0;
}


int {{ model.name }}_acados_sim_update_params(sim_solver_capsule *capsule, double *p, int np)
{
    int status = 0;
    int casadi_np = {{ model.name | upper }}_NP;

    if (casadi_np != np) {
        printf("{{ model.name }}_acados_sim_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }

{%- if solver_options.integrator_type == "ERK" %}
    capsule->sim_forw_vde_casadi[0].set_param(capsule->sim_forw_vde_casadi, p);
    capsule->sim_vde_adj_casadi[0].set_param(capsule->sim_vde_adj_casadi, p);
    capsule->sim_expl_ode_fun_casadi[0].set_param(capsule->sim_expl_ode_fun_casadi, p);
{%- if hessian_approx == "EXACT" %}
    capsule->sim_expl_ode_hess[0].set_param(capsule->sim_expl_ode_hess, p);
{%- endif %}
{%- elif solver_options.integrator_type == "IRK" %}
    capsule->sim_impl_dae_fun[0].set_param(capsule->sim_impl_dae_fun, p);
    capsule->sim_impl_dae_fun_jac_x_xdot_z[0].set_param(capsule->sim_impl_dae_fun_jac_x_xdot_z, p);
    capsule->sim_impl_dae_jac_x_xdot_u_z[0].set_param(capsule->sim_impl_dae_jac_x_xdot_u_z, p);
{%- if hessian_approx == "EXACT" %}
    capsule->sim_impl_dae_hess[0].set_param(capsule->sim_impl_dae_hess, p);
{%- endif %}
{%- elif solver_options.integrator_type == "GNSF" %}
  {% if model.gnsf.purely_linear != 1 %}
    capsule->sim_gnsf_phi_fun[0].set_param(capsule->sim_gnsf_phi_fun, p);
    capsule->sim_gnsf_phi_fun_jac_y[0].set_param(capsule->sim_gnsf_phi_fun_jac_y, p);
    capsule->sim_gnsf_phi_jac_y_uhat[0].set_param(capsule->sim_gnsf_phi_jac_y_uhat, p);
  {% if model.gnsf.nontrivial_f_LO == 1 %}
    capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z[0].set_param(capsule->sim_gnsf_f_lo_jac_x1_x1dot_u_z, p);
  {%- endif %}
  {%- endif %}
    capsule->sim_gnsf_get_matrices_fun[0].set_param(capsule->sim_gnsf_get_matrices_fun, p);
{% endif %}

    return status;
}

/* getters pointers to C objects*/
sim_config * {{ model.name }}_acados_get_sim_config(sim_solver_capsule *capsule)
{
    return capsule->acados_sim_config;
};

sim_in * {{ model.name }}_acados_get_sim_in(sim_solver_capsule *capsule)
{
    return capsule->acados_sim_in;
};

sim_out * {{ model.name }}_acados_get_sim_out(sim_solver_capsule *capsule)
{
    return capsule->acados_sim_out;
};

void * {{ model.name }}_acados_get_sim_dims(sim_solver_capsule *capsule)
{
    return capsule->acados_sim_dims;
};

sim_opts * {{ model.name }}_acados_get_sim_opts(sim_solver_capsule *capsule)
{
    return capsule->acados_sim_opts;
};

sim_solver  * {{ model.name }}_acados_get_sim_solver(sim_solver_capsule *capsule)
{
    return capsule->acados_sim_solver;
};

