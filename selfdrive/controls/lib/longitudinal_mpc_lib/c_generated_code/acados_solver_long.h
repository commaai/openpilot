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

#ifndef ACADOS_SOLVER_long_H_
#define ACADOS_SOLVER_long_H_

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

// ** capsule for solver data **
typedef struct nlp_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics
    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;
    external_function_param_casadi *hess_vde_casadi;
    external_function_param_casadi *impl_dae_fun;
    external_function_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_param_casadi *impl_dae_jac_x_xdot_u_z;
    external_function_param_casadi *impl_dae_fun_jac_x_xdot_u;
    external_function_param_casadi *impl_dae_hess;
    external_function_param_casadi *gnsf_phi_fun;
    external_function_param_casadi *gnsf_phi_fun_jac_y;
    external_function_param_casadi *gnsf_phi_jac_y_uhat;
    external_function_param_casadi *gnsf_f_lo_jac_x1_x1dot_u_z;
    external_function_param_casadi *gnsf_get_matrices_fun;
    external_function_param_casadi *discr_dyn_phi_fun;
    external_function_param_casadi *discr_dyn_phi_fun_jac_ut_xt;
    external_function_param_casadi *discr_dyn_phi_fun_jac_ut_xt_hess;

    // cost
    external_function_param_casadi *cost_y_fun;
    external_function_param_casadi *cost_y_fun_jac_ut_xt;
    external_function_param_casadi *cost_y_hess;
    external_function_param_casadi *ext_cost_fun;
    external_function_param_casadi *ext_cost_fun_jac;
    external_function_param_casadi *ext_cost_fun_jac_hess;

    external_function_param_casadi cost_y_0_fun;
    external_function_param_casadi cost_y_0_fun_jac_ut_xt;
    external_function_param_casadi cost_y_0_hess;
    external_function_param_casadi ext_cost_0_fun;
    external_function_param_casadi ext_cost_0_fun_jac;
    external_function_param_casadi ext_cost_0_fun_jac_hess;

    external_function_param_casadi cost_y_e_fun;
    external_function_param_casadi cost_y_e_fun_jac_ut_xt;
    external_function_param_casadi cost_y_e_hess;
    external_function_param_casadi ext_cost_e_fun;
    external_function_param_casadi ext_cost_e_fun_jac;
    external_function_param_casadi ext_cost_e_fun_jac_hess;

    // constraints
    external_function_param_casadi *phi_constraint;
    external_function_param_casadi *nl_constr_h_fun_jac;
    external_function_param_casadi *nl_constr_h_fun;
    external_function_param_casadi *nl_constr_h_fun_jac_hess;

    external_function_param_casadi phi_e_constraint;
    external_function_param_casadi nl_constr_h_e_fun_jac;
    external_function_param_casadi nl_constr_h_e_fun;
    external_function_param_casadi nl_constr_h_e_fun_jac_hess;
} nlp_solver_capsule;

nlp_solver_capsule * long_acados_create_capsule(void);
int long_acados_free_capsule(nlp_solver_capsule *capsule);

int long_acados_create(nlp_solver_capsule * capsule);
int long_acados_update_params(nlp_solver_capsule * capsule, int stage, double *value, int np);
int long_acados_solve(nlp_solver_capsule * capsule);
int long_acados_free(nlp_solver_capsule * capsule);
void long_acados_print_stats(nlp_solver_capsule * capsule);

ocp_nlp_in *long_acados_get_nlp_in(nlp_solver_capsule * capsule);
ocp_nlp_out *long_acados_get_nlp_out(nlp_solver_capsule * capsule);
ocp_nlp_solver *long_acados_get_nlp_solver(nlp_solver_capsule * capsule);
ocp_nlp_config *long_acados_get_nlp_config(nlp_solver_capsule * capsule);
void *long_acados_get_nlp_opts(nlp_solver_capsule * capsule);
ocp_nlp_dims *long_acados_get_nlp_dims(nlp_solver_capsule * capsule);
ocp_nlp_plan *long_acados_get_nlp_plan(nlp_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_long_H_
