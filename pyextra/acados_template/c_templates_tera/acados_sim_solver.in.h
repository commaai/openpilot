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

#ifndef ACADOS_SIM_{{ model.name }}_H_
#define ACADOS_SIM_{{ model.name }}_H_

#include "acados_c/sim_interface.h"
#include "acados_c/external_function_interface.h"

#define {{ model.name | upper }}_NX     {{ dims.nx }}
#define {{ model.name | upper }}_NZ     {{ dims.nz }}
#define {{ model.name | upper }}_NU     {{ dims.nu }}
#define {{ model.name | upper }}_NP     {{ dims.np }}

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct sim_solver_capsule
{
    // acados objects
    sim_in *acados_sim_in;
    sim_out *acados_sim_out;
    sim_solver *acados_sim_solver;
    sim_opts *acados_sim_opts;
    sim_config *acados_sim_config;
    void *acados_sim_dims;

    /* external functions */
    // ERK
    external_function_param_casadi * sim_forw_vde_casadi;
    external_function_param_casadi * sim_expl_ode_fun_casadi;
    external_function_param_casadi * sim_expl_ode_hess;

    // IRK
    external_function_param_casadi * sim_impl_dae_fun;
    external_function_param_casadi * sim_impl_dae_fun_jac_x_xdot_z;
    external_function_param_casadi * sim_impl_dae_jac_x_xdot_u_z;
    external_function_param_casadi * sim_impl_dae_hess;

    // GNSF
    external_function_param_casadi * sim_gnsf_phi_fun;
    external_function_param_casadi * sim_gnsf_phi_fun_jac_y;
    external_function_param_casadi * sim_gnsf_phi_jac_y_uhat;
    external_function_param_casadi * sim_gnsf_f_lo_jac_x1_x1dot_u_z;
    external_function_param_casadi * sim_gnsf_get_matrices_fun;

} sim_solver_capsule;


ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_sim_create(sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_sim_solve(sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_sim_free(sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_sim_update_params(sim_solver_capsule *capsule, double *value, int np);

ACADOS_SYMBOL_EXPORT sim_config * {{ model.name }}_acados_get_sim_config(sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT sim_in * {{ model.name }}_acados_get_sim_in(sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT sim_out * {{ model.name }}_acados_get_sim_out(sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT void * {{ model.name }}_acados_get_sim_dims(sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT sim_opts * {{ model.name }}_acados_get_sim_opts(sim_solver_capsule *capsule);
ACADOS_SYMBOL_EXPORT sim_solver * {{ model.name }}_acados_get_sim_solver(sim_solver_capsule *capsule);


ACADOS_SYMBOL_EXPORT sim_solver_capsule * {{ model.name }}_acados_sim_solver_create_capsule(void);
ACADOS_SYMBOL_EXPORT int {{ model.name }}_acados_sim_solver_free_capsule(sim_solver_capsule *capsule);

#ifdef __cplusplus
}
#endif

#endif  // ACADOS_SIM_{{ model.name }}_H_
