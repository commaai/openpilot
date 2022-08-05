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


#ifndef ACADOS_SIM_SIM_LIFTED_IRK_INTEGRATOR_H_
#define ACADOS_SIM_SIM_LIFTED_IRK_INTEGRATOR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/sim/sim_common.h"
#include "acados/utils/types.h"

typedef struct
{
    int nx;
    int nu;
    int nz;
} sim_lifted_irk_dims;



typedef struct
{
    /* external functions */
    // implicit ode
    external_function_generic *impl_ode_fun;
    // implicit ode & jax_x & jac_xdot & jac_u implicit ode
    external_function_generic *impl_ode_fun_jac_x_xdot_u;

} lifted_irk_model;



typedef struct
{

    struct blasfeo_dmat *J_temp_x;     // temporary Jacobian of ode w.r.t x (nx, nx)
    struct blasfeo_dmat *J_temp_xdot;  // temporary Jacobian of ode w.r.t xdot (nx, nx)
    struct blasfeo_dmat *J_temp_u;     // temporary Jacobian of ode w.r.t u (nx, nu)

    struct blasfeo_dvec *rG;      // residuals of G (nx*ns)
    struct blasfeo_dvec *xt;      // temporary x
    struct blasfeo_dvec *xn;      // x at each integration step (for evaluations)
    struct blasfeo_dvec *xn_out;  // x at each integration step (output)
    struct blasfeo_dvec *dxn;     // dx at each integration step
    struct blasfeo_dvec *w;       // stacked x and u

    int *ipiv;  // index of pivot vector

} sim_lifted_irk_workspace;



typedef struct
{
    // memory for lifted integrators
    struct blasfeo_dmat *S_forw;    // forward sensitivities
    struct blasfeo_dmat *JGK;       // jacobian of G over K (nx*ns, nx*ns)
    struct blasfeo_dmat *JGf;       // jacobian of G over x and u (nx*ns, nx+nu);
    struct blasfeo_dmat *JKf;       // jacobian of K over x and u (nx*ns, nx+nu);

    struct blasfeo_dvec *K;         // internal variables (nx*ns)
    struct blasfeo_dvec *x;         // states (nx) -- for expansion step
    struct blasfeo_dvec *u;         // controls (nu) -- for expansion step

    int update_sens;
    // int init_K;

	double time_sim;
	double time_ad;
	double time_la;

} sim_lifted_irk_memory;



/* dims */
void sim_lifted_irk_dims_set(void *config_, void *dims_, const char *field, const int *value);
void sim_lifted_irk_dims_get(void *config_, void *dims_, const char *field, int* value);

acados_size_t sim_lifted_irk_dims_calculate_size();
//
void *sim_lifted_irk_dims_assign(void* config_, void *raw_memory);

/* model */
//
acados_size_t sim_lifted_irk_model_calculate_size(void *config, void *dims);
//
void *sim_lifted_irk_model_assign(void *config, void *dims, void *raw_memory);
//
int sim_lifted_irk_model_set(void *model_, const char *field, void *value);

/* opts */
//
acados_size_t sim_lifted_irk_opts_calculate_size(void *config, void *dims);
//
void *sim_lifted_irk_opts_assign(void *config, void *dims, void *raw_memory);
//
void sim_lifted_irk_opts_initialize_default(void *config, void *dims, void *opts_);
//
void sim_lifted_irk_opts_update(void *config_, void *dims, void *opts_);
//
void sim_lifted_irk_opts_set(void *config_, void *opts_, const char *field, void *value);

/* memory */
//
acados_size_t sim_lifted_irk_memory_calculate_size(void *config, void *dims, void *opts_);
//
void *sim_lifted_irk_memory_assign(void *config, void *dims, void *opts_, void *raw_memory);

/* workspace */
//
acados_size_t sim_lifted_irk_workspace_calculate_size(void *config, void *dims, void *opts_);
//
void sim_lifted_irk_config_initialize_default(void *config);

/* solver */
//
int sim_lifted_irk(void *config, sim_in *in, sim_out *out, void *opts_,
        void *mem_, void *work_);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SIM_SIM_LIFTED_IRK_INTEGRATOR_H_
