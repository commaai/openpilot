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


#ifndef ACADOS_SIM_SIM_IRK_INTEGRATOR_H_
#define ACADOS_SIM_SIM_IRK_INTEGRATOR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/sim/sim_common.h"
#include "acados/utils/types.h"

#include "blasfeo/include/blasfeo_common.h"

typedef struct
{
    int nx;
    int nu;
    int nz;

} sim_irk_dims;



typedef struct
{
    /* external functions */
    // implicit fun - can either be fully implicit ode or dae
    //          - i.e. dae has z as additional last argument & nz > 0
    external_function_generic *impl_ode_fun;
    // implicit ode & jac_x & jax_xdot & jac_z
    external_function_generic *impl_ode_fun_jac_x_xdot_z;
    // jax_x & jac_xdot & jac_u & jac_z of implicit ode
    external_function_generic *impl_ode_jac_x_xdot_u_z;
    // hessian of implicit ode:
    external_function_generic *impl_ode_hess;
} irk_model;



typedef struct
{
    struct blasfeo_dvec *rG;        // residuals of G (nx*ns)
    struct blasfeo_dvec *K;         // internal K variables ((nx+nz)*ns)
    struct blasfeo_dvec *xt;        // temporary x
    struct blasfeo_dvec *xn;        // x at each integration step
    struct blasfeo_dvec xtdot;      // temporary xdot

    struct blasfeo_dvec *lambda;    // adjoint sensitivities (nx + nu)
    struct blasfeo_dvec *lambdaK;   // auxiliary variable ((nx+nz)*ns) for adjoint propagation

    struct blasfeo_dmat df_dx;     // temporary Jacobian of ode w.r.t x (nx+nz, nx)
    struct blasfeo_dmat df_dxdot;  // temporary Jacobian of ode w.r.t xdot (nx+nz, nx)
    struct blasfeo_dmat df_du;     // temporary Jacobian of ode w.r.t u (nx+nz, nu)
    struct blasfeo_dmat df_dz;     // temporary Jacobian of ode w.r.t z (nx+nz, nu)

    /* NOTE: the memory allocation corresponding to the following fields is CONDITIONAL */

    // only allocated if (opts->sens_algebraic || opts->output_z)
    int *ipiv_one_stage;  // index of pivot vector (nx + nz)
    double *Z_work;  // used to perform computations to get out->zn (ns)

    // df_dxdotz, dk0_dxu, only allocated if (opts->sens_algebraic && opts->exact_z_output)
    //      used for algebraic sensitivity generation
    struct blasfeo_dmat df_dxdotz;  // temporary Jacobian of ode w.r.t. xdot,z (nx+nz, nx+nz);
    struct blasfeo_dmat dk0_dxu;    // intermediate result, (nx+nz, nx+nu)

    // dK_dxu: if (!opts->sens_hess) - single blasfeo_dmat that is reused
    //         if ( opts->sens_hess) - array of (num_steps) blasfeo_dmat
    //                                  to store intermediate results
    struct blasfeo_dmat *dK_dxu;  // jacobian of (K,Z) over x and u ((nx+nz)*ns, nx+nu);

    // S_forw: if (!opts->sens_hess) - single blasfeo_dmat that is reused
    //         if ( opts->sens_hess) - array of (num_steps + 1) blasfeo_dmat
    //                                  to store intermediate results
    struct blasfeo_dmat *S_forw;  // forward sensitivities (nx, nx+nu)

    // dG_dxu: if (!opts->sens_hess) - single blasfeo_dmat that is reused
    //         if ( opts->sens_hess) - array of blasfeo_dmat to store intermediate results
    struct blasfeo_dmat *dG_dxu;  // jacobian of G over x and u ((nx+nz)*ns, nx+nu)

    // dG_dK:  if (!opts->sens_hess) - single blasfeo_dmat that is reused
    //         if ( opts->sens_hess) - array of blasfeo_dmat to store intermediate results
    struct blasfeo_dmat *dG_dK;   // jacobian of G over K ((nx+nz)*ns, (nx+nz)*ns)

    // ipiv: index of pivot vector
    //         if (!opts->sens_hess) - array (ns * (nx + nz)) that is reused
    //         if ( opts->sens_hess) - array (ns * (nx + nz)) * num_steps, to store all
    //              pivot vectors for dG_dxu
    int *ipiv;  // index of pivot vector

    // xn_traj, K_traj only available if( opts->sens_adj || opts->sens_hess )
    struct blasfeo_dvec *xn_traj;  // xn trajectory
    struct blasfeo_dvec *K_traj;   // K trajectory

    /* the following variables are only available if (opts->sens_hess) */
    // For Hessian propagation
    struct blasfeo_dmat Hess;   // temporary Hessian (nx + nu, nx + nu)
    // output of impl_ode_hess
    struct blasfeo_dmat f_hess;  // size: (nx + nu, nx + nu)
    struct blasfeo_dmat dxkzu_dw0;  // size (2*nx + nu + nz) x (nx + nu)
    struct blasfeo_dmat tmp_dxkzu_dw0;  // size (2*nx + nu + nz) x (nx + nu)

} sim_irk_workspace;


typedef struct
{
    double *xdot;  // xdot[NX] - initialization for state derivatives k within the integrator
    double *z;     // z[NZ] - initialization for algebraic variables z

	double time_sim;
	double time_ad;
	double time_la;
} sim_irk_memory;


// get & set functions
void sim_irk_dims_set(void *config_, void *dims_, const char *field, const int *value);
void sim_irk_dims_get(void *config_, void *dims_, const char *field, int* value);

// dims
acados_size_t sim_irk_dims_calculate_size();
void *sim_irk_dims_assign(void *config_, void *raw_memory);

// model
acados_size_t sim_irk_model_calculate_size(void *config, void *dims);
void *sim_irk_model_assign(void *config, void *dims, void *raw_memory);
int sim_irk_model_set(void *model, const char *field, void *value);

// opts
acados_size_t sim_irk_opts_calculate_size(void *config, void *dims);
void *sim_irk_opts_assign(void *config, void *dims, void *raw_memory);
void sim_irk_opts_initialize_default(void *config, void *dims, void *opts_);
void sim_irk_opts_update(void *config_, void *dims, void *opts_);
void sim_irk_opts_set(void *config_, void *opts_, const char *field, void *value);

// memory
acados_size_t sim_irk_memory_calculate_size(void *config, void *dims, void *opts_);
void *sim_irk_memory_assign(void *config, void *dims, void *opts_, void *raw_memory);
int sim_irk_memory_set(void *config_, void *dims_, void *mem_, const char *field, void *value);

// workspace
acados_size_t sim_irk_workspace_calculate_size(void *config, void *dims, void *opts_);
void sim_irk_config_initialize_default(void *config);

// main
int sim_irk(void *config, sim_in *in, sim_out *out, void *opts_, void *mem_, void *work_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SIM_SIM_IRK_INTEGRATOR_H_
