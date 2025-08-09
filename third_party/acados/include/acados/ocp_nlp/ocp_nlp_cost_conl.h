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
/// \addtogroup ocp_nlp_cost ocp_nlp_cost
/// @{
/// \addtogroup ocp_nlp_cost_conl ocp_nlp_cost_conl
/// \brief This module implements convex-over-nonlinear costs of the form
/// \f$\min_{x,u,z} \psi(y(x,u,z,p) - y_{\text{ref}}, p)\f$,


#ifndef ACADOS_OCP_NLP_OCP_NLP_COST_CONL_H_
#define ACADOS_OCP_NLP_OCP_NLP_COST_CONL_H_

#ifdef __cplusplus
extern "C" {
#endif

// blasfeo
#include "blasfeo/include/blasfeo_common.h"

// acados
#include "acados/ocp_nlp/ocp_nlp_cost_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/types.h"



/************************************************
 * dims
 ************************************************/

typedef struct
{
    int nx;  // number of states
    int nz;  // number of algebraic variables
    int nu;  // number of inputs
    int ny;  // number of outputs
    int ns;  // number of slacks
} ocp_nlp_cost_conl_dims;

//
acados_size_t ocp_nlp_cost_conl_dims_calculate_size(void *config);
//
void *ocp_nlp_cost_conl_dims_assign(void *config, void *raw_memory);
//
void ocp_nlp_cost_conl_dims_initialize(void *config, void *dims, int nx, int nu, int ny, int ns, int nz);
//
void ocp_nlp_cost_conl_dims_set(void *config_, void *dims_, const char *field, int* value);
//
void ocp_nlp_cost_conl_dims_get(void *config_, void *dims_, const char *field, int* value);



/************************************************
 * model
 ************************************************/

typedef struct
{
    // slack penalty has the form z^T * s + .5 * s^T * Z * s
    external_function_generic *conl_cost_fun;
    external_function_generic *conl_cost_fun_jac_hess;
    struct blasfeo_dvec y_ref;
    struct blasfeo_dvec Z;              // diagonal Hessian of slacks as vector
    struct blasfeo_dvec z;              // gradient of slacks as vector
    double scaling;
} ocp_nlp_cost_conl_model;

//
acados_size_t ocp_nlp_cost_conl_model_calculate_size(void *config, void *dims);
//
void *ocp_nlp_cost_conl_model_assign(void *config, void *dims, void *raw_memory);
//
int ocp_nlp_cost_conl_model_set(void *config_, void *dims_, void *model_, const char *field, void *value_);



/************************************************
 * options
 ************************************************/

typedef struct
{
    bool gauss_newton_hess;  // dummy options, we always use a gauss-newton hessian
} ocp_nlp_cost_conl_opts;

//
acados_size_t ocp_nlp_cost_conl_opts_calculate_size(void *config, void *dims);
//
void *ocp_nlp_cost_conl_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_nlp_cost_conl_opts_initialize_default(void *config, void *dims, void *opts);
//
void ocp_nlp_cost_conl_opts_update(void *config, void *dims, void *opts);
//
void ocp_nlp_cost_conl_opts_set(void *config, void *opts, const char *field, void *value);



/************************************************
 * memory
 ************************************************/
typedef struct
{
    struct blasfeo_dvec grad;    // gradient of cost function
    struct blasfeo_dvec *ux;     // pointer to ux in nlp_out
    struct blasfeo_dvec *tmp_ux; // pointer to ux in tmp_nlp_out
    struct blasfeo_dmat *RSQrq;  // pointer to RSQrq in qp_in
    struct blasfeo_dvec *Z;      // pointer to Z in qp_in
    struct blasfeo_dvec *z_alg;         ///< pointer to z in sim_out
    struct blasfeo_dmat *dzdux_tran;    ///< pointer to sensitivity of a wrt ux in sim_out
    double fun;                         ///< value of the cost function
} ocp_nlp_cost_conl_memory;

//
acados_size_t ocp_nlp_cost_conl_memory_calculate_size(void *config, void *dims, void *opts);
//
void *ocp_nlp_cost_conl_memory_assign(void *config, void *dims, void *opts, void *raw_memory);
//
double *ocp_nlp_cost_conl_memory_get_fun_ptr(void *memory_);
//
struct blasfeo_dvec *ocp_nlp_cost_conl_memory_get_grad_ptr(void *memory_);
//
void ocp_nlp_cost_conl_memory_set_RSQrq_ptr(struct blasfeo_dmat *RSQrq, void *memory);
//
void ocp_nlp_cost_conl_memory_set_Z_ptr(struct blasfeo_dvec *Z, void *memory);
//
void ocp_nlp_cost_conl_memory_set_ux_ptr(struct blasfeo_dvec *ux, void *memory_);
//
void ocp_nlp_cost_conl_memory_set_tmp_ux_ptr(struct blasfeo_dvec *tmp_ux, void *memory_);
//
void ocp_nlp_cost_conl_memory_set_z_alg_ptr(struct blasfeo_dvec *z_alg, void *memory_);
//
void ocp_nlp_cost_conl_memory_set_dzdux_tran_ptr(struct blasfeo_dmat *dzdux_tran, void *memory_);

/************************************************
 * workspace
 ************************************************/

typedef struct
{
    struct blasfeo_dmat W;             // hessian of outer loss function
    struct blasfeo_dmat W_chol;        // cholesky factor of hessian of outer loss function
    struct blasfeo_dmat Jt_ux;         // jacobian of inner residual function
    struct blasfeo_dmat Jt_ux_tilde;   // jacobian of inner residual function plus gradient contribution of algebraic variables
    struct blasfeo_dmat Jt_z;          // jacobian of inner residual function wrt algebraic variables
    struct blasfeo_dmat tmp_nv_ny;
    struct blasfeo_dvec tmp_ny;
    struct blasfeo_dvec tmp_2ns;
} ocp_nlp_cost_conl_workspace;

//
acados_size_t ocp_nlp_cost_conl_workspace_calculate_size(void *config, void *dims, void *opts);

/************************************************
 * functions
 ************************************************/

//
void ocp_nlp_cost_conl_precompute(void *config_, void *dims_, void *model_, void *opts_, void *memory_, void *work_);
//
void ocp_nlp_cost_conl_config_initialize_default(void *config);
//
void ocp_nlp_cost_conl_initialize(void *config_, void *dims, void *model_, void *opts_, void *mem_, void *work_);
//
void ocp_nlp_cost_conl_update_qp_matrices(void *config_, void *dims, void *model_, void *opts_, void *memory_, void *work_);
//
void ocp_nlp_cost_conl_compute_fun(void *config_, void *dims, void *model_, void *opts_, void *memory_, void *work_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_COST_CONL_H_
/// @}
/// @}
/// @}
