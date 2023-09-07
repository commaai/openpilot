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
/// \addtogroup ocp_nlp_constraints
/// @{

#ifndef ACADOS_OCP_NLP_OCP_NLP_CONSTRAINTS_BGH_H_
#define ACADOS_OCP_NLP_OCP_NLP_CONSTRAINTS_BGH_H_

#ifdef __cplusplus
extern "C" {
#endif

// acados
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_nlp/ocp_nlp_constraints_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/types.h"



/************************************************
 * dims
 ************************************************/

typedef struct
{
    int nx;
    int nu;
    int nz;
    int nb;  // nbx + nbu
    int nbu; // number of input box constraints
    int nbx; // number of state box constraints
    int ng;  // number of general linear constraints
    int nh;  // number of nonlinear path constraints
    int ns;  // nsbu + nsbx + nsg + nsh
    int nsbu;  // number of softened input bounds
    int nsbx;  // number of softened state bounds
    int nsg;  // number of softened general linear constraints
    int nsh;  // number of softened nonlinear constraints
    int nbue; // number of input box constraints which are equality
    int nbxe; // number of state box constraints which are equality
    int nge;  // number of general linear constraints which are equality
    int nhe;  // number of nonlinear path constraints which are equality
} ocp_nlp_constraints_bgh_dims;

//
acados_size_t ocp_nlp_constraints_bgh_dims_calculate_size(void *config);
//
void *ocp_nlp_constraints_bgh_dims_assign(void *config, void *raw_memory);
//
void ocp_nlp_constraints_bgh_dims_get(void *config_, void *dims_, const char *field, int* value);
//
void ocp_nlp_constraints_bgh_dims_set(void *config_, void *dims_,
                                      const char *field, const int* value);


/************************************************
 * model
 ************************************************/

typedef struct
{
    int *idxb;
    int *idxs;
    int *idxe;
    struct blasfeo_dvec d;  // gathers bounds
    struct blasfeo_dmat DCt;  // general linear constraint matrix
            // lg <= [D, C] * [u; x] <= ug
    external_function_generic *nl_constr_h_fun;  // nonlinear: lh <= h(x,u) <= uh
    external_function_generic *nl_constr_h_fun_jac;  // nonlinear: lh <= h(x,u) <= uh
    external_function_generic *nl_constr_h_fun_jac_hess;  // nonlinear: lh <= h(x,u) <= uh
} ocp_nlp_constraints_bgh_model;

//
acados_size_t ocp_nlp_constraints_bgh_model_calculate_size(void *config, void *dims);
//
void *ocp_nlp_constraints_bgh_model_assign(void *config, void *dims, void *raw_memory);
//
int ocp_nlp_constraints_bgh_model_set(void *config_, void *dims_,
                         void *model_, const char *field, void *value);

//
void ocp_nlp_constraints_bgh_model_get(void *config_, void *dims_,
                         void *model_, const char *field, void *value);


/************************************************
 * options
 ************************************************/

typedef struct
{
    int compute_adj;
    int compute_hess;
} ocp_nlp_constraints_bgh_opts;

//
acados_size_t ocp_nlp_constraints_bgh_opts_calculate_size(void *config, void *dims);
//
void *ocp_nlp_constraints_bgh_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_nlp_constraints_bgh_opts_initialize_default(void *config, void *dims, void *opts);
//
void ocp_nlp_constraints_bgh_opts_update(void *config, void *dims, void *opts);
//
void ocp_nlp_constraints_bgh_opts_set(void *config, void *opts, char *field, void *value);



/************************************************
 * memory
 ************************************************/

typedef struct
{
    struct blasfeo_dvec fun;
    struct blasfeo_dvec adj;
    struct blasfeo_dvec *ux;     // pointer to ux in nlp_out
    struct blasfeo_dvec *tmp_ux; // pointer to ux in tmp_nlp_out
    struct blasfeo_dvec *lam;    // pointer to lam in nlp_out
    struct blasfeo_dvec *tmp_lam;// pointer to lam in tmp_nlp_out
    struct blasfeo_dvec *z_alg;  // pointer to z_alg in ocp_nlp memory
    struct blasfeo_dmat *DCt;    // pointer to DCt in qp_in
    struct blasfeo_dmat *RSQrq;  // pointer to RSQrq in qp_in
    struct blasfeo_dmat *dzduxt; // pointer to dzduxt in ocp_nlp memory
    int *idxb;                   // pointer to idxb[ii] in qp_in
    int *idxs_rev;               // pointer to idxs_rev[ii] in qp_in
    int *idxe;                   // pointer to idxe[ii] in qp_in
} ocp_nlp_constraints_bgh_memory;

//
acados_size_t ocp_nlp_constraints_bgh_memory_calculate_size(void *config, void *dims, void *opts);
//
void *ocp_nlp_constraints_bgh_memory_assign(void *config, void *dims, void *opts, void *raw_memory);
//
struct blasfeo_dvec *ocp_nlp_constraints_bgh_memory_get_fun_ptr(void *memory_);
//
struct blasfeo_dvec *ocp_nlp_constraints_bgh_memory_get_adj_ptr(void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_ux_ptr(struct blasfeo_dvec *ux, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_tmp_ux_ptr(struct blasfeo_dvec *tmp_ux, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_lam_ptr(struct blasfeo_dvec *lam, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_tmp_lam_ptr(struct blasfeo_dvec *tmp_lam, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_DCt_ptr(struct blasfeo_dmat *DCt, void *memory);
//
void ocp_nlp_constraints_bgh_memory_set_RSQrq_ptr(struct blasfeo_dmat *RSQrq, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_z_alg_ptr(struct blasfeo_dvec *z_alg, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_dzduxt_ptr(struct blasfeo_dmat *dzduxt, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_idxb_ptr(int *idxb, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_idxs_rev_ptr(int *idxs_rev, void *memory_);
//
void ocp_nlp_constraints_bgh_memory_set_idxe_ptr(int *idxe, void *memory_);



/************************************************
 * workspace
 ************************************************/

typedef struct
{
    struct blasfeo_dmat tmp_nv_nv;
    struct blasfeo_dmat tmp_nz_nh;
    struct blasfeo_dmat tmp_nv_nh;
    struct blasfeo_dmat tmp_nz_nv;
    struct blasfeo_dmat hess_z;
    struct blasfeo_dvec tmp_ni;
    struct blasfeo_dvec tmp_nh;
} ocp_nlp_constraints_bgh_workspace;

//
acados_size_t ocp_nlp_constraints_bgh_workspace_calculate_size(void *config, void *dims, void *opts);

/* functions */

//
void ocp_nlp_constraints_bgh_config_initialize_default(void *config);
//
void ocp_nlp_constraints_bgh_initialize(void *config, void *dims, void *model, void *opts,
                                    void *mem, void *work);
//
void ocp_nlp_constraints_bgh_update_qp_matrices(void *config_, void *dims, void *model_,
                                            void *opts_, void *memory_, void *work_);

//
void ocp_nlp_constraints_bgh_compute_fun(void *config_, void *dims, void *model_,
                                            void *opts_, void *memory_, void *work_);
//
void ocp_nlp_constraints_bgh_bounds_update(void *config_, void *dims, void *model_,
                                            void *opts_, void *memory_, void *work_);


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_CONSTRAINTS_BGH_H_
/// @}
/// @}
