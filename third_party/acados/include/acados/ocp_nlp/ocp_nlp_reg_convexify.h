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
/// \addtogroup ocp_nlp_reg
/// @{

#ifndef ACADOS_OCP_NLP_OCP_NLP_REG_CONVEXIFY_H_
#define ACADOS_OCP_NLP_OCP_NLP_REG_CONVEXIFY_H_

#ifdef __cplusplus
extern "C" {
#endif



// blasfeo
#include "blasfeo/include/blasfeo_common.h"

// acados
#include "acados/ocp_nlp/ocp_nlp_reg_common.h"



/************************************************
 * dims
 ************************************************/

// use the functions in ocp_nlp_reg_common

/************************************************
 * options
 ************************************************/

typedef struct
{
    double delta;
    double epsilon;
//    double gamma; // 0.0
} ocp_nlp_reg_convexify_opts;

//
acados_size_t ocp_nlp_reg_convexify_opts_calculate_size(void);
//
void *ocp_nlp_reg_convexify_opts_assign(void *raw_memory);
//
void ocp_nlp_reg_convexify_opts_initialize_default(void *config_, ocp_nlp_reg_dims *dims, void *opts_);
//
void ocp_nlp_reg_convexify_opts_set(void *config_, ocp_nlp_reg_dims *dims, void *opts_, char *field, void* value);



/************************************************
 * memory
 ************************************************/

typedef struct {
    double *R;
    double *V; // TODO move to workspace
    double *d; // TODO move to workspace
    double *e; // TODO move to workspace
    double *reg_hess; // TODO move to workspace

    struct blasfeo_dmat Q_tilde;
    struct blasfeo_dmat Q_bar;
    struct blasfeo_dmat BAQ;
    struct blasfeo_dmat L;
    struct blasfeo_dmat delta_eye;
    struct blasfeo_dmat St_copy;

    struct blasfeo_dmat *original_RSQrq;
    struct blasfeo_dmat tmp_RSQ;

	struct blasfeo_dvec tmp_nuxM;
	struct blasfeo_dvec tmp_nbgM;

//    struct blasfeo_dvec grad;
//    struct blasfeo_dvec b2;

    // giaf's
    struct blasfeo_dmat **RSQrq;  // pointer to RSQrq in qp_in
    struct blasfeo_dvec **rq;  // pointer to rq in qp_in
    struct blasfeo_dmat **BAbt;  // pointer to BAbt in qp_in
    struct blasfeo_dvec **b;  // pointer to b in qp_in
    struct blasfeo_dmat **DCt;  // pointer to DCt in qp_in
    struct blasfeo_dvec **ux;  // pointer to ux in qp_out
    struct blasfeo_dvec **pi;  // pointer to pi in qp_out
    struct blasfeo_dvec **lam;  // pointer to lam in qp_out
	int **idxb; // pointer to idxb in qp_in

} ocp_nlp_reg_convexify_memory;

//
acados_size_t ocp_nlp_reg_convexify_calculate_memory_size(void *config, ocp_nlp_reg_dims *dims, void *opts);
//
void *ocp_nlp_reg_convexify_assign_memory(void *config, ocp_nlp_reg_dims *dims, void *opts, void *raw_memory);

/************************************************
 * workspace
 ************************************************/

 // TODO

/************************************************
 * functions
 ************************************************/

//
void ocp_nlp_reg_convexify_config_initialize_default(ocp_nlp_reg_config *config);

#ifdef __cplusplus
}
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_REG_CONVEXIFY_H_
/// @}
/// @}
