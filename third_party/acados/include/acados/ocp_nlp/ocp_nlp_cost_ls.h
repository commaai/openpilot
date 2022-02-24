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


/// \addtogroup ocp_nlp
/// @{
/// \addtogroup ocp_nlp_cost ocp_nlp_cost
/// @{
/// \addtogroup ocp_nlp_cost_ls ocp_nlp_cost_ls
/// \brief This module implements linear-least squares costs of the form
/// \f$\min_{x,u,z} \| V_x x + V_u u + V_z z - y_{\text{ref}}\|_W^2\f$.
/// @{



#ifndef ACADOS_OCP_NLP_OCP_NLP_COST_LS_H_
#define ACADOS_OCP_NLP_OCP_NLP_COST_LS_H_

#ifdef __cplusplus
extern "C" {
#endif

// blasfeo
#include "blasfeo/include/blasfeo_common.h"

// acados
#include "acados/ocp_nlp/ocp_nlp_cost_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/types.h"


////////////////////////////////////////////////////////////////////////////////
//                                     dims                                   //
////////////////////////////////////////////////////////////////////////////////

typedef struct
{
    int nx;  // number of states
    int nz;  // number of algebraic variables
    int nu;  // number of inputs
    int ny;  // number of outputs
    int ns;  // number of slacks
} ocp_nlp_cost_ls_dims;


///  Calculate the size of the ocp_nlp_cost_ls_dims struct
///
///  \param[in] config_ structure containing configuration of ocp_nlp_cost
///  module
///  \param[out] []
///  \return \c size of ocp_nlp_dims struct
acados_size_t ocp_nlp_cost_ls_dims_calculate_size(void *config);


///  Assign memory pointed to by raw_memory to ocp_nlp-cost_ls dims struct 
///
///  \param[in] config structure containing configuration of ocp_nlp_cost 
///  module
///  \param[in] raw_memory pointer to memory location  
///  \param[out] []
///  \return dims
void *ocp_nlp_cost_ls_dims_assign(void *config, void *raw_memory);


///  Initialize the dimensions struct of the 
///  ocp_nlp-cost_ls component    
///
///  \param[in] config structure containing configuration ocp_nlp-cost_ls component 
///  \param[in] nx number of states
///  \param[in] nu number of inputs
///  \param[in] ny number of residuals
///  \param[in] ns number of slacks
///  \param[in] nz number of algebraic variables
///  \param[out] dims
///  \return size
void ocp_nlp_cost_ls_dims_initialize(void *config, void *dims, int nx,
        int nu, int ny, int ns, int nz);

//
void ocp_nlp_cost_ls_dims_set(void *config_, void *dims_, const char *field, int* value);
//
void ocp_nlp_cost_ls_dims_get(void *config_, void *dims_, const char *field, int* value);


////////////////////////////////////////////////////////////////////////////////
//                                     model                                  //
////////////////////////////////////////////////////////////////////////////////


/// structure containing the data describing the linear least-square cost 
typedef struct
{
    // slack penalty has the form z^T * s + .5 * s^T * Z * s
    struct blasfeo_dmat Cyt;            ///< output matrix: Cy * [u,x] = y; in transposed form
    struct blasfeo_dmat Vz;             ///< Vz in ls cost Vx*x + Vu*u + Vz*z
    struct blasfeo_dmat W;              ///< ls norm corresponding to this matrix
    struct blasfeo_dvec y_ref;          ///< yref
    struct blasfeo_dvec Z;              ///< diagonal Hessian of slacks as vector (lower and upper)
    struct blasfeo_dvec z;              ///< gradient of slacks as vector (lower and upper)
    double scaling;
} ocp_nlp_cost_ls_model;

//
acados_size_t ocp_nlp_cost_ls_model_calculate_size(void *config, void *dims);
//
void *ocp_nlp_cost_ls_model_assign(void *config, void *dims, void *raw_memory);
//
int ocp_nlp_cost_ls_model_set(void *config_, void *dims_, void *model_,
                              const char *field, void *value_);



////////////////////////////////////////////////////////////////////////////////
//                                   options                                  //
////////////////////////////////////////////////////////////////////////////////



typedef struct
{
    int dummy; // struct can't be void
} ocp_nlp_cost_ls_opts;

//
acados_size_t ocp_nlp_cost_ls_opts_calculate_size(void *config, void *dims);
//
void *ocp_nlp_cost_ls_opts_assign(void *config, void *dims, void *raw_memory);
//
void ocp_nlp_cost_ls_opts_initialize_default(void *config, void *dims, void *opts);
//
void ocp_nlp_cost_ls_opts_update(void *config, void *dims, void *opts);
//
void ocp_nlp_cost_ls_opts_set(void *config, void *opts, const char *field, void *value);



////////////////////////////////////////////////////////////////////////////////
//                                     memory                                 //
////////////////////////////////////////////////////////////////////////////////



/// structure containing the memory associated with cost_ls component 
/// of the ocp_nlp module
typedef struct
{
    struct blasfeo_dmat hess;           ///< hessian of cost function
    struct blasfeo_dmat W_chol;         ///< cholesky factor of weight matrix
    struct blasfeo_dvec res;            ///< ls residual r(x)
    struct blasfeo_dvec grad;           ///< gradient of cost function
    struct blasfeo_dvec *ux;            ///< pointer to ux in nlp_out
    struct blasfeo_dvec *tmp_ux;        ///< pointer to ux in tmp_nlp_out
    struct blasfeo_dvec *z_alg;         ///< pointer to z in sim_out
    struct blasfeo_dmat *dzdux_tran;    ///< pointer to sensitivity of a wrt ux in sim_out
    struct blasfeo_dmat *RSQrq;         ///< pointer to RSQrq in qp_in
    struct blasfeo_dvec *Z;             ///< pointer to Z in qp_in
	double fun;                         ///< value of the cost function
} ocp_nlp_cost_ls_memory;

//
acados_size_t ocp_nlp_cost_ls_memory_calculate_size(void *config, void *dims, void *opts);
//
void *ocp_nlp_cost_ls_memory_assign(void *config, void *dims, void *opts, void *raw_memory);
//
double *ocp_nlp_cost_ls_memory_get_fun_ptr(void *memory_);
//
struct blasfeo_dvec *ocp_nlp_cost_ls_memory_get_grad_ptr(void *memory_);
//
void ocp_nlp_cost_ls_memory_set_RSQrq_ptr(struct blasfeo_dmat *RSQrq, void *memory);
//
void ocp_nlp_cost_ls_memory_set_Z_ptr(struct blasfeo_dvec *Z, void *memory);
//
void ocp_nlp_cost_ls_memory_set_ux_ptr(struct blasfeo_dvec *ux, void *memory_);
//
void ocp_nlp_cost_ls_memory_set_tmp_ux_ptr(struct blasfeo_dvec *tmp_ux, void *memory_);
//
void ocp_nlp_cost_ls_memory_set_z_alg_ptr(struct blasfeo_dvec *z_alg, void *memory_);
//
void ocp_nlp_cost_ls_memory_set_dzdux_tran_ptr(struct blasfeo_dmat *dzdux_tran, void *memory_);



////////////////////////////////////////////////////////////////////////////////
//                                 workspace                                  //
////////////////////////////////////////////////////////////////////////////////



typedef struct
{
    struct blasfeo_dmat tmp_nv_ny;   // temporary matrix of dimensions nv, ny
    struct blasfeo_dmat Cyt_tilde;   // updated Cyt (after z elimination)
    struct blasfeo_dmat dzdux_tran;  // derivatives of z wrt u and x (tran)
    struct blasfeo_dvec tmp_ny;      // temporary vector of dimension ny
    struct blasfeo_dvec tmp_2ns;     // temporary vector of dimension ny
    struct blasfeo_dvec tmp_nz;      // temporary vector of dimension nz
    struct blasfeo_dvec y_ref_tilde; // updated y_ref (after z elimination)
} ocp_nlp_cost_ls_workspace;

//
acados_size_t ocp_nlp_cost_ls_workspace_calculate_size(void *config, void *dims, void *opts);



////////////////////////////////////////////////////////////////////////////////
//                                 functions                                  //
////////////////////////////////////////////////////////////////////////////////



//
void ocp_nlp_cost_ls_config_initialize_default(void *config);
//
void ocp_nlp_cost_ls_initialize(void *config_, void *dims, void *model_, void *opts_,
                                void *mem_, void *work_);
//
void ocp_nlp_cost_ls_update_qp_matrices(void *config_, void *dims, void *model_,
                                        void *opts_, void *memory_, void *work_);
//
void ocp_nlp_cost_ls_compute_fun(void *config_, void *dims, void *model_, void *opts_, void *memory_, void *work_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_COST_LS_H_
/// @}
/// @}
/// @}
