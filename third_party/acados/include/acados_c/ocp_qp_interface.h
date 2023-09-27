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


#ifndef INTERFACES_ACADOS_C_OCP_QP_INTERFACE_H_
#define INTERFACES_ACADOS_C_OCP_QP_INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_qp/ocp_qp_xcond_solver.h"


/// QP solver types (Enumeration).
///
/// Full list of fields:
///   PARTIAL_CONDENSING_HPIPM
///   PARTIAL_CONDENSING_HPMPC
///   PARTIAL_CONDENSING_OOQP
///   PARTIAL_CONDENSING_OSQP
///   PARTIAL_CONDENSING_QPDUNES
///   FULL_CONDENSING_HPIPM
///   FULL_CONDENSING_QPOASES
///   FULL_CONDENSING_QORE
///   FULL_CONDENSING_OOQP
///   INVALID_QP_SOLVER
///
/// Note: In this enumeration the partial condensing solvers have to be
///       specified before the full condensing solvers.
typedef enum {
    PARTIAL_CONDENSING_HPIPM,
#ifdef ACADOS_WITH_HPMPC
    PARTIAL_CONDENSING_HPMPC,
#else
    PARTIAL_CONDENSING_HPMPC_NOT_AVAILABLE,
#endif
#ifdef ACADOS_WITH_OOQP
    PARTIAL_CONDENSING_OOQP,
#else
    PARTIAL_CONDENSING_OOQP_NOT_AVAILABLE,
#endif
#ifdef ACADOS_WITH_OSQP
    PARTIAL_CONDENSING_OSQP,
#else
    PARTIAL_CONDENSING_OSQP_NOT_AVAILABLE,
#endif
#ifdef ACADOS_WITH_QPDUNES
    PARTIAL_CONDENSING_QPDUNES,
#else
    PARTIAL_CONDENSING_QPDUNES_NOT_AVAILABLE,
#endif
    FULL_CONDENSING_HPIPM,
#ifdef ACADOS_WITH_QPOASES
    FULL_CONDENSING_QPOASES,
#else
    FULL_CONDENSING_QPOASES_NOT_AVAILABLE,
#endif
#ifdef ACADOS_WITH_QORE
    FULL_CONDENSING_QORE,
#else
    FULL_CONDENSING_QORE_NOT_AVAILABLE,
#endif
#ifdef ACADOS_WITH_OOQP
    FULL_CONDENSING_OOQP,
#else
    FULL_CONDENSING_OOQP_NOT_AVAILABLE,
#endif
    INVALID_QP_SOLVER,
} ocp_qp_solver_t;


/// Struct containing qp solver
typedef struct
{
    ocp_qp_solver_t qp_solver;
} ocp_qp_solver_plan_t;


/// Linear ocp configuration.
typedef struct
{
    ocp_qp_xcond_solver_config *config;
    ocp_qp_xcond_solver_dims *dims;
    void *opts;
    void *mem;
    void *work;
} ocp_qp_solver;


/// Initializes the qp solver configuration.
/// TBC should this be private/static - no, used in ocp_nlp
void ocp_qp_xcond_solver_config_initialize_from_plan(
    ocp_qp_solver_t solver_name, ocp_qp_xcond_solver_config *solver_config);

/// Constructs a qp solver config and Initializes with default values.
///
/// \param plan The qp solver plan struct.
ocp_qp_xcond_solver_config *ocp_qp_xcond_solver_config_create(ocp_qp_solver_plan_t plan);

/// Destructor for config struct, frees memory.
///
/// \param config The config object to destroy.
void ocp_qp_xcond_solver_config_free(ocp_qp_xcond_solver_config *config);


/// Constructs a struct that contains the dimensions for the variables of the qp.
///
/// \param N The number of variables.
ocp_qp_dims *ocp_qp_dims_create(int N);

/// Destructor of The dimension struct.
///
/// \param dims The dimension struct.
void ocp_qp_dims_free(void *dims);

//
ocp_qp_xcond_solver_dims *ocp_qp_xcond_solver_dims_create(ocp_qp_xcond_solver_config *config, int N);
//
ocp_qp_xcond_solver_dims *ocp_qp_xcond_solver_dims_create_from_ocp_qp_dims(
    ocp_qp_xcond_solver_config *config, ocp_qp_dims *dims);
//
void ocp_qp_xcond_solver_dims_free(ocp_qp_xcond_solver_dims *dims_);

void ocp_qp_xcond_solver_dims_set(void *config_, ocp_qp_xcond_solver_dims *dims,
                                  int stage, const char *field, int* value);


/// Constructs an input object for the qp.
///
/// \param dims The dimension struct.
ocp_qp_in *ocp_qp_in_create(ocp_qp_dims *dims);


void ocp_qp_in_set(ocp_qp_xcond_solver_config *config, ocp_qp_in *in,
                   int stage, char *field, void *value);

/// Destructor of the inputs struct.
///
/// \param in_ The inputs struct.
void ocp_qp_in_free(void *in_);


/// Constructs an outputs object for the qp.
///
/// \param dims The dimension struct.
ocp_qp_out *ocp_qp_out_create(ocp_qp_dims *dims);

/// Destructor of the output struct.
///
/// \param out_ The output struct.
void ocp_qp_out_free(void *out_);


/// Getter of output struct
void ocp_qp_out_get(ocp_qp_out *out, const char *field, void *value);


/// Constructs an options object for the qp.
///
/// \param config The configuration struct.
/// \param dims The dimension struct.
void *ocp_qp_xcond_solver_opts_create(ocp_qp_xcond_solver_config *config,
                                      ocp_qp_xcond_solver_dims *dims);

/// Destructor of the options struct.
///
/// \param opts The options struct to destroy.
void ocp_qp_xcond_solver_opts_free(ocp_qp_xcond_solver_opts *opts);


/// Setter of the options struct.
///
/// \param opts The options struct.
void ocp_qp_xcond_solver_opts_set(ocp_qp_xcond_solver_config *config,
           ocp_qp_xcond_solver_opts *opts, const char *field, void* value);

/// TBC Should be private/static?
acados_size_t ocp_qp_calculate_size(ocp_qp_xcond_solver_config *config, ocp_qp_xcond_solver_dims *dims, void *opts_);


/// TBC Reserves memory? TBC Should this be private?
///
/// \param config The configuration struct.
/// \param dims The dimension struct.
/// \param opts_ The options struct.
/// \param raw_memory Pointer to raw memory to assign to qp solver.
ocp_qp_solver *ocp_qp_assign(ocp_qp_xcond_solver_config *config, ocp_qp_xcond_solver_dims *dims,
                             void *opts_, void *raw_memory);

/// Creates a qp solver. Reserves memory.
///
/// \param config The configuration struct.
/// \param dims The dimension struct.
/// \param opts_ The options struct.
ocp_qp_solver *ocp_qp_create(ocp_qp_xcond_solver_config *config,
                             ocp_qp_xcond_solver_dims *dims, void *opts_);


/// Destroys a qp solver. Frees memory.
///
/// \param solver The qp solver
void ocp_qp_solver_destroy(ocp_qp_solver *solver);

void ocp_qp_x_cond_solver_free(ocp_qp_xcond_solver_config *config,
                             ocp_qp_xcond_solver_dims *dims, void *opts_);


/// Solves the qp.
///
/// \param solver The solver.
/// \param qp_in The inputs struct.
/// \param qp_out The output struct.
int ocp_qp_solve(ocp_qp_solver *solver, ocp_qp_in *qp_in, ocp_qp_out *qp_out);


/// Calculates the infinity norm of the residuals.
///
/// \param dims The dimension struct.
/// \param qp_in The inputs struct.
/// \param qp_out The output struct.
/// \param res Output array for the residuals.
void ocp_qp_inf_norm_residuals(ocp_qp_dims *dims, ocp_qp_in *qp_in, ocp_qp_out *qp_out,
                               double *res);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // INTERFACES_ACADOS_C_OCP_QP_INTERFACE_H_
