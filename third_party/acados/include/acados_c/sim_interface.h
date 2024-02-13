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


#ifndef INTERFACES_ACADOS_C_SIM_INTERFACE_H_
#define INTERFACES_ACADOS_C_SIM_INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/sim/sim_common.h"



typedef enum
{
    ERK,
    IRK,
    GNSF,
    LIFTED_IRK,
    INVALID_SIM_SOLVER,
} sim_solver_t;



typedef struct
{
    sim_solver_t sim_solver;
} sim_solver_plan_t;



typedef struct
{
    sim_config *config;
    void *dims;
    void *opts;
    void *mem;
    void *work;
} sim_solver;



/* config */
//
ACADOS_SYMBOL_EXPORT sim_config *sim_config_create(sim_solver_plan_t plan);
//
ACADOS_SYMBOL_EXPORT void sim_config_destroy(void *config);

/* dims */
//
ACADOS_SYMBOL_EXPORT void *sim_dims_create(void *config_);
//
ACADOS_SYMBOL_EXPORT void sim_dims_destroy(void *dims);
//
ACADOS_SYMBOL_EXPORT void sim_dims_set(sim_config *config, void *dims, const char *field, const int* value);
//
ACADOS_SYMBOL_EXPORT void sim_dims_get(sim_config *config, void *dims, const char *field, int* value);
//
ACADOS_SYMBOL_EXPORT void sim_dims_get_from_attr(sim_config *config, void *dims, const char *field, int *dims_out);

/* in */
//
ACADOS_SYMBOL_EXPORT sim_in *sim_in_create(sim_config *config, void *dims);
//
ACADOS_SYMBOL_EXPORT void sim_in_destroy(void *out);
//
ACADOS_SYMBOL_EXPORT int sim_in_set(void *config_, void *dims_, sim_in *in, const char *field, void *value);


/* out */
//
ACADOS_SYMBOL_EXPORT sim_out *sim_out_create(sim_config *config, void *dims);
//
ACADOS_SYMBOL_EXPORT void sim_out_destroy(void *out);
//
ACADOS_SYMBOL_EXPORT int sim_out_get(void *config, void *dims, sim_out *out, const char *field, void *value);

/* opts */
//
ACADOS_SYMBOL_EXPORT void *sim_opts_create(sim_config *config, void *dims);
//
ACADOS_SYMBOL_EXPORT void sim_opts_destroy(void *opts);
//
ACADOS_SYMBOL_EXPORT void sim_opts_set(sim_config *config, void *opts, const char *field, void *value);
//
ACADOS_SYMBOL_EXPORT void sim_opts_get(sim_config *config, void *opts, const char *field, void *value);

/* solver */
//
ACADOS_SYMBOL_EXPORT acados_size_t sim_calculate_size(sim_config *config, void *dims, void *opts_);
//
ACADOS_SYMBOL_EXPORT sim_solver *sim_assign(sim_config *config, void *dims, void *opts_, void *raw_memory);
//
ACADOS_SYMBOL_EXPORT sim_solver *sim_solver_create(sim_config *config, void *dims, void *opts_);
//
ACADOS_SYMBOL_EXPORT void sim_solver_destroy(void *solver);
//
ACADOS_SYMBOL_EXPORT int sim_solve(sim_solver *solver, sim_in *in, sim_out *out);
//
ACADOS_SYMBOL_EXPORT int sim_precompute(sim_solver *solver, sim_in *in, sim_out *out);
//
ACADOS_SYMBOL_EXPORT int sim_solver_set(sim_solver *solver, const char *field, void *value);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // INTERFACES_ACADOS_C_SIM_INTERFACE_H_
