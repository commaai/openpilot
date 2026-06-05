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

#ifndef lat_MODEL
#define lat_MODEL

#ifdef __cplusplus
extern "C" {
#endif


/* explicit ODE */

// explicit ODE
int lat_expl_ode_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int lat_expl_ode_fun_work(int *, int *, int *, int *);
const int *lat_expl_ode_fun_sparsity_in(int);
const int *lat_expl_ode_fun_sparsity_out(int);
int lat_expl_ode_fun_n_in(void);
int lat_expl_ode_fun_n_out(void);

// explicit forward VDE
int lat_expl_vde_forw(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int lat_expl_vde_forw_work(int *, int *, int *, int *);
const int *lat_expl_vde_forw_sparsity_in(int);
const int *lat_expl_vde_forw_sparsity_out(int);
int lat_expl_vde_forw_n_in(void);
int lat_expl_vde_forw_n_out(void);

// explicit adjoint VDE
int lat_expl_vde_adj(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int lat_expl_vde_adj_work(int *, int *, int *, int *);
const int *lat_expl_vde_adj_sparsity_in(int);
const int *lat_expl_vde_adj_sparsity_out(int);
int lat_expl_vde_adj_n_in(void);
int lat_expl_vde_adj_n_out(void);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // lat_MODEL
