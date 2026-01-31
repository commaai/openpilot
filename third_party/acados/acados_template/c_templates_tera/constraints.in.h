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

#ifndef {{ model.name }}_CONSTRAINTS
#define {{ model.name }}_CONSTRAINTS

#ifdef __cplusplus
extern "C" {
#endif

{% if dims.nphi > 0 %}
int {{ model.name }}_phi_constraint(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_phi_constraint_work(int *, int *, int *, int *);
const int *{{ model.name }}_phi_constraint_sparsity_in(int);
const int *{{ model.name }}_phi_constraint_sparsity_out(int);
int {{ model.name }}_phi_constraint_n_in(void);
int {{ model.name }}_phi_constraint_n_out(void);
{% endif %}

{% if dims.nphi_e > 0 %}
int {{ model.name }}_phi_e_constraint(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_phi_e_constraint_work(int *, int *, int *, int *);
const int *{{ model.name }}_phi_e_constraint_sparsity_in(int);
const int *{{ model.name }}_phi_e_constraint_sparsity_out(int);
int {{ model.name }}_phi_e_constraint_n_in(void);
int {{ model.name }}_phi_e_constraint_n_out(void);
{% endif %}

{% if dims.nh > 0 %}
int {{ model.name }}_constr_h_fun_jac_uxt_zt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_constr_h_fun_jac_uxt_zt_work(int *, int *, int *, int *);
const int *{{ model.name }}_constr_h_fun_jac_uxt_zt_sparsity_in(int);
const int *{{ model.name }}_constr_h_fun_jac_uxt_zt_sparsity_out(int);
int {{ model.name }}_constr_h_fun_jac_uxt_zt_n_in(void);
int {{ model.name }}_constr_h_fun_jac_uxt_zt_n_out(void);

int {{ model.name }}_constr_h_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_constr_h_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_constr_h_fun_sparsity_in(int);
const int *{{ model.name }}_constr_h_fun_sparsity_out(int);
int {{ model.name }}_constr_h_fun_n_in(void);
int {{ model.name }}_constr_h_fun_n_out(void);

{% if solver_options.hessian_approx == "EXACT" -%}
int {{ model.name }}_constr_h_fun_jac_uxt_zt_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_constr_h_fun_jac_uxt_zt_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_constr_h_fun_jac_uxt_zt_hess_sparsity_in(int);
const int *{{ model.name }}_constr_h_fun_jac_uxt_zt_hess_sparsity_out(int);
int {{ model.name }}_constr_h_fun_jac_uxt_zt_hess_n_in(void);
int {{ model.name }}_constr_h_fun_jac_uxt_zt_hess_n_out(void);
{% endif %}
{% endif %}

{% if dims.nh_e > 0 %}
int {{ model.name }}_constr_h_e_fun_jac_uxt_zt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_constr_h_e_fun_jac_uxt_zt_work(int *, int *, int *, int *);
const int *{{ model.name }}_constr_h_e_fun_jac_uxt_zt_sparsity_in(int);
const int *{{ model.name }}_constr_h_e_fun_jac_uxt_zt_sparsity_out(int);
int {{ model.name }}_constr_h_e_fun_jac_uxt_zt_n_in(void);
int {{ model.name }}_constr_h_e_fun_jac_uxt_zt_n_out(void);

int {{ model.name }}_constr_h_e_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_constr_h_e_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_constr_h_e_fun_sparsity_in(int);
const int *{{ model.name }}_constr_h_e_fun_sparsity_out(int);
int {{ model.name }}_constr_h_e_fun_n_in(void);
int {{ model.name }}_constr_h_e_fun_n_out(void);

{% if solver_options.hessian_approx == "EXACT" -%}
int {{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_sparsity_in(int);
const int *{{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_sparsity_out(int);
int {{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_n_in(void);
int {{ model.name }}_constr_h_e_fun_jac_uxt_zt_hess_n_out(void);
{% endif %}
{% endif %}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // {{ model.name }}_CONSTRAINTS
