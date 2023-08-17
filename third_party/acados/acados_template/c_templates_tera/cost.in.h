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


#ifndef {{ model.name }}_COST
#define {{ model.name }}_COST

#ifdef __cplusplus
extern "C" {
#endif


// Cost at initial shooting node
{% if cost.cost_type_0 == "NONLINEAR_LS" %}
int {{ model.name }}_cost_y_0_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_0_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_0_fun_sparsity_in(int);
const int *{{ model.name }}_cost_y_0_fun_sparsity_out(int);
int {{ model.name }}_cost_y_0_fun_n_in(void);
int {{ model.name }}_cost_y_0_fun_n_out(void);

int {{ model.name }}_cost_y_0_fun_jac_ut_xt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_0_fun_jac_ut_xt_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_0_fun_jac_ut_xt_sparsity_in(int);
const int *{{ model.name }}_cost_y_0_fun_jac_ut_xt_sparsity_out(int);
int {{ model.name }}_cost_y_0_fun_jac_ut_xt_n_in(void);
int {{ model.name }}_cost_y_0_fun_jac_ut_xt_n_out(void);

int {{ model.name }}_cost_y_0_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_0_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_0_hess_sparsity_in(int);
const int *{{ model.name }}_cost_y_0_hess_sparsity_out(int);
int {{ model.name }}_cost_y_0_hess_n_in(void);
int {{ model.name }}_cost_y_0_hess_n_out(void);
{% elif cost.cost_type_0 == "CONVEX_OVER_NONLINEAR" %}

int {{ model.name }}_conl_cost_0_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_conl_cost_0_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_conl_cost_0_fun_sparsity_in(int);
const int *{{ model.name }}_conl_cost_0_fun_sparsity_out(int);
int {{ model.name }}_conl_cost_0_fun_n_in(void);
int {{ model.name }}_conl_cost_0_fun_n_out(void);

int {{ model.name }}_conl_cost_0_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_conl_cost_0_fun_jac_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_conl_cost_0_fun_jac_hess_sparsity_in(int);
const int *{{ model.name }}_conl_cost_0_fun_jac_hess_sparsity_out(int);
int {{ model.name }}_conl_cost_0_fun_jac_hess_n_in(void);
int {{ model.name }}_conl_cost_0_fun_jac_hess_n_out(void);

{% elif cost.cost_type_0 == "EXTERNAL" %}
    {%- if cost.cost_ext_fun_type_0 == "casadi" %}
int {{ model.name }}_cost_ext_cost_0_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_0_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_0_fun_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_0_fun_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_0_fun_n_in(void);
int {{ model.name }}_cost_ext_cost_0_fun_n_out(void);

int {{ model.name }}_cost_ext_cost_0_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_0_fun_jac_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_0_fun_jac_hess_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_0_fun_jac_hess_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_0_fun_jac_hess_n_in(void);
int {{ model.name }}_cost_ext_cost_0_fun_jac_hess_n_out(void);

int {{ model.name }}_cost_ext_cost_0_fun_jac(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_0_fun_jac_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_0_fun_jac_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_0_fun_jac_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_0_fun_jac_n_in(void);
int {{ model.name }}_cost_ext_cost_0_fun_jac_n_out(void);
    {%- else %}
int {{ cost.cost_function_ext_cost_0 }}(void **, void **, void *);
    {%- endif %}
{% endif %}


// Cost at path shooting node
{% if cost.cost_type == "NONLINEAR_LS" %}
int {{ model.name }}_cost_y_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_fun_sparsity_in(int);
const int *{{ model.name }}_cost_y_fun_sparsity_out(int);
int {{ model.name }}_cost_y_fun_n_in(void);
int {{ model.name }}_cost_y_fun_n_out(void);

int {{ model.name }}_cost_y_fun_jac_ut_xt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_fun_jac_ut_xt_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_fun_jac_ut_xt_sparsity_in(int);
const int *{{ model.name }}_cost_y_fun_jac_ut_xt_sparsity_out(int);
int {{ model.name }}_cost_y_fun_jac_ut_xt_n_in(void);
int {{ model.name }}_cost_y_fun_jac_ut_xt_n_out(void);

int {{ model.name }}_cost_y_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_hess_sparsity_in(int);
const int *{{ model.name }}_cost_y_hess_sparsity_out(int);
int {{ model.name }}_cost_y_hess_n_in(void);
int {{ model.name }}_cost_y_hess_n_out(void);

{% elif cost.cost_type == "CONVEX_OVER_NONLINEAR" %}
int {{ model.name }}_conl_cost_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_conl_cost_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_conl_cost_fun_sparsity_in(int);
const int *{{ model.name }}_conl_cost_fun_sparsity_out(int);
int {{ model.name }}_conl_cost_fun_n_in(void);
int {{ model.name }}_conl_cost_fun_n_out(void);

int {{ model.name }}_conl_cost_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_conl_cost_fun_jac_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_conl_cost_fun_jac_hess_sparsity_in(int);
const int *{{ model.name }}_conl_cost_fun_jac_hess_sparsity_out(int);
int {{ model.name }}_conl_cost_fun_jac_hess_n_in(void);
int {{ model.name }}_conl_cost_fun_jac_hess_n_out(void);
{% elif cost.cost_type == "EXTERNAL" %}
    {%- if cost.cost_ext_fun_type == "casadi" %}
int {{ model.name }}_cost_ext_cost_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_fun_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_fun_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_fun_n_in(void);
int {{ model.name }}_cost_ext_cost_fun_n_out(void);

int {{ model.name }}_cost_ext_cost_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_fun_jac_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_fun_jac_hess_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_fun_jac_hess_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_fun_jac_hess_n_in(void);
int {{ model.name }}_cost_ext_cost_fun_jac_hess_n_out(void);

int {{ model.name }}_cost_ext_cost_fun_jac(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_fun_jac_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_fun_jac_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_fun_jac_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_fun_jac_n_in(void);
int {{ model.name }}_cost_ext_cost_fun_jac_n_out(void);
    {%- else %}
int {{ cost.cost_function_ext_cost }}(void **, void **, void *);
    {%- endif %}
{% endif %}

// Cost at terminal shooting node
{% if cost.cost_type_e == "NONLINEAR_LS" %}
int {{ model.name }}_cost_y_e_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_e_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_e_fun_sparsity_in(int);
const int *{{ model.name }}_cost_y_e_fun_sparsity_out(int);
int {{ model.name }}_cost_y_e_fun_n_in(void);
int {{ model.name }}_cost_y_e_fun_n_out(void);

int {{ model.name }}_cost_y_e_fun_jac_ut_xt(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_e_fun_jac_ut_xt_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_e_fun_jac_ut_xt_sparsity_in(int);
const int *{{ model.name }}_cost_y_e_fun_jac_ut_xt_sparsity_out(int);
int {{ model.name }}_cost_y_e_fun_jac_ut_xt_n_in(void);
int {{ model.name }}_cost_y_e_fun_jac_ut_xt_n_out(void);

int {{ model.name }}_cost_y_e_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_y_e_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_y_e_hess_sparsity_in(int);
const int *{{ model.name }}_cost_y_e_hess_sparsity_out(int);
int {{ model.name }}_cost_y_e_hess_n_in(void);
int {{ model.name }}_cost_y_e_hess_n_out(void);
{% elif cost.cost_type_e == "CONVEX_OVER_NONLINEAR" %}
int {{ model.name }}_conl_cost_e_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_conl_cost_e_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_conl_cost_e_fun_sparsity_in(int);
const int *{{ model.name }}_conl_cost_e_fun_sparsity_out(int);
int {{ model.name }}_conl_cost_e_fun_n_in(void);
int {{ model.name }}_conl_cost_e_fun_n_out(void);

int {{ model.name }}_conl_cost_e_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_conl_cost_e_fun_jac_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_conl_cost_e_fun_jac_hess_sparsity_in(int);
const int *{{ model.name }}_conl_cost_e_fun_jac_hess_sparsity_out(int);
int {{ model.name }}_conl_cost_e_fun_jac_hess_n_in(void);
int {{ model.name }}_conl_cost_e_fun_jac_hess_n_out(void);
{% elif cost.cost_type_e == "EXTERNAL" %}
    {%- if cost.cost_ext_fun_type_e == "casadi" %}
int {{ model.name }}_cost_ext_cost_e_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_e_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_e_fun_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_e_fun_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_e_fun_n_in(void);
int {{ model.name }}_cost_ext_cost_e_fun_n_out(void);

int {{ model.name }}_cost_ext_cost_e_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_e_fun_jac_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_e_fun_jac_hess_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_e_fun_jac_hess_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_e_fun_jac_hess_n_in(void);
int {{ model.name }}_cost_ext_cost_e_fun_jac_hess_n_out(void);

int {{ model.name }}_cost_ext_cost_e_fun_jac(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_e_fun_jac_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_e_fun_jac_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_e_fun_jac_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_e_fun_jac_n_in(void);
int {{ model.name }}_cost_ext_cost_e_fun_jac_n_out(void);
    {%- else %}
int {{ cost.cost_function_ext_cost_e }}(void **, void **, void *);
    {%- endif %}
{% endif %}


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // {{ model.name }}_COST
