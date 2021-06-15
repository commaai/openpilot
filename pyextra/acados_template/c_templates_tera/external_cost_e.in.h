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


#ifndef {{ model.name }}_EXT_COST_E
#define {{ model.name }}_EXT_COST_E

#ifdef __cplusplus
extern "C" {
#endif

{% if cost.cost_ext_fun_type_e == "casadi" %}
{% if cost.cost_type_e == "EXTERNAL" %}
int {{ model.name }}_cost_ext_cost_e_fun(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_e_fun_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_e_fun_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_e_fun_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_e_fun_n_in();
int {{ model.name }}_cost_ext_cost_e_fun_n_out();

int {{ model.name }}_cost_ext_cost_e_fun_jac_hess(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_e_fun_jac_hess_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_e_fun_jac_hess_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_e_fun_jac_hess_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_e_fun_jac_hess_n_in();
int {{ model.name }}_cost_ext_cost_e_fun_jac_hess_n_out();

int {{ model.name }}_cost_ext_cost_e_fun_jac(const real_t** arg, real_t** res, int* iw, real_t* w, void *mem);
int {{ model.name }}_cost_ext_cost_e_fun_jac_work(int *, int *, int *, int *);
const int *{{ model.name }}_cost_ext_cost_e_fun_jac_sparsity_in(int);
const int *{{ model.name }}_cost_ext_cost_e_fun_jac_sparsity_out(int);
int {{ model.name }}_cost_ext_cost_e_fun_jac_n_in();
int {{ model.name }}_cost_ext_cost_e_fun_jac_n_out();
{% endif %}

{% else %}
int {{ cost.cost_function_ext_cost_e }}(void **, void **, void *);
{% endif %}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // {{ model.name }}_EXT_COST_E
