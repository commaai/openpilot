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


#ifndef INTERFACES_ACADOS_C_EXTERNAL_FUNCTION_INTERFACE_H_
#define INTERFACES_ACADOS_C_EXTERNAL_FUNCTION_INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/utils/external_function_generic.h"



/************************************************
 * generic external parametric function
 ************************************************/

//
void external_function_param_generic_create(external_function_param_generic *fun, int np);
//
void external_function_param_generic_free(external_function_param_generic *fun);



/************************************************
 * casadi external function
 ************************************************/

//
void external_function_casadi_create(external_function_casadi *fun);
//
void external_function_casadi_free(external_function_casadi *fun);
//
void external_function_casadi_create_array(int size, external_function_casadi *funs);
//
void external_function_casadi_free_array(int size, external_function_casadi *funs);



/************************************************
 * casadi external parametric function
 ************************************************/

//
void external_function_param_casadi_create(external_function_param_casadi *fun, int np);
//
void external_function_param_casadi_free(external_function_param_casadi *fun);
//
void external_function_param_casadi_create_array(int size, external_function_param_casadi *funs,
                                                 int np);
//
void external_function_param_casadi_free_array(int size, external_function_param_casadi *funs);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // INTERFACES_ACADOS_C_EXTERNAL_FUNCTION_INTERFACE_H_
