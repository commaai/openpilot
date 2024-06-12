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


#ifndef ACADOS_SIM_SIM_COLLOCATION_UTILS_H_
#define ACADOS_SIM_SIM_COLLOCATION_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/utils/types.h"



// enum Newton_type_collocation
// {
//     exact = 0,
//     simplified_in,
//     simplified_inis
// };



// typedef struct
// {
//     enum Newton_type_collocation type;
//     double *eig;
//     double *low_tria;
//     bool single;
//     bool freeze;

//     double *transf1;
//     double *transf2;

//     double *transf1_T;
//     double *transf2_T;
// } Newton_scheme;


typedef enum
{
    GAUSS_LEGENDRE,
    GAUSS_RADAU_IIA,
} sim_collocation_type;


//
// acados_size_t gauss_legendre_nodes_work_calculate_size(int ns);
//
// void gauss_legendre_nodes(int ns, double *nodes, void *raw_memory);
//
// acados_size_t gauss_simplified_work_calculate_size(int ns);
// //
// void gauss_simplified(int ns, Newton_scheme *scheme, void *work);
//
acados_size_t butcher_tableau_work_calculate_size(int ns);
//
// void calculate_butcher_tableau_from_nodes(int ns, double *nodes, double *b, double *A, void *work);
//
void calculate_butcher_tableau(int ns, sim_collocation_type collocation_type, double *c_vec,
                               double *b_vec, double *A_mat, void *work);



#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SIM_SIM_COLLOCATION_UTILS_H_
