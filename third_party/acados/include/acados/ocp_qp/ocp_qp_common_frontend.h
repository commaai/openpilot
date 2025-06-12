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


#ifndef ACADOS_OCP_QP_OCP_QP_COMMON_FRONTEND_H_
#define ACADOS_OCP_QP_OCP_QP_COMMON_FRONTEND_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/ocp_qp/ocp_qp_common.h"

typedef struct
{
    int N;
    int *nx;
    int *nu;
    int *nb;
    int *nc;
    double **A;
    double **B;
    double **b;
    double **Q;
    double **S;
    double **R;
    double **q;
    double **r;
    int **idxb;
    double **lb;
    double **ub;
    double **Cx;
    double **Cu;
    double **lc;
    double **uc;
} colmaj_ocp_qp_in;

typedef struct
{
    double **x;
    double **u;
    double **pi;
    double **lam;
} colmaj_ocp_qp_out;

typedef struct
{
    double **res_r;
    double **res_q;
    double **res_ls;
    double **res_us;
    double **res_b;
    double **res_d_lb;
    double **res_d_ub;
    double **res_d_lg;
    double **res_d_ug;
    double **res_d_ls;
    double **res_d_us;
    double **res_m_lb;
    double **res_m_ub;
    double **res_m_lg;
    double **res_m_ug;
    double **res_m_ls;
    double **res_m_us;
    double res_nrm_inf[4];
} colmaj_ocp_qp_res;

//
acados_size_t colmaj_ocp_qp_in_calculate_size(ocp_qp_dims *dims);
//
char *assign_colmaj_ocp_qp_in(ocp_qp_dims *dims, colmaj_ocp_qp_in **qp_in, void *ptr);
//
acados_size_t colmaj_ocp_qp_out_calculate_size(ocp_qp_dims *dims);
//
char *assign_colmaj_ocp_qp_out(ocp_qp_dims *dims, colmaj_ocp_qp_out **qp_out, void *ptr);
//
acados_size_t colmaj_ocp_qp_res_calculate_size(ocp_qp_dims *dims);
//
char *assign_colmaj_ocp_qp_res(ocp_qp_dims *dims, colmaj_ocp_qp_res **qp_res, void *ptr);
//
void convert_colmaj_to_ocp_qp_in(colmaj_ocp_qp_in *cm_qp_in, ocp_qp_in *qp_in);
//
void convert_ocp_qp_out_to_colmaj(ocp_qp_out *qp_out, colmaj_ocp_qp_out *cm_qp_out);
//
void convert_ocp_qp_res_to_colmaj(ocp_qp_res *qp_res, colmaj_ocp_qp_res *cm_qp_res);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_OCP_QP_OCP_QP_COMMON_FRONTEND_H_
