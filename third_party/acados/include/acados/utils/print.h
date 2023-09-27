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


#ifndef ACADOS_UTILS_PRINT_H_
#define ACADOS_UTILS_PRINT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "acados/dense_qp/dense_qp_common.h"
#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_qp/ocp_qp_common_frontend.h"
#include "acados/utils/types.h"

// void print_matrix(char *file_name, const real_t *matrix, const int_t nrows, const int_t ncols);

// void print_matrix_name(char *file_name, char *name, const real_t *matrix, const int_t nrows,
//                        const int_t ncols);

// void print_int_matrix(char *file_name, const int_t *matrix, const int_t nrows, const int_t ncols);

// void print_array(char *file_name, real_t *array, int_t size);

// void print_int_array(char *file_name, const int_t *array, int_t size);

void read_matrix(const char *file_name, real_t *array, const int_t nrows, const int_t ncols);

void write_double_vector_to_txt(real_t *vec, int_t n, const char *fname);

// ocp nlp
// TODO(andrea): inconsistent naming
void ocp_nlp_dims_print(ocp_nlp_dims *dims);
// TODO(andrea): inconsistent naming
void ocp_nlp_out_print(ocp_nlp_dims *dims, ocp_nlp_out *nlp_out);
// TODO(andrea): inconsistent naming
void ocp_nlp_res_print(ocp_nlp_dims *dims, ocp_nlp_res *nlp_res);

// ocp qp
void print_ocp_qp_dims(ocp_qp_dims *dims);

// void print_dense_qp_dims(dense_qp_dims *dims);

void print_ocp_qp_in(ocp_qp_in *qp_in);

void print_ocp_qp_out(ocp_qp_out *qp_out);

// void print_ocp_qp_in_to_string(char string_out[], ocp_qp_in *qp_in);

// void print_ocp_qp_out_to_string(char string_out[], ocp_qp_out *qp_out);

void print_ocp_qp_res(ocp_qp_res *qp_res);

// void print_colmaj_ocp_qp_in(colmaj_ocp_qp_in *qp);

// void print_colmaj_ocp_qp_in_to_file(colmaj_ocp_qp_in *qp);

// void print_colmaj_ocp_qp_out(char *filename, colmaj_ocp_qp_in *qp, colmaj_ocp_qp_out *out);

void print_dense_qp_in(dense_qp_in *qp_in);

void print_qp_info(qp_info *info);

// void acados_warning(char warning_string[]);

// void acados_error(char error_string[]);

// void acados_not_implemented(char feature_string[]);

// blasfeo
// void print_blasfeo_target();

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_UTILS_PRINT_H_
