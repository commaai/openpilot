/**************************************************************************************************
*                                                                                                 *
* This file is part of HPIPM.                                                                     *
*                                                                                                 *
* HPIPM -- High-Performance Interior Point Method.                                                *
* Copyright (C) 2019 by Gianluca Frison.                                                          *
* Developed at IMTEK (University of Freiburg) under the supervision of Moritz Diehl.              *
* All rights reserved.                                                                            *
*                                                                                                 *
* The 2-Clause BSD License                                                                        *
*                                                                                                 *
* Redistribution and use in source and binary forms, with or without                              *
* modification, are permitted provided that the following conditions are met:                     *
*                                                                                                 *
* 1. Redistributions of source code must retain the above copyright notice, this                  *
*    list of conditions and the following disclaimer.                                             *
* 2. Redistributions in binary form must reproduce the above copyright notice,                    *
*    this list of conditions and the following disclaimer in the documentation                    *
*    and/or other materials provided with the distribution.                                       *
*                                                                                                 *
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                 *
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                   *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE                          *
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR                 *
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES                  *
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;                    *
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND                     *
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT                      *
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                   *
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    *
*                                                                                                 *
* Author: Gianluca Frison, gianluca.frison (at) imtek.uni-freiburg.de                             *
*                                                                                                 *
**************************************************************************************************/

#ifndef HPIPM_S_DENSE_QCQP_DIM_H_
#define HPIPM_S_DENSE_QCQP_DIM_H_

#include "hpipm_common.h"

#ifdef __cplusplus
extern "C" {
#endif



struct s_dense_qcqp_dim
	{
	struct s_dense_qp_dim *qp_dim; // dim of qp approximation
	int nv;  // number of variables
	int ne;  // number of equality constraints
	int nb;  // number of box constraints
	int ng;  // number of general constraints
	int nq;  // number of quadratic constraints
	int nsb; // number of softened box constraints
	int nsg; // number of softened general constraints
	int nsq; // number of softened quadratic constraints
	int ns;  // number of softened constraints (nsb+nsg+nsq) TODO number of slacks
	hpipm_size_t memsize;
	};



//
hpipm_size_t s_dense_qcqp_dim_memsize();
//
void s_dense_qcqp_dim_create(struct s_dense_qcqp_dim *dim, void *memory);
//
void s_dense_qcqp_dim_set(char *fiels_name, int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_nv(int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_ne(int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_nb(int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_ng(int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_nq(int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_nsb(int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_nsg(int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_nsq(int value, struct s_dense_qcqp_dim *dim);
//
void s_dense_qcqp_dim_set_ns(int value, struct s_dense_qcqp_dim *dim);



#ifdef __cplusplus
}	// #extern "C"
#endif



#endif // HPIPM_S_DENSE_QCQP_DIM_H_


