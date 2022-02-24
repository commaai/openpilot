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


#ifndef ACADOS_DENSE_QP_DENSE_QP_QPOASES_H_
#define ACADOS_DENSE_QP_DENSE_QP_QPOASES_H_

#ifdef __cplusplus
extern "C" {
#endif

// blasfeo
#include "blasfeo/include/blasfeo_common.h"

// acados
#include "acados/dense_qp/dense_qp_common.h"
#include "acados/utils/types.h"

typedef struct dense_qp_qpoases_opts_
{
    double max_cputime;  // maximum cpu time in seconds
    int max_nwsr;        // maximum number of working set recalculations
    int warm_start;      // warm start with dual_sol in memory
    int use_precomputed_cholesky;
    int hotstart;  // this option requires constant data matrices! (eg linear MPC, inexact schemes
                   // with frozen sensitivities)
    int set_acado_opts;  // use same options as in acado code generation
    int compute_t;       // compute t in qp_out (to have correct residuals in NLP)
    double tolerance;  // terminationTolerance
} dense_qp_qpoases_opts;

typedef struct dense_qp_qpoases_memory_
{
    double *H;
    double *HH;
    double *R;
    double *g;
    double *gg;
    double *Zl;
    double *Zu;
    double *zl;
    double *zu;
    double *A;
    double *b;
    double *d_lb0;
    double *d_ub0;
    double *d_lb;
    double *d_ub;
    double *C;
    double *CC;
    double *d_lg0;
    double *d_ug0;
    double *d_lg;
    double *d_ug;
    double *d_ls;
    double *d_us;
    int *idxb;
    int *idxb_stacked;
    int *idxs;
    double *prim_sol;
    double *dual_sol;
    void *QPB;       // NOTE(giaf): cast to QProblemB to use
    void *QP;        // NOTE(giaf): cast to QProblem to use
    double cputime;  // cputime of qpoases
    int nwsr;        // performed number of working set recalculations
    int first_it;    // to be used with hotstart
    dense_qp_in *qp_stacked;
    double time_qp_solver_call; // equal to cputime
    int iter;

} dense_qp_qpoases_memory;

acados_size_t dense_qp_qpoases_opts_calculate_size(void *config, dense_qp_dims *dims);
//
void *dense_qp_qpoases_opts_assign(void *config, dense_qp_dims *dims, void *raw_memory);
//
void dense_qp_qpoases_opts_initialize_default(void *config, dense_qp_dims *dims, void *opts_);
//
void dense_qp_qpoases_opts_update(void *config, dense_qp_dims *dims, void *opts_);
//
acados_size_t dense_qp_qpoases__memorycalculate_size(void *config, dense_qp_dims *dims, void *opts_);
//
void *dense_qp_qpoases_memory_assign(void *config, dense_qp_dims *dims, void *opts_, void *raw_memory);
//
acados_size_t dense_qp_qpoases_workspace_calculate_size(void *config, dense_qp_dims *dims, void *opts_);
//
int dense_qp_qpoases(void *config, dense_qp_in *qp_in, dense_qp_out *qp_out, void *opts_, void *memory_, void *work_);
//
void dense_qp_qpoases_eval_sens(void *config_, void *qp_in, void *qp_out, void *opts_, void *mem_, void *work_);
//
void dense_qp_qpoases_config_initialize_default(void *config_);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_DENSE_QP_DENSE_QP_QPOASES_H_
