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


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_{{ model.name }}.h"
// mex
#include "mex.h"

/* auxilary mex */
// prints a matrix in column-major format (exponential notation)
void MEX_print_exp_mat(int m, int n, double *A, int lda)
{
	for (int i=0; i<m; i++)
    {
		for (int j=0; j<n; j++)
        {
			mexPrintf("%e\t", A[i+lda*j]);
        }
		mexPrintf("\n");
    }
	mexPrintf("\n");
}

// prints the transposed of a matrix in column-major format (exponential notation)
void MEX_print_exp_tran_mat(int row, int col, double *A, int lda)
{
	for (int j=0; j<col; j++)
    {
		for (int i=0; i<row; i++)
        {
			mexPrintf("%e\t", A[i+lda*j]);
        }
		mexPrintf("\n");
    }
	mexPrintf("\n");
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    int status = 0;
    status = {{ model.name }}_acados_create();

    if (status)
    {
        mexPrintf("{{ model.name }}_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }


    // get pointers to nlp solver related objects
    ocp_nlp_config *nlp_config = {{ model.name }}_acados_get_nlp_config();
    ocp_nlp_dims *nlp_dims = {{ model.name }}_acados_get_nlp_dims();
    ocp_nlp_in *nlp_in = {{ model.name }}_acados_get_nlp_in();
    ocp_nlp_out *nlp_out = {{ model.name }}_acados_get_nlp_out();
    ocp_nlp_solver *nlp_solver = {{ model.name }}_acados_get_nlp_solver();
    void *nlp_opts = {{ model.name }}_acados_get_nlp_opts();

    // initial condition
    int idxbx0[{{ dims.nbx_0 }}];
    {% for i in range(end=dims.nbx_0) %}
    idxbx0[{{ i }}] = {{ constraints.idxbx_0[i] }};
    {%- endfor %}

    double lbx0[{{ dims.nbx_0 }}];
    double ubx0[{{ dims.nbx_0 }}];
    {% for i in range(end=dims.nbx_0) %}
    lbx0[{{ i }}] = {{ constraints.lbx_0[i] }};
    ubx0[{{ i }}] = {{ constraints.ubx_0[i] }};
    {%- endfor %}

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[{{ dims.nx }}];
    {%- for i in range(end=dims.nx) %}
    x_init[{{ i }}] = 0.0;
    {%- endfor %}

    // initial value for control input
    double u0[{{ dims.nu }}];
    {%- for i in range(end=dims.nu) %}
    u0[{{ i }}] = 0.0;
    {%- endfor %}

    // prepare evaluation
    int NTIMINGS = 10;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[{{ dims.nx }} * ({{ dims.N }}+1)];
    double utraj[{{ dims.nu }} * ({{ dims.N }})];

    // solve ocp in loop
    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize primal solution
        for (int i = 0; i <= nlp_dims->N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        status = {{ model.name }}_acados_solve();
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*{{ dims.nx }}]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*{{ dims.nu }}]);

    mexPrintf("\n--- xtraj ---\n");
    MEX_print_exp_tran_mat( {{ dims.nx }}, {{ dims.N }}+1, xtraj, {{ dims.nx }} );
    mexPrintf("\n--- utraj ---\n");
    MEX_print_exp_tran_mat( {{ dims.nu }}, {{ dims.N }}, utraj, {{ dims.nu }} );

    mexPrintf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
        mexPrintf("{{ model.name }}_acados_solve(): SUCCESS!\n");
    else
        mexPrintf("{{ model.name }}_acados_solve() failed with status %d.\n", status);

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);

    mexPrintf("\nSolver info:\n");
    mexPrintf(" SQP iterations %2d\n minimum time for 1 solve %f [ms]\n KKT %e\n",
           sqp_iter, min_time*1000, kkt_norm_inf);

    // free solver
    status = {{ model.name }}_acados_free();
    if (status)
    {
        mexPrintf("{{ model.name }}_acados_free() returned status %d.\n", status);
    }

    return;
}
