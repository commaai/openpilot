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
#include "acados_c/external_function_interface.h"
#include "acados_solver_lat.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

#define NX     LAT_NX
#define NZ     LAT_NZ
#define NU     LAT_NU
#define NP     LAT_NP
#define NBX    LAT_NBX
#define NBX0   LAT_NBX0
#define NBU    LAT_NBU
#define NSBX   LAT_NSBX
#define NSBU   LAT_NSBU
#define NSH    LAT_NSH
#define NSG    LAT_NSG
#define NSPHI  LAT_NSPHI
#define NSHN   LAT_NSHN
#define NSGN   LAT_NSGN
#define NSPHIN LAT_NSPHIN
#define NSBXN  LAT_NSBXN
#define NS     LAT_NS
#define NSN    LAT_NSN
#define NG     LAT_NG
#define NBXN   LAT_NBXN
#define NGN    LAT_NGN
#define NY0    LAT_NY0
#define NY     LAT_NY
#define NYN    LAT_NYN
#define NH     LAT_NH
#define NPHI   LAT_NPHI
#define NHN    LAT_NHN
#define NPHIN  LAT_NPHIN
#define NR     LAT_NR


int main()
{

    lat_solver_capsule *acados_ocp_capsule = lat_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = LAT_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = lat_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("lat_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = lat_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = lat_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = lat_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = lat_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = lat_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = lat_acados_get_nlp_opts(acados_ocp_capsule);

    // initial condition
    int idxbx0[NBX0];
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;

    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = 0.0;
    ubx0[0] = 0.0;
    lbx0[1] = 0.0;
    ubx0[1] = 0.0;
    lbx0[2] = 0.0;
    ubx0[2] = 0.0;
    lbx0[3] = 0.0;
    ubx0[3] = 0.0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;
    x_init[3] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    // set parameters
    double p[NP];
    p[0] = 0.0;
    p[1] = 0.0;

    for (int ii = 0; ii <= N; ii++)
    {
        lat_acados_update_params(acados_ocp_capsule, ii, p, NP);
    }
  

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];


    // solve ocp in loop
    int rti_phase = 0;

    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
        status = lat_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat( NX, N+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat( NU, N, utraj, NU );
    // ocp_nlp_out_print(nlp_solver->dims, nlp_out);

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("lat_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("lat_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);

    lat_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);

    // free solver
    status = lat_acados_free(acados_ocp_capsule);
    if (status) {
        printf("lat_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = lat_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("lat_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
