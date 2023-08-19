#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

cimport acados_solver_common

cdef extern from "acados_solver_{{ model.name }}.h":
    ctypedef struct nlp_solver_capsule "{{ model.name }}_solver_capsule":
        pass

    nlp_solver_capsule * acados_create_capsule "{{ model.name }}_acados_create_capsule"()
    int acados_free_capsule "{{ model.name }}_acados_free_capsule"(nlp_solver_capsule *capsule)

    int acados_create "{{ model.name }}_acados_create"(nlp_solver_capsule * capsule)

    int acados_create_with_discretization "{{ model.name }}_acados_create_with_discretization"(nlp_solver_capsule * capsule, int n_time_steps, double* new_time_steps)
    int acados_update_time_steps "{{ model.name }}_acados_update_time_steps"(nlp_solver_capsule * capsule, int N, double* new_time_steps)
    int acados_update_qp_solver_cond_N "{{ model.name }}_acados_update_qp_solver_cond_N"(nlp_solver_capsule * capsule, int qp_solver_cond_N)

    int acados_update_params "{{ model.name }}_acados_update_params"(nlp_solver_capsule * capsule, int stage, double *value, int np_)
    int acados_update_params_sparse "{{ model.name }}_acados_update_params_sparse"(nlp_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
    int acados_solve "{{ model.name }}_acados_solve"(nlp_solver_capsule * capsule)
    int acados_reset "{{ model.name }}_acados_reset"(nlp_solver_capsule * capsule, int reset_qp_solver_mem)
    int acados_free "{{ model.name }}_acados_free"(nlp_solver_capsule * capsule)
    void acados_print_stats "{{ model.name }}_acados_print_stats"(nlp_solver_capsule * capsule)

    int acados_custom_update "{{ model.name }}_acados_custom_update"(nlp_solver_capsule* capsule, double * data, int data_len)

    acados_solver_common.ocp_nlp_in *acados_get_nlp_in "{{ model.name }}_acados_get_nlp_in"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_out *acados_get_nlp_out "{{ model.name }}_acados_get_nlp_out"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_out *acados_get_sens_out "{{ model.name }}_acados_get_sens_out"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_solver *acados_get_nlp_solver "{{ model.name }}_acados_get_nlp_solver"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_config *acados_get_nlp_config "{{ model.name }}_acados_get_nlp_config"(nlp_solver_capsule * capsule)
    void *acados_get_nlp_opts "{{ model.name }}_acados_get_nlp_opts"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_dims *acados_get_nlp_dims "{{ model.name }}_acados_get_nlp_dims"(nlp_solver_capsule * capsule)
    acados_solver_common.ocp_nlp_plan *acados_get_nlp_plan "{{ model.name }}_acados_get_nlp_plan"(nlp_solver_capsule * capsule)
