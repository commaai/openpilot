# -*- coding: future_fstrings -*-
#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
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


cdef extern from "acados/ocp_nlp/ocp_nlp_common.h":
    ctypedef struct ocp_nlp_config:
        pass

    ctypedef struct ocp_nlp_dims:
        pass

    ctypedef struct ocp_nlp_in:
        pass

    ctypedef struct ocp_nlp_out:
        pass


cdef extern from "acados_c/ocp_nlp_interface.h":
    ctypedef enum ocp_nlp_solver_t:
        pass

    ctypedef enum ocp_nlp_cost_t:
        pass

    ctypedef enum ocp_nlp_dynamics_t:
        pass

    ctypedef enum ocp_nlp_constraints_t:
        pass

    ctypedef enum ocp_nlp_reg_t:
        pass

    ctypedef struct ocp_nlp_plan:
        pass

    ctypedef struct ocp_nlp_solver:
        pass

    int ocp_nlp_cost_model_set(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_in *in_,
        int start_stage, const char *field, void *value)
    int ocp_nlp_constraints_model_set(ocp_nlp_config *config, ocp_nlp_dims *dims,
        ocp_nlp_in *in_, int stage, const char *field, void *value)

    # out
    void ocp_nlp_out_set(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_out *out,
        int stage, const char *field, void *value)
    void ocp_nlp_out_get(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_out *out,
        int stage, const char *field, void *value)
    void ocp_nlp_get_at_stage(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_solver *solver,
        int stage, const char *field, void *value)
    int ocp_nlp_dims_get_from_attr(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_out *out,
        int stage, const char *field)
    void ocp_nlp_constraint_dims_get_from_attr(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_out *out,
        int stage, const char *field, int *dims_out)
    void ocp_nlp_cost_dims_get_from_attr(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_out *out,
        int stage, const char *field, int *dims_out)
    void ocp_nlp_dynamics_dims_get_from_attr(ocp_nlp_config *config, ocp_nlp_dims *dims, ocp_nlp_out *out,
        int stage, const char *field, int *dims_out)

    # opts
    void ocp_nlp_solver_opts_set(ocp_nlp_config *config, void *opts_, const char *field, void* value)

    # solver
    void ocp_nlp_eval_residuals(ocp_nlp_solver *solver, ocp_nlp_in *nlp_in, ocp_nlp_out *nlp_out)
    void ocp_nlp_eval_cost(ocp_nlp_solver *solver, ocp_nlp_in *nlp_in_, ocp_nlp_out *nlp_out)

    # get/set
    void ocp_nlp_get(ocp_nlp_config *config, ocp_nlp_solver *solver, const char *field, void *return_value_)
    void ocp_nlp_set(ocp_nlp_config *config, ocp_nlp_solver *solver, int stage, const char *field, void *value)
