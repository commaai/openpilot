# -*- coding: future_fstrings -*-
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


cdef extern from "acados/sim/sim_common.h":
    ctypedef struct sim_config:
        pass

    ctypedef struct sim_opts:
        pass

    ctypedef struct sim_in:
        pass

    ctypedef struct sim_out:
        pass


cdef extern from "acados_c/sim_interface.h":

    ctypedef struct sim_plan:
        pass

    ctypedef struct sim_solver:
        pass

    # out
    void sim_out_get(sim_config *config, void *dims, sim_out *out, const char *field, void *value)
    int sim_dims_get_from_attr(sim_config *config, void *dims, const char *field, void *dims_data)

    # opts
    void sim_opts_set(sim_config *config, void *opts_, const char *field, void *value)

    # get/set
    void sim_in_set(sim_config *config, void *dims, sim_in *sim_in, const char *field, void *value)
    void sim_solver_set(sim_solver *solver, const char *field, void *value)