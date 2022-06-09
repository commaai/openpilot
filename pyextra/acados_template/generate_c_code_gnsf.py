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

import os
from casadi import *
from .utils import ALLOWED_CASADI_VERSIONS, is_empty, casadi_version_warning

def generate_c_code_gnsf( model, opts ):

    casadi_version = CasadiMeta.version()
    casadi_opts = dict(mex=False, casadi_int='int', casadi_real='double')
    if casadi_version not in (ALLOWED_CASADI_VERSIONS):
        casadi_version_warning(casadi_version)

    model_name = model.name
    code_export_dir = opts["code_export_directory"]

    # set up directory
    if not os.path.exists(code_export_dir):
        os.makedirs(code_export_dir)

    cwd = os.getcwd()
    os.chdir(code_export_dir)
    model_dir = model_name + '_model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir_location = os.path.join('.', model_dir)
    os.chdir(model_dir_location)

    # obtain gnsf dimensions
    get_matrices_fun = model.get_matrices_fun
    phi_fun = model.phi_fun

    size_gnsf_A = get_matrices_fun.size_out(0)
    gnsf_nx1 = size_gnsf_A[1]
    gnsf_nz1 = size_gnsf_A[0] - size_gnsf_A[1]
    gnsf_nuhat = max(phi_fun.size_in(1))
    gnsf_ny = max(phi_fun.size_in(0))
    gnsf_nout = max(phi_fun.size_out(0))

    # set up expressions
    # if the model uses MX because of cost/constraints
    # the DAE can be exported as SX -> detect GNSF in Matlab
    # -> evaluated SX GNSF functions with MX.
    u = model.u

    if isinstance(u, casadi.MX):
        symbol = MX.sym
    else:
        symbol = SX.sym

    y = symbol("y", gnsf_ny, 1)
    uhat = symbol("uhat", gnsf_nuhat, 1)
    p = model.p
    x1 = symbol("gnsf_x1", gnsf_nx1, 1)
    x1dot = symbol("gnsf_x1dot", gnsf_nx1, 1)
    z1 = symbol("gnsf_z1", gnsf_nz1, 1)
    dummy = symbol("gnsf_dummy", 1, 1)
    empty_var = symbol("gnsf_empty_var", 0, 0)

    ## generate C code
    fun_name = model_name + '_gnsf_phi_fun'
    phi_fun_ = Function(fun_name, [y, uhat, p], [phi_fun(y, uhat, p)])
    phi_fun_.generate(fun_name, casadi_opts)

    fun_name = model_name + '_gnsf_phi_fun_jac_y'
    phi_fun_jac_y = model.phi_fun_jac_y
    phi_fun_jac_y_ = Function(fun_name, [y, uhat, p], phi_fun_jac_y(y, uhat, p))
    phi_fun_jac_y_.generate(fun_name, casadi_opts)

    fun_name = model_name + '_gnsf_phi_jac_y_uhat'
    phi_jac_y_uhat = model.phi_jac_y_uhat
    phi_jac_y_uhat_ = Function(fun_name, [y, uhat, p], phi_jac_y_uhat(y, uhat, p))
    phi_jac_y_uhat_.generate(fun_name, casadi_opts)

    fun_name = model_name + '_gnsf_f_lo_fun_jac_x1k1uz'
    f_lo_fun_jac_x1k1uz = model.f_lo_fun_jac_x1k1uz
    f_lo_fun_jac_x1k1uz_eval = f_lo_fun_jac_x1k1uz(x1, x1dot, z1, u, p)

    # avoid codegeneration issue
    if not isinstance(f_lo_fun_jac_x1k1uz_eval, tuple) and is_empty(f_lo_fun_jac_x1k1uz_eval):
        f_lo_fun_jac_x1k1uz_eval = [empty_var]

    f_lo_fun_jac_x1k1uz_ = Function(fun_name, [x1, x1dot, z1, u, p],
                 f_lo_fun_jac_x1k1uz_eval)
    f_lo_fun_jac_x1k1uz_.generate(fun_name, casadi_opts)

    fun_name = model_name + '_gnsf_get_matrices_fun'
    get_matrices_fun_ = Function(fun_name, [dummy], get_matrices_fun(1))
    get_matrices_fun_.generate(fun_name, casadi_opts)

    # remove fields for json dump
    del model.phi_fun
    del model.phi_fun_jac_y
    del model.phi_jac_y_uhat
    del model.f_lo_fun_jac_x1k1uz
    del model.get_matrices_fun

    os.chdir(cwd)

    return
