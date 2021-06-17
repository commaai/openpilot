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
from casadi import SX, MX, Function, transpose, vertcat, horzcat, hessian, CasadiMeta
from .utils import ALLOWED_CASADI_VERSIONS, casadi_version_warning


def generate_c_code_external_cost(model, stage_type, opts):

    casadi_version = CasadiMeta.version()
    casadi_opts = dict(mex=False, casadi_int="int", casadi_real="double")

    if casadi_version not in (ALLOWED_CASADI_VERSIONS):
        casadi_version_warning(casadi_version)

    x = model.x
    p = model.p

    if isinstance(x, MX):
        symbol = MX.sym
    else:
        symbol = SX.sym

    if stage_type == 'terminal':
        suffix_name = "_cost_ext_cost_e_fun"
        suffix_name_hess = "_cost_ext_cost_e_fun_jac_hess"
        suffix_name_jac = "_cost_ext_cost_e_fun_jac"
        u = symbol("u", 0, 0)
        ext_cost = model.cost_expr_ext_cost_e

    elif stage_type == 'path':
        suffix_name = "_cost_ext_cost_fun"
        suffix_name_hess = "_cost_ext_cost_fun_jac_hess"
        suffix_name_jac = "_cost_ext_cost_fun_jac"
        u = model.u
        ext_cost = model.cost_expr_ext_cost

    elif stage_type == 'initial':
        suffix_name = "_cost_ext_cost_0_fun"
        suffix_name_hess = "_cost_ext_cost_0_fun_jac_hess"
        suffix_name_jac = "_cost_ext_cost_0_fun_jac"
        u = model.u
        ext_cost = model.cost_expr_ext_cost_0

    # set up functions to be exported
    fun_name = model.name + suffix_name
    fun_name_hess = model.name + suffix_name_hess
    fun_name_jac = model.name + suffix_name_jac

    # generate expression for full gradient and Hessian
    full_hess, grad = hessian(ext_cost, vertcat(u, x))

    ext_cost_fun = Function(fun_name, [x, u, p], [ext_cost])
    ext_cost_fun_jac_hess = Function(
        fun_name_hess, [x, u, p], [ext_cost, grad, full_hess]
    )
    ext_cost_fun_jac = Function(
        fun_name_jac, [x, u, p], [ext_cost, grad]
    )

    # generate C code
    code_export_dir = opts["code_export_directory"]
    if not os.path.exists(code_export_dir):
        os.makedirs(code_export_dir)

    cwd = os.getcwd()
    os.chdir(code_export_dir)
    gen_dir = model.name + '_cost'
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)
    gen_dir_location = "./" + gen_dir
    os.chdir(gen_dir_location)

    ext_cost_fun.generate(fun_name, casadi_opts)
    ext_cost_fun_jac_hess.generate(fun_name_hess, casadi_opts)
    ext_cost_fun_jac.generate(fun_name_jac, casadi_opts)

    os.chdir(cwd)
    return
