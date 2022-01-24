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

def generate_c_code_explicit_ode( model, opts ):

    casadi_version = CasadiMeta.version()
    casadi_opts = dict(mex=False, casadi_int='int', casadi_real='double')
    if casadi_version not in (ALLOWED_CASADI_VERSIONS):
        casadi_version_warning(casadi_version)


    generate_hess = opts["generate_hess"]
    code_export_dir = opts["code_export_directory"]

    # load model
    x = model.x
    u = model.u
    p = model.p
    f_expl = model.f_expl_expr
    model_name = model.name

    ## get model dimensions
    nx = x.size()[0]
    nu = u.size()[0]

    if isinstance(f_expl, casadi.MX):
        symbol = MX.sym
    elif isinstance(f_expl, casadi.SX):
        symbol = SX.sym
    else:
        raise Exception("Invalid type for f_expl! Possible types are 'SX' and 'MX'. Exiting.")
    ## set up functions to be exported
    Sx = symbol('Sx', nx, nx)
    Sp = symbol('Sp', nx, nu)
    lambdaX = symbol('lambdaX', nx, 1)

    fun_name = model_name + '_expl_ode_fun'

    ## Set up functions
    expl_ode_fun = Function(fun_name, [x, u, p], [f_expl])

    vdeX = jtimes(f_expl,x,Sx)
    vdeP = jacobian(f_expl,u) + jtimes(f_expl,x,Sp)

    fun_name = model_name + '_expl_vde_forw'

    expl_vde_forw = Function(fun_name, [x, Sx, Sp, u, p], [f_expl, vdeX, vdeP])

    adj = jtimes(f_expl, vertcat(x, u), lambdaX, True)

    fun_name = model_name + '_expl_vde_adj'
    expl_vde_adj = Function(fun_name, [x, lambdaX, u, p], [adj])

    if generate_hess:
        S_forw = vertcat(horzcat(Sx, Sp), horzcat(DM.zeros(nu,nx), DM.eye(nu)))
        hess = mtimes(transpose(S_forw),jtimes(adj, vertcat(x,u), S_forw))
        hess2 = []
        for j in range(nx+nu):
            for i in range(j,nx+nu):
                hess2 = vertcat(hess2, hess[i,j])

        fun_name = model_name + '_expl_ode_hess'
        expl_ode_hess = Function(fun_name, [x, Sx, Sp, lambdaX, u, p], [adj, hess2])

    ## generate C code
    if not os.path.exists(code_export_dir):
        os.makedirs(code_export_dir)

    cwd = os.getcwd()
    os.chdir(code_export_dir)
    model_dir = model_name + '_model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir_location = os.path.join('.', model_dir)
    os.chdir(model_dir_location)
    fun_name = model_name + '_expl_ode_fun'
    expl_ode_fun.generate(fun_name, casadi_opts)

    fun_name = model_name + '_expl_vde_forw'
    expl_vde_forw.generate(fun_name, casadi_opts)

    fun_name = model_name + '_expl_vde_adj'
    expl_vde_adj.generate(fun_name, casadi_opts)

    if generate_hess:
        fun_name = model_name + '_expl_ode_hess'
        expl_ode_hess.generate(fun_name, casadi_opts)
    os.chdir(cwd)

    return
