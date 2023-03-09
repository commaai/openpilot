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
from .utils import ALLOWED_CASADI_VERSIONS, is_empty, casadi_length, casadi_version_warning

def generate_c_code_constraint( model, con_name, is_terminal, opts ):

    casadi_version = CasadiMeta.version()
    casadi_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    if casadi_version not in (ALLOWED_CASADI_VERSIONS):
        casadi_version_warning(casadi_version)

    # load constraint variables and expression
    x = model.x
    p = model.p

    if isinstance(x, casadi.MX):
        symbol = MX.sym
    else:
        symbol = SX.sym

    if is_terminal:
        con_h_expr = model.con_h_expr_e
        con_phi_expr = model.con_phi_expr_e
        # create dummy u, z
        u = symbol('u', 0, 0)
        z = symbol('z', 0, 0)
    else:
        con_h_expr = model.con_h_expr
        con_phi_expr = model.con_phi_expr
        u = model.u
        z = model.z

    if (not is_empty(con_h_expr)) and (not is_empty(con_phi_expr)):
        raise Exception("acados: you can either have constraint_h, or constraint_phi, not both.")

    if not (is_empty(con_h_expr) and is_empty(con_phi_expr)):
        if is_empty(con_h_expr):
            constr_type = 'BGP'
        else:
            constr_type = 'BGH'

        if is_empty(p):
            p = symbol('p', 0, 0)

        if is_empty(z):
            z = symbol('z', 0, 0)

        if not (is_empty(con_h_expr)) and opts['generate_hess']:
            # multipliers for hessian
            nh = casadi_length(con_h_expr)
            lam_h = symbol('lam_h', nh, 1)

        # set up & change directory
        code_export_dir = opts["code_export_directory"]
        if not os.path.exists(code_export_dir):
            os.makedirs(code_export_dir)

        cwd = os.getcwd()
        os.chdir(code_export_dir)
        gen_dir = con_name + '_constraints'
        if not os.path.exists(gen_dir):
            os.mkdir(gen_dir)
        gen_dir_location = os.path.join('.', gen_dir)
        os.chdir(gen_dir_location)

        # export casadi functions
        if constr_type == 'BGH':
            if is_terminal:
                fun_name = con_name + '_constr_h_e_fun_jac_uxt_zt'
            else:
                fun_name = con_name + '_constr_h_fun_jac_uxt_zt'

            jac_ux_t = transpose(jacobian(con_h_expr, vertcat(u,x)))
            jac_z_t = jacobian(con_h_expr, z)
            constraint_fun_jac_tran = Function(fun_name, [x, u, z, p], \
                    [con_h_expr, jac_ux_t, jac_z_t])

            constraint_fun_jac_tran.generate(fun_name, casadi_opts)
            if opts['generate_hess']:

                if is_terminal:
                    fun_name = con_name + '_constr_h_e_fun_jac_uxt_zt_hess'
                else:
                    fun_name = con_name + '_constr_h_fun_jac_uxt_zt_hess'

                # adjoint
                adj_ux = jtimes(con_h_expr, vertcat(u, x), lam_h, True)
                # hessian
                hess_ux = jacobian(adj_ux, vertcat(u, x))

                adj_z = jtimes(con_h_expr, z, lam_h, True)
                hess_z = jacobian(adj_z, z)

                # set up functions
                constraint_fun_jac_tran_hess = \
                    Function(fun_name, [x, u, lam_h, z, p], \
                      [con_h_expr, jac_ux_t, hess_ux, jac_z_t, hess_z])

                # generate C code
                constraint_fun_jac_tran_hess.generate(fun_name, casadi_opts)

            if is_terminal:
                fun_name = con_name + '_constr_h_e_fun'
            else:
                fun_name = con_name + '_constr_h_fun'
            h_fun = Function(fun_name, [x, u, z, p], [con_h_expr])
            h_fun.generate(fun_name, casadi_opts)

        else: # BGP constraint
            if is_terminal:
                fun_name = con_name + '_phi_e_constraint'
                r = model.con_r_in_phi_e
                con_r_expr = model.con_r_expr_e
            else:
                fun_name = con_name + '_phi_constraint'
                r = model.con_r_in_phi
                con_r_expr = model.con_r_expr

            nphi = casadi_length(con_phi_expr)
            con_phi_expr_x_u_z = substitute(con_phi_expr, r, con_r_expr)
            phi_jac_u = jacobian(con_phi_expr_x_u_z, u)
            phi_jac_x = jacobian(con_phi_expr_x_u_z, x)
            phi_jac_z = jacobian(con_phi_expr_x_u_z, z)

            hess = hessian(con_phi_expr[0], r)[0]
            for i in range(1, nphi):
                hess = vertcat(hess, hessian(con_phi_expr[i], r)[0])

            r_jac_u = jacobian(con_r_expr, u)
            r_jac_x = jacobian(con_r_expr, x)

            constraint_phi = \
                Function(fun_name, [x, u, z, p], \
                    [con_phi_expr_x_u_z, \
                    vertcat(transpose(phi_jac_u), \
                    transpose(phi_jac_x)), \
                    transpose(phi_jac_z), \
                    hess, vertcat(transpose(r_jac_u), \
                    transpose(r_jac_x))])

            constraint_phi.generate(fun_name, casadi_opts)

        # change directory back
        os.chdir(cwd)

    return
