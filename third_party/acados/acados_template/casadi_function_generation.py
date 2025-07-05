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

import os
import casadi as ca
from .utils import is_empty, casadi_length


def get_casadi_symbol(x):
    if isinstance(x, ca.MX):
        return ca.MX.sym
    elif isinstance(x, ca.SX):
        return ca.SX.sym
    else:
        raise TypeError("Expected casadi SX or MX.")

################
# Dynamics
################


def generate_c_code_discrete_dynamics( model, opts ):

    casadi_codegen_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    # load model
    x = model.x
    u = model.u
    p = model.p
    phi = model.disc_dyn_expr
    model_name = model.name
    nx = casadi_length(x)

    symbol = get_casadi_symbol(x)
    # assume nx1 = nx !!!
    lam = symbol('lam', nx, 1)

    # generate jacobians
    ux = ca.vertcat(u,x)
    jac_ux = ca.jacobian(phi, ux)
    # generate adjoint
    adj_ux = ca.jtimes(phi, ux, lam, True)
    # generate hessian
    hess_ux = ca.jacobian(adj_ux, ux)

    # change directory
    cwd = os.getcwd()
    model_dir = os.path.abspath(os.path.join(opts["code_export_directory"], f'{model_name}_model'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.chdir(model_dir)

    # set up & generate ca.Functions
    fun_name = model_name + '_dyn_disc_phi_fun'
    phi_fun = ca.Function(fun_name, [x, u, p], [phi])
    phi_fun.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_dyn_disc_phi_fun_jac'
    phi_fun_jac_ut_xt = ca.Function(fun_name, [x, u, p], [phi, jac_ux.T])
    phi_fun_jac_ut_xt.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_dyn_disc_phi_fun_jac_hess'
    phi_fun_jac_ut_xt_hess = ca.Function(fun_name, [x, u, lam, p], [phi, jac_ux.T, hess_ux])
    phi_fun_jac_ut_xt_hess.generate(fun_name, casadi_codegen_opts)

    os.chdir(cwd)
    return



def generate_c_code_explicit_ode( model, opts ):

    casadi_codegen_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    generate_hess = opts["generate_hess"]

    # load model
    x = model.x
    u = model.u
    p = model.p
    f_expl = model.f_expl_expr
    model_name = model.name

    ## get model dimensions
    nx = x.size()[0]
    nu = u.size()[0]

    symbol = get_casadi_symbol(x)

    ## set up functions to be exported
    Sx = symbol('Sx', nx, nx)
    Sp = symbol('Sp', nx, nu)
    lambdaX = symbol('lambdaX', nx, 1)

    fun_name = model_name + '_expl_ode_fun'

    ## Set up functions
    expl_ode_fun = ca.Function(fun_name, [x, u, p], [f_expl])

    vdeX = ca.jtimes(f_expl,x,Sx)
    vdeP = ca.jacobian(f_expl,u) + ca.jtimes(f_expl,x,Sp)

    fun_name = model_name + '_expl_vde_forw'

    expl_vde_forw = ca.Function(fun_name, [x, Sx, Sp, u, p], [f_expl, vdeX, vdeP])

    adj = ca.jtimes(f_expl, ca.vertcat(x, u), lambdaX, True)

    fun_name = model_name + '_expl_vde_adj'
    expl_vde_adj = ca.Function(fun_name, [x, lambdaX, u, p], [adj])

    if generate_hess:
        S_forw = ca.vertcat(ca.horzcat(Sx, Sp), ca.horzcat(ca.DM.zeros(nu,nx), ca.DM.eye(nu)))
        hess = ca.mtimes(ca.transpose(S_forw),ca.jtimes(adj, ca.vertcat(x,u), S_forw))
        hess2 = []
        for j in range(nx+nu):
            for i in range(j,nx+nu):
                hess2 = ca.vertcat(hess2, hess[i,j])

        fun_name = model_name + '_expl_ode_hess'
        expl_ode_hess = ca.Function(fun_name, [x, Sx, Sp, lambdaX, u, p], [adj, hess2])

    # change directory
    cwd = os.getcwd()
    model_dir = os.path.abspath(os.path.join(opts["code_export_directory"], f'{model_name}_model'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.chdir(model_dir)

    # generate C code
    fun_name = model_name + '_expl_ode_fun'
    expl_ode_fun.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_expl_vde_forw'
    expl_vde_forw.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_expl_vde_adj'
    expl_vde_adj.generate(fun_name, casadi_codegen_opts)

    if generate_hess:
        fun_name = model_name + '_expl_ode_hess'
        expl_ode_hess.generate(fun_name, casadi_codegen_opts)
    os.chdir(cwd)

    return


def generate_c_code_implicit_ode( model, opts ):

    casadi_codegen_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    # load model
    x = model.x
    xdot = model.xdot
    u = model.u
    z = model.z
    p = model.p
    f_impl = model.f_impl_expr
    model_name = model.name

    # get model dimensions
    nx = casadi_length(x)
    nz = casadi_length(z)

    # generate jacobians
    jac_x       = ca.jacobian(f_impl, x)
    jac_xdot    = ca.jacobian(f_impl, xdot)
    jac_u       = ca.jacobian(f_impl, u)
    jac_z       = ca.jacobian(f_impl, z)

    # Set up functions
    p = model.p
    fun_name = model_name + '_impl_dae_fun'
    impl_dae_fun = ca.Function(fun_name, [x, xdot, u, z, p], [f_impl])

    fun_name = model_name + '_impl_dae_fun_jac_x_xdot_z'
    impl_dae_fun_jac_x_xdot_z = ca.Function(fun_name, [x, xdot, u, z, p], [f_impl, jac_x, jac_xdot, jac_z])

    fun_name = model_name + '_impl_dae_fun_jac_x_xdot_u_z'
    impl_dae_fun_jac_x_xdot_u_z = ca.Function(fun_name, [x, xdot, u, z, p], [f_impl, jac_x, jac_xdot, jac_u, jac_z])

    fun_name = model_name + '_impl_dae_fun_jac_x_xdot_u'
    impl_dae_fun_jac_x_xdot_u = ca.Function(fun_name, [x, xdot, u, z, p], [f_impl, jac_x, jac_xdot, jac_u])

    fun_name = model_name + '_impl_dae_jac_x_xdot_u_z'
    impl_dae_jac_x_xdot_u_z = ca.Function(fun_name, [x, xdot, u, z, p], [jac_x, jac_xdot, jac_u, jac_z])

    if opts["generate_hess"]:
        x_xdot_z_u = ca.vertcat(x, xdot, z, u)
        symbol = get_casadi_symbol(x)
        multiplier = symbol('multiplier', nx + nz)
        ADJ = ca.jtimes(f_impl, x_xdot_z_u, multiplier, True)
        HESS = ca.jacobian(ADJ, x_xdot_z_u)
        fun_name = model_name + '_impl_dae_hess'
        impl_dae_hess = ca.Function(fun_name, [x, xdot, u, z, multiplier, p], [HESS])

    # change directory
    cwd = os.getcwd()
    model_dir = os.path.abspath(os.path.join(opts["code_export_directory"], f'{model_name}_model'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.chdir(model_dir)

    # generate C code
    fun_name = model_name + '_impl_dae_fun'
    impl_dae_fun.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_impl_dae_fun_jac_x_xdot_z'
    impl_dae_fun_jac_x_xdot_z.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_impl_dae_jac_x_xdot_u_z'
    impl_dae_jac_x_xdot_u_z.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_impl_dae_fun_jac_x_xdot_u_z'
    impl_dae_fun_jac_x_xdot_u_z.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_impl_dae_fun_jac_x_xdot_u'
    impl_dae_fun_jac_x_xdot_u.generate(fun_name, casadi_codegen_opts)

    if opts["generate_hess"]:
        fun_name = model_name + '_impl_dae_hess'
        impl_dae_hess.generate(fun_name, casadi_codegen_opts)

    os.chdir(cwd)
    return


def generate_c_code_gnsf( model, opts ):

    casadi_codegen_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    model_name = model.name

    # set up directory
    cwd = os.getcwd()
    model_dir = os.path.abspath(os.path.join(opts["code_export_directory"], f'{model_name}_model'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.chdir(model_dir)

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
    # if the model uses ca.MX because of cost/constraints
    # the DAE can be exported as ca.SX -> detect GNSF in Matlab
    # -> evaluated ca.SX GNSF functions with ca.MX.
    u = model.u
    symbol = get_casadi_symbol(u)

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
    phi_fun_ = ca.Function(fun_name, [y, uhat, p], [phi_fun(y, uhat, p)])
    phi_fun_.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_gnsf_phi_fun_jac_y'
    phi_fun_jac_y = model.phi_fun_jac_y
    phi_fun_jac_y_ = ca.Function(fun_name, [y, uhat, p], phi_fun_jac_y(y, uhat, p))
    phi_fun_jac_y_.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_gnsf_phi_jac_y_uhat'
    phi_jac_y_uhat = model.phi_jac_y_uhat
    phi_jac_y_uhat_ = ca.Function(fun_name, [y, uhat, p], phi_jac_y_uhat(y, uhat, p))
    phi_jac_y_uhat_.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_gnsf_f_lo_fun_jac_x1k1uz'
    f_lo_fun_jac_x1k1uz = model.f_lo_fun_jac_x1k1uz
    f_lo_fun_jac_x1k1uz_eval = f_lo_fun_jac_x1k1uz(x1, x1dot, z1, u, p)

    # avoid codegeneration issue
    if not isinstance(f_lo_fun_jac_x1k1uz_eval, tuple) and is_empty(f_lo_fun_jac_x1k1uz_eval):
        f_lo_fun_jac_x1k1uz_eval = [empty_var]

    f_lo_fun_jac_x1k1uz_ = ca.Function(fun_name, [x1, x1dot, z1, u, p],
                 f_lo_fun_jac_x1k1uz_eval)
    f_lo_fun_jac_x1k1uz_.generate(fun_name, casadi_codegen_opts)

    fun_name = model_name + '_gnsf_get_matrices_fun'
    get_matrices_fun_ = ca.Function(fun_name, [dummy], get_matrices_fun(1))
    get_matrices_fun_.generate(fun_name, casadi_codegen_opts)

    # remove fields for json dump
    del model.phi_fun
    del model.phi_fun_jac_y
    del model.phi_jac_y_uhat
    del model.f_lo_fun_jac_x1k1uz
    del model.get_matrices_fun

    os.chdir(cwd)

    return


################
# Cost
################

def generate_c_code_external_cost(model, stage_type, opts):

    casadi_codegen_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    x = model.x
    p = model.p
    u = model.u
    z = model.z
    symbol = get_casadi_symbol(x)

    if stage_type == 'terminal':
        suffix_name = "_cost_ext_cost_e_fun"
        suffix_name_hess = "_cost_ext_cost_e_fun_jac_hess"
        suffix_name_jac = "_cost_ext_cost_e_fun_jac"
        ext_cost = model.cost_expr_ext_cost_e
        custom_hess = model.cost_expr_ext_cost_custom_hess_e
        # Last stage cannot depend on u and z
        u = symbol("u", 0, 0)
        z = symbol("z", 0, 0)

    elif stage_type == 'path':
        suffix_name = "_cost_ext_cost_fun"
        suffix_name_hess = "_cost_ext_cost_fun_jac_hess"
        suffix_name_jac = "_cost_ext_cost_fun_jac"
        ext_cost = model.cost_expr_ext_cost
        custom_hess = model.cost_expr_ext_cost_custom_hess

    elif stage_type == 'initial':
        suffix_name = "_cost_ext_cost_0_fun"
        suffix_name_hess = "_cost_ext_cost_0_fun_jac_hess"
        suffix_name_jac = "_cost_ext_cost_0_fun_jac"
        ext_cost = model.cost_expr_ext_cost_0
        custom_hess = model.cost_expr_ext_cost_custom_hess_0

    nunx = x.shape[0] + u.shape[0]

    # set up functions to be exported
    fun_name = model.name + suffix_name
    fun_name_hess = model.name + suffix_name_hess
    fun_name_jac = model.name + suffix_name_jac

    # generate expression for full gradient and Hessian
    hess_uxz, grad_uxz = ca.hessian(ext_cost, ca.vertcat(u, x, z))

    hess_ux = hess_uxz[:nunx, :nunx]
    hess_z = hess_uxz[nunx:, nunx:]
    hess_z_ux = hess_uxz[nunx:, :nunx]

    if custom_hess is not None:
        hess_ux = custom_hess

    ext_cost_fun = ca.Function(fun_name, [x, u, z, p], [ext_cost])

    ext_cost_fun_jac_hess = ca.Function(
        fun_name_hess, [x, u, z, p], [ext_cost, grad_uxz, hess_ux, hess_z, hess_z_ux]
    )
    ext_cost_fun_jac = ca.Function(
        fun_name_jac, [x, u, z, p], [ext_cost, grad_uxz]
    )

    # change directory
    cwd = os.getcwd()
    cost_dir = os.path.abspath(os.path.join(opts["code_export_directory"], f'{model.name}_cost'))
    if not os.path.exists(cost_dir):
        os.makedirs(cost_dir)
    os.chdir(cost_dir)

    ext_cost_fun.generate(fun_name, casadi_codegen_opts)
    ext_cost_fun_jac_hess.generate(fun_name_hess, casadi_codegen_opts)
    ext_cost_fun_jac.generate(fun_name_jac, casadi_codegen_opts)

    os.chdir(cwd)
    return


def generate_c_code_nls_cost( model, cost_name, stage_type, opts ):

    casadi_codegen_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    x = model.x
    z = model.z
    p = model.p
    u = model.u

    symbol = get_casadi_symbol(x)

    if stage_type == 'terminal':
        middle_name = '_cost_y_e'
        u = symbol('u', 0, 0)
        y_expr = model.cost_y_expr_e

    elif stage_type == 'initial':
        middle_name = '_cost_y_0'
        y_expr = model.cost_y_expr_0

    elif stage_type == 'path':
        middle_name = '_cost_y'
        y_expr = model.cost_y_expr

    # change directory
    cwd = os.getcwd()
    cost_dir = os.path.abspath(os.path.join(opts["code_export_directory"], f'{model.name}_cost'))
    if not os.path.exists(cost_dir):
        os.makedirs(cost_dir)
    os.chdir(cost_dir)

    # set up expressions
    cost_jac_expr = ca.transpose(ca.jacobian(y_expr, ca.vertcat(u, x)))
    dy_dz = ca.jacobian(y_expr, z)
    ny = casadi_length(y_expr)

    y = symbol('y', ny, 1)

    y_adj = ca.jtimes(y_expr, ca.vertcat(u, x), y, True)
    y_hess = ca.jacobian(y_adj, ca.vertcat(u, x))

    ## generate C code
    suffix_name = '_fun'
    fun_name = cost_name + middle_name + suffix_name
    y_fun = ca.Function( fun_name, [x, u, z, p], [ y_expr ])
    y_fun.generate( fun_name, casadi_codegen_opts )

    suffix_name = '_fun_jac_ut_xt'
    fun_name = cost_name + middle_name + suffix_name
    y_fun_jac_ut_xt = ca.Function(fun_name, [x, u, z, p], [ y_expr, cost_jac_expr, dy_dz ])
    y_fun_jac_ut_xt.generate( fun_name, casadi_codegen_opts )

    suffix_name = '_hess'
    fun_name = cost_name + middle_name + suffix_name
    y_hess = ca.Function(fun_name, [x, u, z, y, p], [ y_hess ])
    y_hess.generate( fun_name, casadi_codegen_opts )

    os.chdir(cwd)

    return



def generate_c_code_conl_cost(model, cost_name, stage_type, opts):

    casadi_codegen_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    x = model.x
    z = model.z
    p = model.p

    symbol = get_casadi_symbol(x)

    if stage_type == 'terminal':
        u = symbol('u', 0, 0)

        yref = model.cost_r_in_psi_expr_e
        inner_expr = model.cost_y_expr_e - yref
        outer_expr = model.cost_psi_expr_e
        res_expr = model.cost_r_in_psi_expr_e

        suffix_name_fun = '_conl_cost_e_fun'
        suffix_name_fun_jac_hess = '_conl_cost_e_fun_jac_hess'

        custom_hess = model.cost_conl_custom_outer_hess_e

    elif stage_type == 'initial':
        u = model.u

        yref = model.cost_r_in_psi_expr_0
        inner_expr = model.cost_y_expr_0 - yref
        outer_expr = model.cost_psi_expr_0
        res_expr = model.cost_r_in_psi_expr_0

        suffix_name_fun = '_conl_cost_0_fun'
        suffix_name_fun_jac_hess = '_conl_cost_0_fun_jac_hess'

        custom_hess = model.cost_conl_custom_outer_hess_0

    elif stage_type == 'path':
        u = model.u

        yref = model.cost_r_in_psi_expr
        inner_expr = model.cost_y_expr - yref
        outer_expr = model.cost_psi_expr
        res_expr = model.cost_r_in_psi_expr

        suffix_name_fun = '_conl_cost_fun'
        suffix_name_fun_jac_hess = '_conl_cost_fun_jac_hess'

        custom_hess = model.cost_conl_custom_outer_hess

    # set up function names
    fun_name_cost_fun = model.name + suffix_name_fun
    fun_name_cost_fun_jac_hess = model.name + suffix_name_fun_jac_hess

    # set up functions to be exported
    outer_loss_fun = ca.Function('psi', [res_expr, p], [outer_expr])
    cost_expr = outer_loss_fun(inner_expr, p)

    outer_loss_grad_fun = ca.Function('outer_loss_grad', [res_expr, p], [ca.jacobian(outer_expr, res_expr).T])

    if custom_hess is None:
        outer_hess_fun = ca.Function('inner_hess', [res_expr, p], [ca.hessian(outer_loss_fun(res_expr, p), res_expr)[0]])
    else:
        outer_hess_fun = ca.Function('inner_hess', [res_expr, p], [custom_hess])

    Jt_ux_expr = ca.jacobian(inner_expr, ca.vertcat(u, x)).T
    Jt_z_expr = ca.jacobian(inner_expr, z).T

    cost_fun = ca.Function(
        fun_name_cost_fun,
        [x, u, z, yref, p],
        [cost_expr])

    cost_fun_jac_hess = ca.Function(
        fun_name_cost_fun_jac_hess,
        [x, u, z, yref, p],
        [cost_expr, outer_loss_grad_fun(inner_expr, p), Jt_ux_expr, Jt_z_expr, outer_hess_fun(inner_expr, p)]
    )
    # change directory
    cwd = os.getcwd()
    cost_dir = os.path.abspath(os.path.join(opts["code_export_directory"], f'{model.name}_cost'))
    if not os.path.exists(cost_dir):
        os.makedirs(cost_dir)
    os.chdir(cost_dir)

    # generate C code
    cost_fun.generate(fun_name_cost_fun, casadi_codegen_opts)
    cost_fun_jac_hess.generate(fun_name_cost_fun_jac_hess, casadi_codegen_opts)

    os.chdir(cwd)

    return


################
# Constraints
################
def generate_c_code_constraint( model, con_name, is_terminal, opts ):

    casadi_codegen_opts = dict(mex=False, casadi_int='int', casadi_real='double')

    # load constraint variables and expression
    x = model.x
    p = model.p

    symbol = get_casadi_symbol(x)

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

    if (is_empty(con_h_expr) and is_empty(con_phi_expr)):
        # both empty -> nothing to generate
        return

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
    cwd = os.getcwd()
    constraints_dir = os.path.abspath(os.path.join(opts["code_export_directory"], f'{model.name}_constraints'))
    if not os.path.exists(constraints_dir):
        os.makedirs(constraints_dir)
    os.chdir(constraints_dir)

    # export casadi functions
    if constr_type == 'BGH':
        if is_terminal:
            fun_name = con_name + '_constr_h_e_fun_jac_uxt_zt'
        else:
            fun_name = con_name + '_constr_h_fun_jac_uxt_zt'

        jac_ux_t = ca.transpose(ca.jacobian(con_h_expr, ca.vertcat(u,x)))
        jac_z_t = ca.jacobian(con_h_expr, z)
        constraint_fun_jac_tran = ca.Function(fun_name, [x, u, z, p], \
                [con_h_expr, jac_ux_t, jac_z_t])

        constraint_fun_jac_tran.generate(fun_name, casadi_codegen_opts)
        if opts['generate_hess']:

            if is_terminal:
                fun_name = con_name + '_constr_h_e_fun_jac_uxt_zt_hess'
            else:
                fun_name = con_name + '_constr_h_fun_jac_uxt_zt_hess'

            # adjoint
            adj_ux = ca.jtimes(con_h_expr, ca.vertcat(u, x), lam_h, True)
            # hessian
            hess_ux = ca.jacobian(adj_ux, ca.vertcat(u, x))

            adj_z = ca.jtimes(con_h_expr, z, lam_h, True)
            hess_z = ca.jacobian(adj_z, z)

            # set up functions
            constraint_fun_jac_tran_hess = \
                ca.Function(fun_name, [x, u, lam_h, z, p], \
                    [con_h_expr, jac_ux_t, hess_ux, jac_z_t, hess_z])

            # generate C code
            constraint_fun_jac_tran_hess.generate(fun_name, casadi_codegen_opts)

        if is_terminal:
            fun_name = con_name + '_constr_h_e_fun'
        else:
            fun_name = con_name + '_constr_h_fun'
        h_fun = ca.Function(fun_name, [x, u, z, p], [con_h_expr])
        h_fun.generate(fun_name, casadi_codegen_opts)

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
        con_phi_expr_x_u_z = ca.substitute(con_phi_expr, r, con_r_expr)
        phi_jac_u = ca.jacobian(con_phi_expr_x_u_z, u)
        phi_jac_x = ca.jacobian(con_phi_expr_x_u_z, x)
        phi_jac_z = ca.jacobian(con_phi_expr_x_u_z, z)

        hess = ca.hessian(con_phi_expr[0], r)[0]
        for i in range(1, nphi):
            hess = ca.vertcat(hess, ca.hessian(con_phi_expr[i], r)[0])

        r_jac_u = ca.jacobian(con_r_expr, u)
        r_jac_x = ca.jacobian(con_r_expr, x)

        constraint_phi = \
            ca.Function(fun_name, [x, u, z, p], \
                [con_phi_expr_x_u_z, \
                ca.vertcat(ca.transpose(phi_jac_u), ca.transpose(phi_jac_x)), \
                ca.transpose(phi_jac_z), \
                hess,
                ca.vertcat(ca.transpose(r_jac_u), ca.transpose(r_jac_x))])

        constraint_phi.generate(fun_name, casadi_codegen_opts)

    # change directory back
    os.chdir(cwd)

    return

