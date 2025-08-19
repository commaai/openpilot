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

import sys
import os
import json
import numpy as np
from datetime import datetime
import importlib
import shutil

from subprocess import DEVNULL, call, STDOUT

from ctypes import POINTER, cast, CDLL, c_void_p, c_char_p, c_double, c_int, c_int64, byref

from copy import deepcopy
from pathlib import Path

from .casadi_function_generation import generate_c_code_explicit_ode, \
    generate_c_code_implicit_ode, generate_c_code_gnsf, generate_c_code_discrete_dynamics, \
    generate_c_code_constraint, generate_c_code_nls_cost, generate_c_code_conl_cost, \
    generate_c_code_external_cost
from .gnsf.detect_gnsf_structure import detect_gnsf_structure
from .acados_ocp import AcadosOcp
from .acados_model import AcadosModel
from .utils import is_column, is_empty, casadi_length, render_template,\
     format_class_dict, make_object_json_dumpable, make_model_consistent,\
     set_up_imported_gnsf_model, get_ocp_nlp_layout, get_python_interface_path, get_lib_ext, check_casadi_version
from .builders import CMakeBuilder


def make_ocp_dims_consistent(acados_ocp: AcadosOcp):
    dims = acados_ocp.dims
    cost = acados_ocp.cost
    constraints = acados_ocp.constraints
    model = acados_ocp.model
    opts = acados_ocp.solver_options

    # nx
    if is_column(model.x):
        dims.nx = casadi_length(model.x)
    else:
        raise Exception('model.x should be column vector!')

    # nu
    if is_empty(model.u):
        dims.nu = 0
    else:
        dims.nu = casadi_length(model.u)

    # nz
    if is_empty(model.z):
        dims.nz = 0
    else:
        dims.nz = casadi_length(model.z)

    # np
    if is_empty(model.p):
        dims.np = 0
    else:
        dims.np = casadi_length(model.p)
    if acados_ocp.parameter_values.shape[0] != dims.np:
        raise Exception('inconsistent dimension np, regarding model.p and parameter_values.' + \
            f'\nGot np = {dims.np}, acados_ocp.parameter_values.shape = {acados_ocp.parameter_values.shape[0]}\n')

    ## cost
    # initial stage - if not set, copy fields from path constraints
    if cost.cost_type_0 is None:
        cost.cost_type_0 = cost.cost_type
        cost.W_0 = cost.W
        cost.Vx_0 = cost.Vx
        cost.Vu_0 = cost.Vu
        cost.Vz_0 = cost.Vz
        cost.yref_0 = cost.yref
        cost.cost_ext_fun_type_0 = cost.cost_ext_fun_type
        model.cost_y_expr_0 = model.cost_y_expr
        model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
        model.cost_expr_ext_cost_custom_hess_0 = model.cost_expr_ext_cost_custom_hess

        model.cost_psi_expr_0 = model.cost_psi_expr
        model.cost_r_in_psi_expr_0 = model.cost_r_in_psi_expr

    if cost.cost_type_0 == 'LINEAR_LS':
        ny_0 = cost.W_0.shape[0]
        if cost.Vx_0.shape[0] != ny_0 or cost.Vu_0.shape[0] != ny_0:
            raise Exception('inconsistent dimension ny_0, regarding W_0, Vx_0, Vu_0.' + \
                            f'\nGot W_0[{cost.W_0.shape}], Vx_0[{cost.Vx_0.shape}], Vu_0[{cost.Vu_0.shape}]\n')
        if dims.nz != 0 and cost.Vz_0.shape[0] != ny_0:
            raise Exception('inconsistent dimension ny_0, regarding W_0, Vx_0, Vu_0, Vz_0.' + \
                            f'\nGot W_0[{cost.W_0.shape}], Vx_0[{cost.Vx_0.shape}], Vu_0[{cost.Vu_0.shape}], Vz_0[{cost.Vz_0.shape}]\n')
        if cost.Vx_0.shape[1] != dims.nx and ny_0 != 0:
            raise Exception('inconsistent dimension: Vx_0 should have nx columns.')
        if cost.Vu_0.shape[1] != dims.nu and ny_0 != 0:
            raise Exception('inconsistent dimension: Vu_0 should have nu columns.')
        if cost.yref_0.shape[0] != ny_0:
            raise Exception('inconsistent dimension: regarding W_0, yref_0.' + \
                            f'\nGot W_0[{cost.W_0.shape}], yref_0[{cost.yref_0.shape}]\n')
        dims.ny_0 = ny_0

    elif cost.cost_type_0 == 'NONLINEAR_LS':
        ny_0 = cost.W_0.shape[0]
        if is_empty(model.cost_y_expr_0) and ny_0 != 0:
            raise Exception('inconsistent dimension ny_0: regarding W_0, cost_y_expr.')
        elif casadi_length(model.cost_y_expr_0) != ny_0:
            raise Exception('inconsistent dimension ny_0: regarding W_0, cost_y_expr.')
        if cost.yref_0.shape[0] != ny_0:
            raise Exception('inconsistent dimension: regarding W_0, yref_0.' + \
                            f'\nGot W_0[{cost.W.shape}], yref_0[{cost.yref_0.shape}]\n')
        dims.ny_0 = ny_0

    elif cost.cost_type_0 == 'CONVEX_OVER_NONLINEAR':
        if is_empty(model.cost_y_expr_0):
            raise Exception('cost_y_expr_0 and/or cost_y_expr not provided.')
        ny_0 = casadi_length(model.cost_y_expr_0)
        if is_empty(model.cost_r_in_psi_expr_0) or casadi_length(model.cost_r_in_psi_expr_0) != ny_0:
            raise Exception('inconsistent dimension ny_0: regarding cost_y_expr_0 and cost_r_in_psi_0.')
        if is_empty(model.cost_psi_expr_0) or casadi_length(model.cost_psi_expr_0) != 1:
            raise Exception('cost_psi_expr_0 not provided or not scalar-valued.')
        if cost.yref_0.shape[0] != ny_0:
            raise Exception('inconsistent dimension: regarding yref_0 and cost_y_expr_0, cost_r_in_psi_0.')
        dims.ny_0 = ny_0

        if not (opts.hessian_approx=='EXACT' and opts.exact_hess_cost==False) and opts.hessian_approx != 'GAUSS_NEWTON':
            raise Exception("\nWith CONVEX_OVER_NONLINEAR cost type, possible Hessian approximations are:\n"
            "GAUSS_NEWTON or EXACT with 'exact_hess_cost' == False.\n")

    elif cost.cost_type_0 == 'EXTERNAL':
        if opts.hessian_approx == 'GAUSS_NEWTON' and opts.ext_cost_num_hess == 0 and model.cost_expr_ext_cost_custom_hess_0 is None:
            print("\nWARNING: Gauss-Newton Hessian approximation with EXTERNAL cost type not possible!\n"
            "got cost_type_0: EXTERNAL, hessian_approx: 'GAUSS_NEWTON.'\n"
            "GAUSS_NEWTON hessian is only supported for cost_types [NON]LINEAR_LS.\n"
            "If you continue, acados will proceed computing the exact hessian for the cost term.\n"
            "Note: There is also the option to use the external cost module with a numerical hessian approximation (see `ext_cost_num_hess`).\n"
            "OR the option to provide a symbolic custom hessian approximation (see `cost_expr_ext_cost_custom_hess`).\n")

    # path
    if cost.cost_type == 'LINEAR_LS':
        ny = cost.W.shape[0]
        if cost.Vx.shape[0] != ny or cost.Vu.shape[0] != ny:
            raise Exception('inconsistent dimension ny, regarding W, Vx, Vu.' + \
                            f'\nGot W[{cost.W.shape}], Vx[{cost.Vx.shape}], Vu[{cost.Vu.shape}]\n')
        if dims.nz != 0 and cost.Vz.shape[0] != ny:
            raise Exception('inconsistent dimension ny, regarding W, Vx, Vu, Vz.' + \
                            f'\nGot W[{cost.W.shape}], Vx[{cost.Vx.shape}], Vu[{cost.Vu.shape}], Vz[{cost.Vz.shape}]\n')
        if cost.Vx.shape[1] != dims.nx and ny != 0:
            raise Exception('inconsistent dimension: Vx should have nx columns.')
        if cost.Vu.shape[1] != dims.nu and ny != 0:
            raise Exception('inconsistent dimension: Vu should have nu columns.')
        if cost.yref.shape[0] != ny:
            raise Exception('inconsistent dimension: regarding W, yref.' + \
                            f'\nGot W[{cost.W.shape}], yref[{cost.yref.shape}]\n')
        dims.ny = ny

    elif cost.cost_type == 'NONLINEAR_LS':
        ny = cost.W.shape[0]
        if is_empty(model.cost_y_expr) and ny != 0:
            raise Exception('inconsistent dimension ny: regarding W, cost_y_expr.')
        elif casadi_length(model.cost_y_expr) != ny:
            raise Exception('inconsistent dimension ny: regarding W, cost_y_expr.')
        if cost.yref.shape[0] != ny:
            raise Exception('inconsistent dimension: regarding W, yref.' + \
                            f'\nGot W[{cost.W.shape}], yref[{cost.yref.shape}]\n')
        dims.ny = ny

    elif cost.cost_type == 'CONVEX_OVER_NONLINEAR':
        if is_empty(model.cost_y_expr):
            raise Exception('cost_y_expr and/or cost_y_expr not provided.')
        ny = casadi_length(model.cost_y_expr)
        if is_empty(model.cost_r_in_psi_expr) or casadi_length(model.cost_r_in_psi_expr) != ny:
            raise Exception('inconsistent dimension ny: regarding cost_y_expr and cost_r_in_psi.')
        if is_empty(model.cost_psi_expr) or casadi_length(model.cost_psi_expr) != 1:
            raise Exception('cost_psi_expr not provided or not scalar-valued.')
        if cost.yref.shape[0] != ny:
            raise Exception('inconsistent dimension: regarding yref and cost_y_expr, cost_r_in_psi.')
        dims.ny = ny

        if not (opts.hessian_approx=='EXACT' and opts.exact_hess_cost==False) and opts.hessian_approx != 'GAUSS_NEWTON':
            raise Exception("\nWith CONVEX_OVER_NONLINEAR cost type, possible Hessian approximations are:\n"
            "GAUSS_NEWTON or EXACT with 'exact_hess_cost' == False.\n")


    elif cost.cost_type == 'EXTERNAL':
        if opts.hessian_approx == 'GAUSS_NEWTON' and opts.ext_cost_num_hess == 0 and model.cost_expr_ext_cost_custom_hess is None:
            print("\nWARNING: Gauss-Newton Hessian approximation with EXTERNAL cost type not possible!\n"
            "got cost_type: EXTERNAL, hessian_approx: 'GAUSS_NEWTON.'\n"
            "GAUSS_NEWTON hessian is only supported for cost_types [NON]LINEAR_LS.\n"
            "If you continue, acados will proceed computing the exact hessian for the cost term.\n"
            "Note: There is also the option to use the external cost module with a numerical hessian approximation (see `ext_cost_num_hess`).\n"
            "OR the option to provide a symbolic custom hessian approximation (see `cost_expr_ext_cost_custom_hess`).\n")

    # terminal
    if cost.cost_type_e == 'LINEAR_LS':
        ny_e = cost.W_e.shape[0]
        if cost.Vx_e.shape[0] != ny_e:
            raise Exception('inconsistent dimension ny_e: regarding W_e, cost_y_expr_e.' + \
                f'\nGot W_e[{cost.W_e.shape}], Vx_e[{cost.Vx_e.shape}]')
        if cost.Vx_e.shape[1] != dims.nx and ny_e != 0:
            raise Exception('inconsistent dimension: Vx_e should have nx columns.')
        if cost.yref_e.shape[0] != ny_e:
            raise Exception('inconsistent dimension: regarding W_e, yref_e.')
        dims.ny_e = ny_e

    elif cost.cost_type_e == 'NONLINEAR_LS':
        ny_e = cost.W_e.shape[0]
        if is_empty(model.cost_y_expr_e) and ny_e != 0:
            raise Exception('inconsistent dimension ny_e: regarding W_e, cost_y_expr_e.')
        elif casadi_length(model.cost_y_expr_e) != ny_e:
            raise Exception('inconsistent dimension ny_e: regarding W_e, cost_y_expr_e.')
        if cost.yref_e.shape[0] != ny_e:
            raise Exception('inconsistent dimension: regarding W_e, yref_e.')
        dims.ny_e = ny_e

    elif cost.cost_type_e == 'CONVEX_OVER_NONLINEAR':
        if is_empty(model.cost_y_expr_e):
            raise Exception('cost_y_expr_e not provided.')
        ny_e = casadi_length(model.cost_y_expr_e)
        if is_empty(model.cost_r_in_psi_expr_e) or casadi_length(model.cost_r_in_psi_expr_e) != ny_e:
            raise Exception('inconsistent dimension ny_e: regarding cost_y_expr_e and cost_r_in_psi_e.')
        if is_empty(model.cost_psi_expr_e) or casadi_length(model.cost_psi_expr_e) != 1:
            raise Exception('cost_psi_expr_e not provided or not scalar-valued.')
        if cost.yref_e.shape[0] != ny_e:
            raise Exception('inconsistent dimension: regarding yref_e and cost_y_expr_e, cost_r_in_psi_e.')
        dims.ny_e = ny_e

        if not (opts.hessian_approx=='EXACT' and opts.exact_hess_cost==False) and opts.hessian_approx != 'GAUSS_NEWTON':
            raise Exception("\nWith CONVEX_OVER_NONLINEAR cost type, possible Hessian approximations are:\n"
            "GAUSS_NEWTON or EXACT with 'exact_hess_cost' == False.\n")



    elif cost.cost_type_e == 'EXTERNAL':
        if opts.hessian_approx == 'GAUSS_NEWTON' and opts.ext_cost_num_hess == 0 and model.cost_expr_ext_cost_custom_hess_e is None:
            print("\nWARNING: Gauss-Newton Hessian approximation with EXTERNAL cost type not possible!\n"
            "got cost_type_e: EXTERNAL, hessian_approx: 'GAUSS_NEWTON.'\n"
            "GAUSS_NEWTON hessian is only supported for cost_types [NON]LINEAR_LS.\n"
            "If you continue, acados will proceed computing the exact hessian for the cost term.\n"
            "Note: There is also the option to use the external cost module with a numerical hessian approximation (see `ext_cost_num_hess`).\n"
            "OR the option to provide a symbolic custom hessian approximation (see `cost_expr_ext_cost_custom_hess`).\n")

    ## constraints
    # initial
    this_shape = constraints.lbx_0.shape
    other_shape = constraints.ubx_0.shape
    if not this_shape == other_shape:
        raise Exception('lbx_0, ubx_0 have different shapes!')
    if not is_column(constraints.lbx_0):
        raise Exception('lbx_0, ubx_0 must be column vectors!')
    dims.nbx_0 = constraints.lbx_0.size

    if all(constraints.lbx_0 == constraints.ubx_0) and dims.nbx_0 == dims.nx \
        and dims.nbxe_0 is None \
        and (constraints.idxbxe_0.shape == constraints.idxbx_0.shape)\
            and all(constraints.idxbxe_0 == constraints.idxbx_0):
        # case: x0 was set: nbx0 are all equlities.
        dims.nbxe_0 = dims.nbx_0
    elif constraints.idxbxe_0 is not None:
        dims.nbxe_0 = constraints.idxbxe_0.shape[0]
    elif dims.nbxe_0 is None:
        # case: x0 and idxbxe_0 were not set -> dont assume nbx0 to be equality constraints.
        dims.nbxe_0 = 0

    # path
    nbx = constraints.idxbx.shape[0]
    if constraints.ubx.shape[0] != nbx or constraints.lbx.shape[0] != nbx:
        raise Exception('inconsistent dimension nbx, regarding idxbx, ubx, lbx.')
    else:
        dims.nbx = nbx

    nbu = constraints.idxbu.shape[0]
    if constraints.ubu.shape[0] != nbu or constraints.lbu.shape[0] != nbu:
        raise Exception('inconsistent dimension nbu, regarding idxbu, ubu, lbu.')
    else:
        dims.nbu = nbu

    ng = constraints.lg.shape[0]
    if constraints.ug.shape[0] != ng or constraints.C.shape[0] != ng \
       or constraints.D.shape[0] != ng:
        raise Exception('inconsistent dimension ng, regarding lg, ug, C, D.')
    else:
        dims.ng = ng

    if not is_empty(model.con_h_expr):
        nh = casadi_length(model.con_h_expr)
    else:
        nh = 0

    if constraints.uh.shape[0] != nh or constraints.lh.shape[0] != nh:
        raise Exception('inconsistent dimension nh, regarding lh, uh, con_h_expr.')
    else:
        dims.nh = nh

    if is_empty(model.con_phi_expr):
        dims.nphi = 0
        dims.nr = 0
    else:
        dims.nphi = casadi_length(model.con_phi_expr)
        if is_empty(model.con_r_expr):
            raise Exception('convex over nonlinear constraints: con_r_expr but con_phi_expr is nonempty')
        else:
            dims.nr = casadi_length(model.con_r_expr)

    # terminal
    nbx_e = constraints.idxbx_e.shape[0]
    if constraints.ubx_e.shape[0] != nbx_e or constraints.lbx_e.shape[0] != nbx_e:
        raise Exception('inconsistent dimension nbx_e, regarding idxbx_e, ubx_e, lbx_e.')
    else:
        dims.nbx_e = nbx_e

    ng_e = constraints.lg_e.shape[0]
    if constraints.ug_e.shape[0] != ng_e or constraints.C_e.shape[0] != ng_e:
        raise Exception('inconsistent dimension ng_e, regarding_e lg_e, ug_e, C_e.')
    else:
        dims.ng_e = ng_e

    if not is_empty(model.con_h_expr_e):
        nh_e = casadi_length(model.con_h_expr_e)
    else:
        nh_e = 0

    if constraints.uh_e.shape[0] != nh_e or constraints.lh_e.shape[0] != nh_e:
        raise Exception('inconsistent dimension nh_e, regarding lh_e, uh_e, con_h_expr_e.')
    else:
        dims.nh_e = nh_e

    if is_empty(model.con_phi_expr_e):
        dims.nphi_e = 0
        dims.nr_e = 0
    else:
        dims.nphi_e = casadi_length(model.con_phi_expr_e)
        if is_empty(model.con_r_expr_e):
            raise Exception('convex over nonlinear constraints: con_r_expr_e but con_phi_expr_e is nonempty')
        else:
            dims.nr_e = casadi_length(model.con_r_expr_e)

    # Slack dimensions
    nsbx = constraints.idxsbx.shape[0]
    if nsbx > nbx:
        raise Exception(f'inconsistent dimension nsbx = {nsbx}. Is greater than nbx = {nbx}.')
    if is_empty(constraints.lsbx):
        constraints.lsbx = np.zeros((nsbx,))
    elif constraints.lsbx.shape[0] != nsbx:
        raise Exception('inconsistent dimension nsbx, regarding idxsbx, lsbx.')
    if is_empty(constraints.usbx):
        constraints.usbx = np.zeros((nsbx,))
    elif constraints.usbx.shape[0] != nsbx:
        raise Exception('inconsistent dimension nsbx, regarding idxsbx, usbx.')
    dims.nsbx = nsbx

    nsbu = constraints.idxsbu.shape[0]
    if nsbu > nbu:
        raise Exception(f'inconsistent dimension nsbu = {nsbu}. Is greater than nbu = {nbu}.')
    if is_empty(constraints.lsbu):
        constraints.lsbu = np.zeros((nsbu,))
    elif constraints.lsbu.shape[0] != nsbu:
        raise Exception('inconsistent dimension nsbu, regarding idxsbu, lsbu.')
    if is_empty(constraints.usbu):
        constraints.usbu = np.zeros((nsbu,))
    elif constraints.usbu.shape[0] != nsbu:
        raise Exception('inconsistent dimension nsbu, regarding idxsbu, usbu.')
    dims.nsbu = nsbu

    nsh = constraints.idxsh.shape[0]
    if nsh > nh:
        raise Exception(f'inconsistent dimension nsh = {nsh}. Is greater than nh = {nh}.')
    if is_empty(constraints.lsh):
        constraints.lsh = np.zeros((nsh,))
    elif constraints.lsh.shape[0] != nsh:
        raise Exception('inconsistent dimension nsh, regarding idxsh, lsh.')
    if is_empty(constraints.ush):
        constraints.ush = np.zeros((nsh,))
    elif constraints.ush.shape[0] != nsh:
        raise Exception('inconsistent dimension nsh, regarding idxsh, ush.')
    dims.nsh = nsh

    nsphi = constraints.idxsphi.shape[0]
    if nsphi > dims.nphi:
        raise Exception(f'inconsistent dimension nsphi = {nsphi}. Is greater than nphi = {dims.nphi}.')
    if is_empty(constraints.lsphi):
        constraints.lsphi = np.zeros((nsphi,))
    elif constraints.lsphi.shape[0] != nsphi:
        raise Exception('inconsistent dimension nsphi, regarding idxsphi, lsphi.')
    if is_empty(constraints.usphi):
        constraints.usphi = np.zeros((nsphi,))
    elif constraints.usphi.shape[0] != nsphi:
        raise Exception('inconsistent dimension nsphi, regarding idxsphi, usphi.')
    dims.nsphi = nsphi

    nsg = constraints.idxsg.shape[0]
    if nsg > ng:
        raise Exception(f'inconsistent dimension nsg = {nsg}. Is greater than ng = {ng}.')
    if is_empty(constraints.lsg):
        constraints.lsg = np.zeros((nsg,))
    elif constraints.lsg.shape[0] != nsg:
        raise Exception('inconsistent dimension nsg, regarding idxsg, lsg.')
    if is_empty(constraints.usg):
        constraints.usg = np.zeros((nsg,))
    elif constraints.usg.shape[0] != nsg:
        raise Exception('inconsistent dimension nsg, regarding idxsg, usg.')
    dims.nsg = nsg

    ns = nsbx + nsbu + nsh + nsg + nsphi
    wrong_field = ""
    if cost.Zl.shape[0] != ns:
        wrong_field = "Zl"
        dim = cost.Zl.shape[0]
    elif cost.Zu.shape[0] != ns:
        wrong_field = "Zu"
        dim = cost.Zu.shape[0]
    elif cost.zl.shape[0] != ns:
        wrong_field = "zl"
        dim = cost.zl.shape[0]
    elif cost.zu.shape[0] != ns:
        wrong_field = "zu"
        dim = cost.zu.shape[0]

    if wrong_field != "":
        raise Exception(f'Inconsistent size for field {wrong_field}, with dimension {dim}, \n\t'\
            + f'Detected ns = {ns} = nsbx + nsbu + nsg + nsh + nsphi.\n\t'\
            + f'With nsbx = {nsbx}, nsbu = {nsbu}, nsg = {nsg}, nsh = {nsh}, nsphi = {nsphi}')

    dims.ns = ns

    nsbx_e = constraints.idxsbx_e.shape[0]
    if nsbx_e > nbx_e:
        raise Exception(f'inconsistent dimension nsbx_e = {nsbx_e}. Is greater than nbx_e = {nbx_e}.')
    if is_empty(constraints.lsbx_e):
        constraints.lsbx_e = np.zeros((nsbx_e,))
    elif constraints.lsbx_e.shape[0] != nsbx_e:
        raise Exception('inconsistent dimension nsbx_e, regarding idxsbx_e, lsbx_e.')
    if is_empty(constraints.usbx_e):
        constraints.usbx_e = np.zeros((nsbx_e,))
    elif constraints.usbx_e.shape[0] != nsbx_e:
        raise Exception('inconsistent dimension nsbx_e, regarding idxsbx_e, usbx_e.')
    dims.nsbx_e = nsbx_e

    nsh_e = constraints.idxsh_e.shape[0]
    if nsh_e > nh_e:
        raise Exception(f'inconsistent dimension nsh_e = {nsh_e}. Is greater than nh_e = {nh_e}.')
    if is_empty(constraints.lsh_e):
        constraints.lsh_e = np.zeros((nsh_e,))
    elif constraints.lsh_e.shape[0] != nsh_e:
        raise Exception('inconsistent dimension nsh_e, regarding idxsh_e, lsh_e.')
    if is_empty(constraints.ush_e):
        constraints.ush_e = np.zeros((nsh_e,))
    elif constraints.ush_e.shape[0] != nsh_e:
        raise Exception('inconsistent dimension nsh_e, regarding idxsh_e, ush_e.')
    dims.nsh_e = nsh_e

    nsg_e = constraints.idxsg_e.shape[0]
    if nsg_e > ng_e:
        raise Exception(f'inconsistent dimension nsg_e = {nsg_e}. Is greater than ng_e = {ng_e}.')
    if is_empty(constraints.lsg_e):
        constraints.lsg_e = np.zeros((nsg_e,))
    elif constraints.lsg_e.shape[0] != nsg_e:
        raise Exception('inconsistent dimension nsg_e, regarding idxsg_e, lsg_e.')
    if is_empty(constraints.usg_e):
        constraints.usg_e = np.zeros((nsg_e,))
    elif constraints.usg_e.shape[0] != nsg_e:
        raise Exception('inconsistent dimension nsg_e, regarding idxsg_e, usg_e.')
    dims.nsg_e = nsg_e

    nsphi_e = constraints.idxsphi_e.shape[0]
    if nsphi_e > dims.nphi_e:
        raise Exception(f'inconsistent dimension nsphi_e = {nsphi_e}. Is greater than nphi_e = {dims.nphi_e}.')
    if is_empty(constraints.lsphi_e):
        constraints.lsphi_e = np.zeros((nsphi_e,))
    elif constraints.lsphi_e.shape[0] != nsphi_e:
        raise Exception('inconsistent dimension nsphi_e, regarding idxsphi_e, lsphi_e.')
    if is_empty(constraints.usphi_e):
        constraints.usphi_e = np.zeros((nsphi_e,))
    elif constraints.usphi_e.shape[0] != nsphi_e:
        raise Exception('inconsistent dimension nsphi_e, regarding idxsphi_e, usphi_e.')
    dims.nsphi_e = nsphi_e

    # terminal
    ns_e = nsbx_e + nsh_e + nsg_e + nsphi_e
    wrong_field = ""
    if cost.Zl_e.shape[0] != ns_e:
        wrong_field = "Zl_e"
        dim = cost.Zl_e.shape[0]
    elif cost.Zu_e.shape[0] != ns_e:
        wrong_field = "Zu_e"
        dim = cost.Zu_e.shape[0]
    elif cost.zl_e.shape[0] != ns_e:
        wrong_field = "zl_e"
        dim = cost.zl_e.shape[0]
    elif cost.zu_e.shape[0] != ns_e:
        wrong_field = "zu_e"
        dim = cost.zu_e.shape[0]

    if wrong_field != "":
        raise Exception(f'Inconsistent size for field {wrong_field}, with dimension {dim}, \n\t'\
            + f'Detected ns_e = {ns_e} = nsbx_e + nsg_e + nsh_e + nsphi_e.\n\t'\
            + f'With nsbx_e = {nsbx_e}, nsg_e = {nsg_e}, nsh_e = {nsh_e}, nsphi_e = {nsphi_e}')

    dims.ns_e = ns_e

    # discretization
    if is_empty(opts.time_steps) and is_empty(opts.shooting_nodes):
        # uniform discretization
        opts.time_steps = opts.tf / dims.N * np.ones((dims.N,))

    elif not is_empty(opts.shooting_nodes):
        if np.shape(opts.shooting_nodes)[0] != dims.N+1:
            raise Exception('inconsistent dimension N, regarding shooting_nodes.')

        time_steps = opts.shooting_nodes[1:] - opts.shooting_nodes[0:-1]
        # identify constant time_steps: due to numerical reasons the content of time_steps might vary a bit
        avg_time_steps = np.average(time_steps)
        # criterion for constant time step detection: the min/max difference in values normalized by the average
        check_const_time_step = (np.max(time_steps)-np.min(time_steps)) / avg_time_steps
        # if the criterion is small, we have a constant time_step
        if check_const_time_step < 1e-9:
            time_steps[:] = avg_time_steps  # if we have a constant time_step: apply the average time_step

        opts.time_steps = time_steps

    elif (not is_empty(opts.time_steps)) and (not is_empty(opts.shooting_nodes)):
        Exception('Please provide either time_steps or shooting_nodes for nonuniform discretization')

    tf = np.sum(opts.time_steps)
    if (tf - opts.tf) / tf > 1e-15:
        raise Exception(f'Inconsistent discretization: {opts.tf}'\
            f' = tf != sum(opts.time_steps) = {tf}.')

    # num_steps
    if isinstance(opts.sim_method_num_steps, np.ndarray) and opts.sim_method_num_steps.size == 1:
        opts.sim_method_num_steps = opts.sim_method_num_steps.item()

    if isinstance(opts.sim_method_num_steps, (int, float)) and opts.sim_method_num_steps % 1 == 0:
        opts.sim_method_num_steps = opts.sim_method_num_steps * np.ones((dims.N,), dtype=np.int64)
    elif isinstance(opts.sim_method_num_steps, np.ndarray) and opts.sim_method_num_steps.size == dims.N \
           and np.all(np.equal(np.mod(opts.sim_method_num_steps, 1), 0)):
        opts.sim_method_num_steps = np.reshape(opts.sim_method_num_steps, (dims.N,)).astype(np.int64)
    else:
        raise Exception("Wrong value for sim_method_num_steps. Should be either int or array of ints of shape (N,).")

    # num_stages
    if isinstance(opts.sim_method_num_stages, np.ndarray) and opts.sim_method_num_stages.size == 1:
        opts.sim_method_num_stages = opts.sim_method_num_stages.item()

    if isinstance(opts.sim_method_num_stages, (int, float)) and opts.sim_method_num_stages % 1 == 0:
        opts.sim_method_num_stages = opts.sim_method_num_stages * np.ones((dims.N,), dtype=np.int64)
    elif isinstance(opts.sim_method_num_stages, np.ndarray) and opts.sim_method_num_stages.size == dims.N \
           and np.all(np.equal(np.mod(opts.sim_method_num_stages, 1), 0)):
        opts.sim_method_num_stages = np.reshape(opts.sim_method_num_stages, (dims.N,)).astype(np.int64)
    else:
        raise Exception("Wrong value for sim_method_num_stages. Should be either int or array of ints of shape (N,).")

    # jac_reuse
    if isinstance(opts.sim_method_jac_reuse, np.ndarray) and opts.sim_method_jac_reuse.size == 1:
        opts.sim_method_jac_reuse = opts.sim_method_jac_reuse.item()

    if isinstance(opts.sim_method_jac_reuse, (int, float)) and opts.sim_method_jac_reuse % 1 == 0:
        opts.sim_method_jac_reuse = opts.sim_method_jac_reuse * np.ones((dims.N,), dtype=np.int64)
    elif isinstance(opts.sim_method_jac_reuse, np.ndarray) and opts.sim_method_jac_reuse.size == dims.N \
           and np.all(np.equal(np.mod(opts.sim_method_jac_reuse, 1), 0)):
        opts.sim_method_jac_reuse = np.reshape(opts.sim_method_jac_reuse, (dims.N,)).astype(np.int64)
    else:
        raise Exception("Wrong value for sim_method_jac_reuse. Should be either int or array of ints of shape (N,).")


def get_simulink_default_opts():
    python_interface_path = get_python_interface_path()
    abs_path = os.path.join(python_interface_path, 'simulink_default_opts.json')
    with open(abs_path , 'r') as f:
        simulink_default_opts = json.load(f)
    return simulink_default_opts


def ocp_formulation_json_dump(acados_ocp, simulink_opts=None, json_file='acados_ocp_nlp.json'):
    # Load acados_ocp_nlp structure description
    ocp_layout = get_ocp_nlp_layout()

    # Copy input ocp object dictionary
    ocp_nlp_dict = dict(deepcopy(acados_ocp).__dict__)
    # TODO: maybe make one function with formatting

    for acados_struct, v in ocp_layout.items():
        # skip non dict attributes
        if not isinstance(v, dict):
            continue
        #  setattr(ocp_nlp, acados_struct, dict(getattr(acados_ocp, acados_struct).__dict__))
        # Copy ocp object attributes dictionaries
        ocp_nlp_dict[acados_struct]=dict(getattr(acados_ocp, acados_struct).__dict__)

    ocp_nlp_dict = format_class_dict(ocp_nlp_dict)

    if simulink_opts is not None:
        ocp_nlp_dict['simulink_opts'] = simulink_opts

    with open(json_file, 'w') as f:
        json.dump(ocp_nlp_dict, f, default=make_object_json_dumpable, indent=4, sort_keys=True)



def ocp_formulation_json_load(json_file='acados_ocp_nlp.json'):
    # Load acados_ocp_nlp structure description
    ocp_layout = get_ocp_nlp_layout()

    with open(json_file, 'r') as f:
        ocp_nlp_json = json.load(f)

    ocp_nlp_dict = json2dict(ocp_nlp_json, ocp_nlp_json['dims'])

    # Instantiate AcadosOcp object
    acados_ocp = AcadosOcp()

    # load class dict
    acados_ocp.__dict__ = ocp_nlp_dict

    # load class attributes dict, dims, constraints, etc
    for acados_struct, v in ocp_layout.items():
        # skip non dict attributes
        if not isinstance(v, dict):
            continue
        acados_attribute = getattr(acados_ocp, acados_struct)
        acados_attribute.__dict__ = ocp_nlp_dict[acados_struct]
        setattr(acados_ocp, acados_struct, acados_attribute)

    return acados_ocp


def ocp_generate_external_functions(acados_ocp: AcadosOcp, model: AcadosModel):

    model = make_model_consistent(model)

    if acados_ocp.solver_options.hessian_approx == 'EXACT':
        opts = dict(generate_hess=1)
    else:
        opts = dict(generate_hess=0)

    # create code_export_dir, model_dir
    code_export_dir = acados_ocp.code_export_directory
    opts['code_export_directory'] = code_export_dir
    model_dir = os.path.join(code_export_dir, model.name + '_model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    check_casadi_version()
    # TODO: remove dir gen from all the generate_c_* functions
    if acados_ocp.model.dyn_ext_fun_type == 'casadi':
        if acados_ocp.solver_options.integrator_type == 'ERK':
            generate_c_code_explicit_ode(model, opts)
        elif acados_ocp.solver_options.integrator_type == 'IRK':
            generate_c_code_implicit_ode(model, opts)
        elif acados_ocp.solver_options.integrator_type == 'LIFTED_IRK':
            generate_c_code_implicit_ode(model, opts)
        elif acados_ocp.solver_options.integrator_type == 'GNSF':
            generate_c_code_gnsf(model, opts)
        elif acados_ocp.solver_options.integrator_type == 'DISCRETE':
            generate_c_code_discrete_dynamics(model, opts)
        else:
            raise Exception("ocp_generate_external_functions: unknown integrator type.")
    else:
        target_location = os.path.join(code_export_dir, model_dir, model.dyn_generic_source)
        shutil.copyfile(model.dyn_generic_source, target_location)

    if acados_ocp.dims.nphi > 0 or acados_ocp.dims.nh > 0:
        generate_c_code_constraint(model, model.name, False, opts)

    if acados_ocp.dims.nphi_e > 0 or acados_ocp.dims.nh_e > 0:
        generate_c_code_constraint(model, model.name, True, opts)

    if acados_ocp.cost.cost_type_0 == 'NONLINEAR_LS':
        generate_c_code_nls_cost(model, model.name, 'initial', opts)
    elif acados_ocp.cost.cost_type_0 == 'CONVEX_OVER_NONLINEAR':
        generate_c_code_conl_cost(model, model.name, 'initial', opts)
    elif acados_ocp.cost.cost_type_0 == 'EXTERNAL':
        generate_c_code_external_cost(model, 'initial', opts)

    if acados_ocp.cost.cost_type == 'NONLINEAR_LS':
        generate_c_code_nls_cost(model, model.name, 'path', opts)
    elif acados_ocp.cost.cost_type == 'CONVEX_OVER_NONLINEAR':
        generate_c_code_conl_cost(model, model.name, 'path', opts)
    elif acados_ocp.cost.cost_type == 'EXTERNAL':
        generate_c_code_external_cost(model, 'path', opts)

    if acados_ocp.cost.cost_type_e == 'NONLINEAR_LS':
        generate_c_code_nls_cost(model, model.name, 'terminal', opts)
    elif acados_ocp.cost.cost_type_e == 'CONVEX_OVER_NONLINEAR':
        generate_c_code_conl_cost(model, model.name, 'terminal', opts)
    elif acados_ocp.cost.cost_type_e == 'EXTERNAL':
        generate_c_code_external_cost(model, 'terminal', opts)


def ocp_get_default_cmake_builder() -> CMakeBuilder:
    """
    If :py:class:`~acados_template.acados_ocp_solver.AcadosOcpSolver` is used with `CMake` this function returns a good first setting.
    :return: default :py:class:`~acados_template.builders.CMakeBuilder`
    """
    cmake_builder = CMakeBuilder()
    cmake_builder.options_on = ['BUILD_ACADOS_OCP_SOLVER_LIB']
    return cmake_builder



def ocp_render_templates(acados_ocp: AcadosOcp, json_file, cmake_builder=None, simulink_opts=None):

    # setting up loader and environment
    json_path = os.path.abspath(json_file)

    if not os.path.exists(json_path):
        raise Exception(f'Path "{json_path}" not found!')

    # Render templates
    template_list = __ocp_get_template_list(acados_ocp, cmake_builder=cmake_builder, simulink_opts=simulink_opts)
    for tup in template_list:
        if len(tup) > 2:
            output_dir = tup[2]
        else:
            output_dir = acados_ocp.code_export_directory
        render_template(tup[0], tup[1], output_dir, json_path)

    # Custom templates
    acados_template_path = os.path.dirname(os.path.abspath(__file__))
    custom_template_glob = os.path.join(acados_template_path, 'custom_update_templates', '*')
    for tup in acados_ocp.solver_options.custom_templates:
        render_template(tup[0], tup[1], acados_ocp.code_export_directory, json_path, template_glob=custom_template_glob)

    return



def __ocp_get_template_list(acados_ocp: AcadosOcp, cmake_builder=None, simulink_opts=None) -> list:
    """
    returns a list of tuples in the form:
    (input_filename, output_filname)
    or
    (input_filename, output_filname, output_directory)
    """
    name = acados_ocp.model.name
    code_export_directory = acados_ocp.code_export_directory
    template_list = []

    template_list.append(('main.in.c', f'main_{name}.c'))
    template_list.append(('acados_solver.in.c', f'acados_solver_{name}.c'))
    template_list.append(('acados_solver.in.h', f'acados_solver_{name}.h'))
    template_list.append(('acados_solver.in.pxd', f'acados_solver.pxd'))
    if cmake_builder is not None:
        template_list.append(('CMakeLists.in.txt', 'CMakeLists.txt'))
    else:
        template_list.append(('Makefile.in', 'Makefile'))


    # sim
    template_list.append(('acados_sim_solver.in.c', f'acados_sim_solver_{name}.c'))
    template_list.append(('acados_sim_solver.in.h', f'acados_sim_solver_{name}.h'))
    template_list.append(('main_sim.in.c', f'main_sim_{name}.c'))

    # model
    model_dir = os.path.join(code_export_directory, f'{name}_model')
    template_list.append(('model.in.h', f'{name}_model.h', model_dir))
    # constraints
    constraints_dir = os.path.join(code_export_directory, f'{name}_constraints')
    template_list.append(('constraints.in.h', f'{name}_constraints.h', constraints_dir))
    # cost
    cost_dir = os.path.join(code_export_directory, f'{name}_cost')
    template_list.append(('cost.in.h', f'{name}_cost.h', cost_dir))

    # Simulink
    if simulink_opts is not None:
        template_file = os.path.join('matlab_templates', 'acados_solver_sfun.in.c')
        template_list.append((template_file, f'acados_solver_sfunction_{name}.c'))
        template_file = os.path.join('matlab_templates', 'acados_solver_sfun.in.c')
        template_list.append((template_file, f'make_sfun_{name}.m'))

    return template_list


def remove_x0_elimination(acados_ocp):
    acados_ocp.constraints.idxbxe_0 = np.zeros((0,))
    acados_ocp.dims.nbxe_0 = 0


class AcadosOcpSolver:
    """
    Class to interact with the acados ocp solver C object.

        :param acados_ocp: type :py:class:`~acados_template.acados_ocp.AcadosOcp` - description of the OCP for acados
        :param json_file: name for the json file used to render the templated code - default: acados_ocp_nlp.json
        :param simulink_opts: Options to configure Simulink S-function blocks, mainly to activate possible Inputs and Outputs
    """
    if sys.platform=="win32":
        from ctypes import wintypes
        from ctypes import WinDLL
        dlclose = WinDLL('kernel32', use_last_error=True).FreeLibrary
        dlclose.argtypes = [wintypes.HMODULE]
    else:
        dlclose = CDLL(None).dlclose
        dlclose.argtypes = [c_void_p]

    @classmethod
    def generate(cls, acados_ocp: AcadosOcp, json_file='acados_ocp_nlp.json', simulink_opts=None, cmake_builder: CMakeBuilder = None):
        """
        Generates the code for an acados OCP solver, given the description in acados_ocp.
            :param acados_ocp: type AcadosOcp - description of the OCP for acados
            :param json_file: name for the json file used to render the templated code - default: `acados_ocp_nlp.json`
            :param simulink_opts: Options to configure Simulink S-function blocks, mainly to activate possible inputs and
                   outputs; default: `None`
            :param cmake_builder: type :py:class:`~acados_template.builders.CMakeBuilder` generate a `CMakeLists.txt` and use
                   the `CMake` pipeline instead of a `Makefile` (`CMake` seems to be the better option in conjunction with
                   `MS Visual Studio`); default: `None`
        """
        model = acados_ocp.model
        acados_ocp.code_export_directory = os.path.abspath(acados_ocp.code_export_directory)

        # make dims consistent
        make_ocp_dims_consistent(acados_ocp)

        # module dependent post processing
        if acados_ocp.solver_options.integrator_type == 'GNSF':
            if 'gnsf_model' in acados_ocp.__dict__:
                set_up_imported_gnsf_model(acados_ocp)
            else:
                detect_gnsf_structure(acados_ocp)

        if acados_ocp.solver_options.qp_solver == 'PARTIAL_CONDENSING_QPDUNES':
            remove_x0_elimination(acados_ocp)

        # set integrator time automatically
        acados_ocp.solver_options.Tsim = acados_ocp.solver_options.time_steps[0]

        # generate external functions
        ocp_generate_external_functions(acados_ocp, model)

        # dump to json
        acados_ocp.json_file = json_file
        ocp_formulation_json_dump(acados_ocp, simulink_opts=simulink_opts, json_file=json_file)

        # render templates
        ocp_render_templates(acados_ocp, json_file, cmake_builder=cmake_builder, simulink_opts=simulink_opts)

        # copy custom update function
        if acados_ocp.solver_options.custom_update_filename != "" and acados_ocp.solver_options.custom_update_copy:
            target_location = os.path.join(acados_ocp.code_export_directory, acados_ocp.solver_options.custom_update_filename)
            shutil.copyfile(acados_ocp.solver_options.custom_update_filename, target_location)
        if acados_ocp.solver_options.custom_update_header_filename != "" and acados_ocp.solver_options.custom_update_copy:
            target_location = os.path.join(acados_ocp.code_export_directory, acados_ocp.solver_options.custom_update_header_filename)
            shutil.copyfile(acados_ocp.solver_options.custom_update_header_filename, target_location)


    @classmethod
    def build(cls, code_export_dir, with_cython=False, cmake_builder: CMakeBuilder = None, verbose: bool = True):
        """
        Builds the code for an acados OCP solver, that has been generated in code_export_dir
            :param code_export_dir: directory in which acados OCP solver has been generated, see generate()
            :param with_cython: option indicating if the cython interface is build, default: False.
            :param cmake_builder: type :py:class:`~acados_template.builders.CMakeBuilder` generate a `CMakeLists.txt` and use
                   the `CMake` pipeline instead of a `Makefile` (`CMake` seems to be the better option in conjunction with
                   `MS Visual Studio`); default: `None`
            :param verbose: indicating if build command is printed
        """
        code_export_dir = os.path.abspath(code_export_dir)
        cwd = os.getcwd()
        os.chdir(code_export_dir)
        if with_cython:
            call(
                ['make', 'clean_all'],
                stdout=None if verbose else DEVNULL,
                stderr=None if verbose else STDOUT
            )
            call(
                ['make', 'ocp_cython'],
                stdout=None if verbose else DEVNULL,
                stderr=None if verbose else STDOUT
            )
        else:
            if cmake_builder is not None:
                cmake_builder.exec(code_export_dir)
            else:
                call(
                    ['make', 'clean_ocp_shared_lib'],
                    stdout=None if verbose else DEVNULL,
                    stderr=None if verbose else STDOUT
                )
                call(
                    ['make', 'ocp_shared_lib'],
                    stdout=None if verbose else DEVNULL,
                    stderr=None if verbose else STDOUT
                )
        os.chdir(cwd)


    @classmethod
    def create_cython_solver(cls, json_file):
        """
        Returns an `AcadosOcpSolverCython` object.

        This is an alternative Cython based Python wrapper to the acados OCP solver in C.
        This offers faster interaction with the solver, because getter and setter calls, which include shape checking are done in compiled C code.

        The default wrapper `AcadosOcpSolver` is based on ctypes.
        """
        with open(json_file, 'r') as f:
            acados_ocp_json = json.load(f)
        code_export_directory = acados_ocp_json['code_export_directory']

        importlib.invalidate_caches()
        rel_code_export_directory = os.path.relpath(code_export_directory)
        acados_ocp_solver_pyx = importlib.import_module(f'{rel_code_export_directory}.acados_ocp_solver_pyx')

        AcadosOcpSolverCython = getattr(acados_ocp_solver_pyx, 'AcadosOcpSolverCython')
        return AcadosOcpSolverCython(acados_ocp_json['model']['name'],
                    acados_ocp_json['solver_options']['nlp_solver_type'],
                    acados_ocp_json['dims']['N'])


    def __init__(self, acados_ocp: AcadosOcp, json_file='acados_ocp_nlp.json', simulink_opts=None, build=True, generate=True, cmake_builder: CMakeBuilder = None, verbose=True):

        self.solver_created = False
        if generate:
            self.generate(acados_ocp, json_file=json_file, simulink_opts=simulink_opts, cmake_builder=cmake_builder)

        # load json, store options in object
        with open(json_file, 'r') as f:
            acados_ocp_json = json.load(f)
        self.N = acados_ocp_json['dims']['N']
        self.model_name = acados_ocp_json['model']['name']
        self.solver_options = acados_ocp_json['solver_options']

        acados_lib_path = acados_ocp_json['acados_lib_path']
        code_export_directory = acados_ocp_json['code_export_directory']

        if build:
            self.build(code_export_directory, with_cython=False, cmake_builder=cmake_builder, verbose=verbose)

        # prepare library loading
        lib_prefix = 'lib'
        lib_ext = get_lib_ext()
        if os.name == 'nt':
            lib_prefix = ''

        # Load acados library to avoid unloading the library.
        # This is necessary if acados was compiled with OpenMP, since the OpenMP threads can't be destroyed.
        # Unloading a library which uses OpenMP results in a segfault (on any platform?).
        # see [https://stackoverflow.com/questions/34439956/vc-crash-when-freeing-a-dll-built-with-openmp]
        # or [https://python.hotexamples.com/examples/_ctypes/-/dlclose/python-dlclose-function-examples.html]
        libacados_name = f'{lib_prefix}acados{lib_ext}'
        libacados_filepath = os.path.join(acados_lib_path, libacados_name)
        self.__acados_lib = CDLL(libacados_filepath)
        # find out if acados was compiled with OpenMP
        try:
            self.__acados_lib_uses_omp = getattr(self.__acados_lib, 'omp_get_thread_num') is not None
        except AttributeError as e:
            self.__acados_lib_uses_omp = False
        if self.__acados_lib_uses_omp:
            print('acados was compiled with OpenMP.')
        else:
            print('acados was compiled without OpenMP.')
        libacados_ocp_solver_name = f'{lib_prefix}acados_ocp_solver_{self.model_name}{lib_ext}'
        self.shared_lib_name = os.path.join(code_export_directory, libacados_ocp_solver_name)

        # get shared_lib
        self.shared_lib = CDLL(self.shared_lib_name)

        # create capsule
        getattr(self.shared_lib, f"{self.model_name}_acados_create_capsule").restype = c_void_p
        self.capsule = getattr(self.shared_lib, f"{self.model_name}_acados_create_capsule")()

        # create solver
        getattr(self.shared_lib, f"{self.model_name}_acados_create").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_create").restype = c_int
        assert getattr(self.shared_lib, f"{self.model_name}_acados_create")(self.capsule)==0
        self.solver_created = True

        self.acados_ocp = acados_ocp

        # get pointers solver
        self.__get_pointers_solver()

        self.status = 0

        # gettable fields
        self.__qp_dynamics_fields = ['A', 'B', 'b']
        self.__qp_cost_fields = ['Q', 'R', 'S', 'q', 'r']
        self.__qp_constraint_fields = ['C', 'D', 'lg', 'ug', 'lbx', 'ubx', 'lbu', 'ubu']


    def __get_pointers_solver(self):
        # """
        # Private function to get the pointers for solver
        # """
        # get pointers solver
        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_opts").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_opts").restype = c_void_p
        self.nlp_opts = getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_opts")(self.capsule)

        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_dims").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_dims").restype = c_void_p
        self.nlp_dims = getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_dims")(self.capsule)

        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_config").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_config").restype = c_void_p
        self.nlp_config = getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_config")(self.capsule)

        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_out").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_out").restype = c_void_p
        self.nlp_out = getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_out")(self.capsule)

        getattr(self.shared_lib, f"{self.model_name}_acados_get_sens_out").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_get_sens_out").restype = c_void_p
        self.sens_out = getattr(self.shared_lib, f"{self.model_name}_acados_get_sens_out")(self.capsule)

        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_in").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_in").restype = c_void_p
        self.nlp_in = getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_in")(self.capsule)

        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_solver").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_solver").restype = c_void_p
        self.nlp_solver = getattr(self.shared_lib, f"{self.model_name}_acados_get_nlp_solver")(self.capsule)



    def solve_for_x0(self, x0_bar):
        """
        Wrapper around `solve()` which sets initial state constraint, solves the OCP, and returns u0.
        """
        self.set(0, "lbx", x0_bar)
        self.set(0, "ubx", x0_bar)

        status = self.solve()

        if status == 2:
            print("Warning: acados_ocp_solver reached maximum iterations.")
        elif status != 0:
            raise Exception(f'acados acados_ocp_solver returned status {status}')

        u0 = self.get(0, "u")
        return u0


    def solve(self):
        """
        Solve the ocp with current input.
        """
        getattr(self.shared_lib, f"{self.model_name}_acados_solve").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_solve").restype = c_int
        self.status = getattr(self.shared_lib, f"{self.model_name}_acados_solve")(self.capsule)

        return self.status


    def custom_update(self, data_: np.ndarray):
        """
        A custom function that can be implemented by a user to be called between solver calls.
        By default this does nothing.
        The idea is to have a convenient wrapper to do complex updates of parameters and numerical data efficiently in C,
        in a function that is compiled into the solver library and can be conveniently used in the Python environment.
        """
        data = np.ascontiguousarray(data_, dtype=np.float64)
        c_data = cast(data.ctypes.data, POINTER(c_double))
        data_len = len(data)

        getattr(self.shared_lib, f"{self.model_name}_acados_custom_update").argtypes = [c_void_p, POINTER(c_double), c_int]
        getattr(self.shared_lib, f"{self.model_name}_acados_custom_update").restype = c_int
        status = getattr(self.shared_lib, f"{self.model_name}_acados_custom_update")(self.capsule, c_data, data_len)

        return status


    def reset(self, reset_qp_solver_mem=1):
        """
        Sets current iterate to all zeros.
        """
        getattr(self.shared_lib, f"{self.model_name}_acados_reset").argtypes = [c_void_p, c_int]
        getattr(self.shared_lib, f"{self.model_name}_acados_reset").restype = c_int
        getattr(self.shared_lib, f"{self.model_name}_acados_reset")(self.capsule, reset_qp_solver_mem)

        return


    def set_new_time_steps(self, new_time_steps):
        """
        Set new time steps.
        Recreates the solver if N changes.

            :param new_time_steps: 1 dimensional np array of new time steps for the solver

            .. note:: This allows for different use-cases: either set a new size of time_steps or a new distribution of
                      the shooting nodes without changing the number, e.g., to reach a different final time. Both cases
                      do not require a new code export and compilation.
        """

        # unlikely but still possible
        if not self.solver_created:
            raise Exception('Solver was not yet created!')

        # check if time steps really changed in value
        if np.array_equal(self.solver_options['time_steps'], new_time_steps):
            return

        N = new_time_steps.size
        new_time_steps_data = cast(new_time_steps.ctypes.data, POINTER(c_double))

        # check if recreation of acados is necessary (no need to recreate acados if sizes are identical)
        if len(self.solver_options['time_steps']) == N:
            getattr(self.shared_lib, f"{self.model_name}_acados_update_time_steps").argtypes = [c_void_p, c_int, c_void_p]
            getattr(self.shared_lib, f"{self.model_name}_acados_update_time_steps").restype = c_int
            assert getattr(self.shared_lib, f"{self.model_name}_acados_update_time_steps")(self.capsule, N, new_time_steps_data) == 0
        else:  # recreate the solver with the new time steps
            self.solver_created = False

            # delete old memory (analog to __del__)
            getattr(self.shared_lib, f"{self.model_name}_acados_free").argtypes = [c_void_p]
            getattr(self.shared_lib, f"{self.model_name}_acados_free").restype = c_int
            getattr(self.shared_lib, f"{self.model_name}_acados_free")(self.capsule)

            # create solver with new time steps
            getattr(self.shared_lib, f"{self.model_name}_acados_create_with_discretization").argtypes = [c_void_p, c_int, c_void_p]
            getattr(self.shared_lib, f"{self.model_name}_acados_create_with_discretization").restype = c_int
            assert getattr(self.shared_lib, f"{self.model_name}_acados_create_with_discretization")(self.capsule, N, new_time_steps_data) == 0

            self.solver_created = True

            # get pointers solver
            self.__get_pointers_solver()

        # store time_steps, N
        self.solver_options['time_steps'] = new_time_steps
        self.N = N
        self.solver_options['Tsim'] = self.solver_options['time_steps'][0]


    def update_qp_solver_cond_N(self, qp_solver_cond_N: int):
        """
        Recreate solver with new value `qp_solver_cond_N` with a partial condensing QP solver.
        This function is relevant for code reuse, i.e., if either `set_new_time_steps(...)` is used or
        the influence of a different `qp_solver_cond_N` is studied without code export and compilation.
            :param qp_solver_cond_N: new number of condensing stages for the solver

            .. note:: This function can only be used in combination with a partial condensing QP solver.

            .. note:: After `set_new_time_steps(...)` is used and depending on the new number of time steps it might be
                      necessary to change `qp_solver_cond_N` as well (using this function), i.e., typically
                      `qp_solver_cond_N < N`.
        """
        # unlikely but still possible
        if not self.solver_created:
            raise Exception('Solver was not yet created!')
        if self.N < qp_solver_cond_N:
            raise Exception('Setting qp_solver_cond_N to be larger than N does not work!')
        if self.solver_options['qp_solver_cond_N'] != qp_solver_cond_N:
            self.solver_created = False

            # recreate the solver
            fun_name = f'{self.model_name}_acados_update_qp_solver_cond_N'
            getattr(self.shared_lib, fun_name).argtypes = [c_void_p, c_int]
            getattr(self.shared_lib, fun_name).restype = c_int
            assert getattr(self.shared_lib, fun_name)(self.capsule, qp_solver_cond_N) == 0

            # store the new value
            self.solver_options['qp_solver_cond_N'] = qp_solver_cond_N
            self.solver_created = True

            # get pointers solver
            self.__get_pointers_solver()


    def eval_param_sens(self, index, stage=0, field="ex"):
        """
        Calculate the sensitivity of the curent solution with respect to the initial state component of index

            :param index: integer corresponding to initial state index in range(nx)
        """

        field_ = field
        field = field_.encode('utf-8')

        # checks
        if not isinstance(index, int):
            raise Exception('AcadosOcpSolver.eval_param_sens(): index must be Integer.')

        self.shared_lib.ocp_nlp_dims_get_from_attr.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_char_p]
        self.shared_lib.ocp_nlp_dims_get_from_attr.restype = c_int
        nx = self.shared_lib.ocp_nlp_dims_get_from_attr(self.nlp_config, self.nlp_dims, self.nlp_out, 0, "x".encode('utf-8'))

        if index < 0 or index > nx:
            raise Exception(f'AcadosOcpSolver.eval_param_sens(): index must be in [0, nx-1], got: {index}.')

        # actual eval_param
        self.shared_lib.ocp_nlp_eval_param_sens.argtypes = [c_void_p, c_char_p, c_int, c_int, c_void_p]
        self.shared_lib.ocp_nlp_eval_param_sens.restype = None
        self.shared_lib.ocp_nlp_eval_param_sens(self.nlp_solver, field, stage, index, self.sens_out)

        return


    def get(self, stage_, field_):
        """
        Get the last solution of the solver:

            :param stage: integer corresponding to shooting node
            :param field: string in ['x', 'u', 'z', 'pi', 'lam', 't', 'sl', 'su',]

            .. note:: regarding lam, t: \n
                    the inequalities are internally organized in the following order: \n
                    [ lbu lbx lg lh lphi ubu ubx ug uh uphi; \n
                      lsbu lsbx lsg lsh lsphi usbu usbx usg ush usphi]

            .. note:: pi: multipliers for dynamics equality constraints \n
                      lam: multipliers for inequalities \n
                      t: slack variables corresponding to evaluation of all inequalities (at the solution) \n
                      sl: slack variables of soft lower inequality constraints \n
                      su: slack variables of soft upper inequality constraints \n
        """

        out_fields = ['x', 'u', 'z', 'pi', 'lam', 't', 'sl', 'su']
        # mem_fields = ['sl', 'su']
        sens_fields = ['sens_u', "sens_x"]
        all_fields = out_fields + sens_fields

        field = field_

        if (field_ not in all_fields):
            raise Exception(f'AcadosOcpSolver.get(stage={stage_}, field={field_}): \'{field_}\' is an invalid argument.\
                    \n Possible values are {all_fields}.')

        if not isinstance(stage_, int):
            raise Exception(f'AcadosOcpSolver.get(stage={stage_}, field={field_}): stage index must be an integer, got type {type(stage_)}.')

        if stage_ < 0 or stage_ > self.N:
            raise Exception(f'AcadosOcpSolver.get(stage={stage_}, field={field_}): stage index must be in [0, {self.N}], got: {stage_}.')

        if stage_ == self.N and field_ == 'pi':
            raise Exception(f'AcadosOcpSolver.get(stage={stage_}, field={field_}): field \'{field_}\' does not exist at final stage {stage_}.')

        if field_ in sens_fields:
            field = field_.replace('sens_', '')

        field = field.encode('utf-8')

        self.shared_lib.ocp_nlp_dims_get_from_attr.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p]
        self.shared_lib.ocp_nlp_dims_get_from_attr.restype = c_int

        dims = self.shared_lib.ocp_nlp_dims_get_from_attr(self.nlp_config, \
            self.nlp_dims, self.nlp_out, stage_, field)

        out = np.ascontiguousarray(np.zeros((dims,)), dtype=np.float64)
        out_data = cast(out.ctypes.data, POINTER(c_double))

        if (field_ in out_fields):
            self.shared_lib.ocp_nlp_out_get.argtypes = \
                [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
            self.shared_lib.ocp_nlp_out_get(self.nlp_config, \
                self.nlp_dims, self.nlp_out, stage_, field, out_data)
        # elif field_ in mem_fields:
        #     self.shared_lib.ocp_nlp_get_at_stage.argtypes = \
        #         [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
        #     self.shared_lib.ocp_nlp_get_at_stage(self.nlp_config, \
        #         self.nlp_dims, self.nlp_solver, stage_, field, out_data)
        elif field_ in sens_fields:
            self.shared_lib.ocp_nlp_out_get.argtypes = \
                [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
            self.shared_lib.ocp_nlp_out_get(self.nlp_config, \
                self.nlp_dims, self.sens_out, stage_, field, out_data)

        return out


    def print_statistics(self):
        """
        prints statistics of previous solver run as a table:
            - iter: iteration number
            - res_stat: stationarity residual
            - res_eq: residual wrt equality constraints (dynamics)
            - res_ineq: residual wrt inequality constraints (constraints)
            - res_comp: residual wrt complementarity conditions
            - qp_stat: status of QP solver
            - qp_iter: number of QP iterations
            - alpha: SQP step size
            - qp_res_stat: stationarity residual of the last QP solution
            - qp_res_eq: residual wrt equality constraints (dynamics) of the last QP solution
            - qp_res_ineq: residual wrt inequality constraints (constraints)  of the last QP solution
            - qp_res_comp: residual wrt complementarity conditions of the last QP solution
        """
        stat = self.get_stats("statistics")

        if self.solver_options['nlp_solver_type'] == 'SQP':
            print('\niter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha')
            if stat.shape[0]>8:
                print('\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp')
            for jj in range(stat.shape[1]):
                print(f'{int(stat[0][jj]):d}\t{stat[1][jj]:e}\t{stat[2][jj]:e}\t{stat[3][jj]:e}\t' +
                      f'{stat[4][jj]:e}\t{int(stat[5][jj]):d}\t{int(stat[6][jj]):d}\t{stat[7][jj]:e}\t')
                if stat.shape[0]>8:
                    print('\t{:e}\t{:e}\t{:e}\t{:e}'.format( \
                        stat[8][jj], stat[9][jj], stat[10][jj], stat[11][jj]))
            print('\n')
        elif self.solver_options['nlp_solver_type'] == 'SQP_RTI':
            print('\niter\tqp_stat\tqp_iter')
            if stat.shape[0]>3:
                print('\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp')
            for jj in range(stat.shape[1]):
                print('{:d}\t{:d}\t{:d}'.format( int(stat[0][jj]), int(stat[1][jj]), int(stat[2][jj])))
                if stat.shape[0]>3:
                    print('\t{:e}\t{:e}\t{:e}\t{:e}'.format( \
                         stat[3][jj], stat[4][jj], stat[5][jj], stat[6][jj]))
            print('\n')

        return


    def store_iterate(self, filename: str = '', overwrite=False):
        """
        Stores the current iterate of the ocp solver in a json file.

            :param filename: if not set, use f'{self.model_name}_iterate.json'
            :param overwrite: if false and filename exists add timestamp to filename
        """
        if filename == '':
            filename = f'{self.model_name}_iterate.json'

        if not overwrite:
            # append timestamp
            if os.path.isfile(filename):
                filename = filename[:-5]
                filename += datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S.%f') + '.json'

        # get iterate:
        solution = dict()

        lN = len(str(self.N+1))
        for i in range(self.N+1):
            i_string = f'{i:0{lN}d}'
            solution['x_'+i_string] = self.get(i,'x')
            solution['u_'+i_string] = self.get(i,'u')
            solution['z_'+i_string] = self.get(i,'z')
            solution['lam_'+i_string] = self.get(i,'lam')
            solution['t_'+i_string] = self.get(i, 't')
            solution['sl_'+i_string] = self.get(i, 'sl')
            solution['su_'+i_string] = self.get(i, 'su')
            if i < self.N:
                solution['pi_'+i_string] = self.get(i,'pi')

        for k in list(solution.keys()):
            if len(solution[k]) == 0:
                del solution[k]

        # save
        with open(filename, 'w') as f:
            json.dump(solution, f, default=make_object_json_dumpable, indent=4, sort_keys=True)
        print("stored current iterate in ", os.path.join(os.getcwd(), filename))



    def dump_last_qp_to_json(self, filename: str = '', overwrite=False):
        """
        Dumps the latest QP data into a json file

            :param filename: if not set, use model_name + timestamp + '.json'
            :param overwrite: if false and filename exists add timestamp to filename
        """
        if filename == '':
            filename = f'{self.model_name}_QP.json'

        if not overwrite:
            # append timestamp
            if os.path.isfile(filename):
                filename = filename[:-5]
                filename += datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S.%f') + '.json'

        # get QP data:
        qp_data = dict()

        lN = len(str(self.N+1))
        for field in self.__qp_dynamics_fields:
            for i in range(self.N):
                qp_data[f'{field}_{i:0{lN}d}'] = self.get_from_qp_in(i,field)

        for field in self.__qp_constraint_fields + self.__qp_cost_fields:
            for i in range(self.N+1):
                qp_data[f'{field}_{i:0{lN}d}'] = self.get_from_qp_in(i,field)

        # remove empty fields
        for k in list(qp_data.keys()):
            if len(qp_data[k]) == 0:
                del qp_data[k]

        # save
        with open(filename, 'w') as f:
            json.dump(qp_data, f, default=make_object_json_dumpable, indent=4, sort_keys=True)
        print("stored qp from solver memory in ", os.path.join(os.getcwd(), filename))



    def load_iterate(self, filename):
        """
        Loads the iterate stored in json file with filename into the ocp solver.
        """
        if not os.path.isfile(filename):
            raise Exception('load_iterate: failed, file does not exist: ' + os.path.join(os.getcwd(), filename))

        with open(filename, 'r') as f:
            solution = json.load(f)

        print(f"loading iterate {filename}")
        for key in solution.keys():
            (field, stage) = key.split('_')
            self.set(int(stage), field, np.array(solution[key]))


    def get_stats(self, field_):
        """
        Get the information of the last solver call.

            :param field: string in ['statistics', 'time_tot', 'time_lin', 'time_sim', 'time_sim_ad', 'time_sim_la', 'time_qp', 'time_qp_solver_call', 'time_reg', 'sqp_iter', 'residuals', 'qp_iter', 'alpha']

        Available fileds:
            - time_tot: total CPU time previous call
            - time_lin: CPU time for linearization
            - time_sim: CPU time for integrator
            - time_sim_ad: CPU time for integrator contribution of external function calls
            - time_sim_la: CPU time for integrator contribution of linear algebra
            - time_qp: CPU time qp solution
            - time_qp_solver_call: CPU time inside qp solver (without converting the QP)
            - time_qp_xcond: time_glob: CPU time globalization
            - time_solution_sensitivities: CPU time for previous call to eval_param_sens
            - time_reg: CPU time regularization
            - sqp_iter: number of SQP iterations
            - qp_iter: vector of QP iterations for last SQP call
            - statistics: table with info about last iteration
            - stat_m: number of rows in statistics matrix
            - stat_n: number of columns in statistics matrix
            - residuals: residuals of last iterate
            - alpha: step sizes of SQP iterations
        """

        double_fields = ['time_tot',
                  'time_lin',
                  'time_sim',
                  'time_sim_ad',
                  'time_sim_la',
                  'time_qp',
                  'time_qp_solver_call',
                  'time_qp_xcond',
                  'time_glob',
                  'time_solution_sensitivities',
                  'time_reg'
        ]
        fields = double_fields + [
                  'sqp_iter',
                  'qp_iter',
                  'statistics',
                  'stat_m',
                  'stat_n',
                  'residuals',
                  'alpha',
                ]
        field = field_.encode('utf-8')


        if field_ in ['sqp_iter', 'stat_m', 'stat_n']:
            out = np.ascontiguousarray(np.zeros((1,)), dtype=np.int64)
            out_data = cast(out.ctypes.data, POINTER(c_int64))
            self.shared_lib.ocp_nlp_get.argtypes = [c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)
            return out

        # TODO: just return double instead of np.
        elif field_ in double_fields:
            out = np.zeros((1,))
            out_data = cast(out.ctypes.data, POINTER(c_double))
            self.shared_lib.ocp_nlp_get.argtypes = [c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)
            return out

        elif field_ == 'statistics':
            sqp_iter = self.get_stats("sqp_iter")
            stat_m = self.get_stats("stat_m")
            stat_n = self.get_stats("stat_n")
            min_size = min([stat_m, sqp_iter+1])
            out = np.ascontiguousarray(
                        np.zeros((stat_n[0]+1, min_size[0])), dtype=np.float64)
            out_data = cast(out.ctypes.data, POINTER(c_double))
            self.shared_lib.ocp_nlp_get.argtypes = [c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)
            return out

        elif field_ == 'qp_iter':
            full_stats = self.get_stats('statistics')
            if self.solver_options['nlp_solver_type'] == 'SQP':
                return full_stats[6, :]
            elif self.solver_options['nlp_solver_type'] == 'SQP_RTI':
                return full_stats[2, :]

        elif field_ == 'alpha':
            full_stats = self.get_stats('statistics')
            if self.solver_options['nlp_solver_type'] == 'SQP':
                return full_stats[7, :]
            else: # self.solver_options['nlp_solver_type'] == 'SQP_RTI':
                raise Exception("alpha values are not available for SQP_RTI")

        elif field_ == 'residuals':
            return self.get_residuals()

        else:
            raise Exception(f'AcadosOcpSolver.get_stats(): \'{field}\' is not a valid argument.'
                    + f'\n Possible values are {fields}.')


    def get_cost(self):
        """
        Returns the cost value of the current solution.
        """
        # compute cost internally
        self.shared_lib.ocp_nlp_eval_cost.argtypes = [c_void_p, c_void_p, c_void_p]
        self.shared_lib.ocp_nlp_eval_cost(self.nlp_solver, self.nlp_in, self.nlp_out)

        # create output array
        out = np.ascontiguousarray(np.zeros((1,)), dtype=np.float64)
        out_data = cast(out.ctypes.data, POINTER(c_double))

        # call getter
        self.shared_lib.ocp_nlp_get.argtypes = [c_void_p, c_void_p, c_char_p, c_void_p]

        field = "cost_value".encode('utf-8')
        self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)

        return out[0]


    def get_residuals(self, recompute=False):
        """
        Returns an array of the form [res_stat, res_eq, res_ineq, res_comp].
        This residual has to be computed for SQP_RTI solver, since it is not available by default.

        - res_stat: stationarity residual
        - res_eq: residual wrt equality constraints (dynamics)
        - res_ineq: residual wrt inequality constraints (constraints)
        - res_comp: residual wrt complementarity conditions
        """
        # compute residuals if RTI
        if self.solver_options['nlp_solver_type'] == 'SQP_RTI' or recompute:
            self.shared_lib.ocp_nlp_eval_residuals.argtypes = [c_void_p, c_void_p, c_void_p]
            self.shared_lib.ocp_nlp_eval_residuals(self.nlp_solver, self.nlp_in, self.nlp_out)

        # create output array
        out = np.ascontiguousarray(np.zeros((4, 1)), dtype=np.float64)
        out_data = cast(out.ctypes.data, POINTER(c_double))

        # call getters
        self.shared_lib.ocp_nlp_get.argtypes = [c_void_p, c_void_p, c_char_p, c_void_p]

        field = "res_stat".encode('utf-8')
        self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)

        out_data = cast(out[1].ctypes.data, POINTER(c_double))
        field = "res_eq".encode('utf-8')
        self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)

        out_data = cast(out[2].ctypes.data, POINTER(c_double))
        field = "res_ineq".encode('utf-8')
        self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)

        out_data = cast(out[3].ctypes.data, POINTER(c_double))
        field = "res_comp".encode('utf-8')
        self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)
        return out.flatten()


    # Note: this function should not be used anymore, better use cost_set, constraints_set
    def set(self, stage_, field_, value_):
        """
        Set numerical data inside the solver.

            :param stage: integer corresponding to shooting node
            :param field: string in ['x', 'u', 'pi', 'lam', 't', 'p', 'xdot_guess', 'z_guess']

            .. note:: regarding lam, t: \n
                    the inequalities are internally organized in the following order: \n
                    [ lbu lbx lg lh lphi ubu ubx ug uh uphi; \n
                      lsbu lsbx lsg lsh lsphi usbu usbx usg ush usphi]

            .. note:: pi: multipliers for dynamics equality constraints \n
                      lam: multipliers for inequalities \n
                      t: slack variables corresponding to evaluation of all inequalities (at the solution) \n
                      sl: slack variables of soft lower inequality constraints \n
                      su: slack variables of soft upper inequality constraints \n
        """
        cost_fields = ['y_ref', 'yref']
        constraints_fields = ['lbx', 'ubx', 'lbu', 'ubu']
        out_fields = ['x', 'u', 'pi', 'lam', 't', 'z', 'sl', 'su']
        mem_fields = ['xdot_guess', 'z_guess']

        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = value_.astype(float)

        field = field_.encode('utf-8')

        stage = c_int(stage_)

        # treat parameters separately
        if field_ == 'p':
            getattr(self.shared_lib, f"{self.model_name}_acados_update_params").argtypes = [c_void_p, c_int, POINTER(c_double), c_int]
            getattr(self.shared_lib, f"{self.model_name}_acados_update_params").restype = c_int

            value_data = cast(value_.ctypes.data, POINTER(c_double))

            assert getattr(self.shared_lib, f"{self.model_name}_acados_update_params")(self.capsule, stage, value_data, value_.shape[0])==0
        else:
            if field_ not in constraints_fields + cost_fields + out_fields + mem_fields:
                raise Exception(f"AcadosOcpSolver.set(): '{field}' is not a valid argument.\n"
                    f" Possible values are {constraints_fields + cost_fields + out_fields + mem_fields + ['p']}.")

            self.shared_lib.ocp_nlp_dims_get_from_attr.argtypes = \
                [c_void_p, c_void_p, c_void_p, c_int, c_char_p]
            self.shared_lib.ocp_nlp_dims_get_from_attr.restype = c_int

            dims = self.shared_lib.ocp_nlp_dims_get_from_attr(self.nlp_config, \
                self.nlp_dims, self.nlp_out, stage_, field)

            if value_.shape[0] != dims:
                msg = f'AcadosOcpSolver.set(): mismatching dimension for field "{field_}" '
                msg += f'with dimension {dims} (you have {value_.shape[0]})'
                raise Exception(msg)

            value_data = cast(value_.ctypes.data, POINTER(c_double))
            value_data_p = cast((value_data), c_void_p)

            if field_ in constraints_fields:
                self.shared_lib.ocp_nlp_constraints_model_set.argtypes = \
                    [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_constraints_model_set(self.nlp_config, \
                    self.nlp_dims, self.nlp_in, stage, field, value_data_p)
            elif field_ in cost_fields:
                self.shared_lib.ocp_nlp_cost_model_set.argtypes = \
                    [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_cost_model_set(self.nlp_config, \
                    self.nlp_dims, self.nlp_in, stage, field, value_data_p)
            elif field_ in out_fields:
                self.shared_lib.ocp_nlp_out_set.argtypes = \
                    [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_out_set(self.nlp_config, \
                    self.nlp_dims, self.nlp_out, stage, field, value_data_p)
            elif field_ in mem_fields:
                self.shared_lib.ocp_nlp_set.argtypes = \
                    [c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_set(self.nlp_config, \
                    self.nlp_solver, stage, field, value_data_p)
            # also set z_guess, when setting z.
            if field_ == 'z':
                field = 'z_guess'.encode('utf-8')
                self.shared_lib.ocp_nlp_set.argtypes = \
                    [c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_set(self.nlp_config, \
                    self.nlp_solver, stage, field, value_data_p)
        return


    def cost_set(self, stage_, field_, value_, api='warn'):
        """
        Set numerical data in the cost module of the solver.

            :param stage: integer corresponding to shooting node
            :param field: string, e.g. 'yref', 'W', 'ext_cost_num_hess', 'zl', 'zu', 'Zl', 'Zu'
            :param value: of appropriate size
        """
        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = value_.astype(float)

        field = field_
        field = field.encode('utf-8')

        stage = c_int(stage_)
        self.shared_lib.ocp_nlp_cost_dims_get_from_attr.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p, POINTER(c_int)]
        self.shared_lib.ocp_nlp_cost_dims_get_from_attr.restype = c_int

        dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
        dims_data = cast(dims.ctypes.data, POINTER(c_int))

        self.shared_lib.ocp_nlp_cost_dims_get_from_attr(self.nlp_config, \
            self.nlp_dims, self.nlp_out, stage_, field, dims_data)

        value_shape = value_.shape
        if len(value_shape) == 1:
            value_shape = (value_shape[0], 0)

        elif len(value_shape) == 2:
            if api=='old':
                pass
            elif api=='warn':
                if not np.all(np.ravel(value_, order='F')==np.ravel(value_, order='K')):
                    raise Exception("Ambiguity in API detected.\n"
                                    "Are you making an acados model from scrach? Add api='new' to cost_set and carry on.\n"
                                    "Are you seeing this error suddenly in previously running code? Read on.\n"
                                    f"  You are relying on a now-fixed bug in cost_set for field '{field_}'.\n" +
                                    "  acados_template now correctly passes on any matrices to acados in column major format.\n" +
                                    "  Two options to fix this error: \n" +
                                    "   * Add api='old' to cost_set to restore old incorrect behaviour\n" +
                                    "   * Add api='new' to cost_set and remove any unnatural manipulation of the value argument " +
                                    "such as non-mathematical transposes, reshaping, casting to fortran order, etc... " +
                                    "If there is no such manipulation, then you have probably been getting an incorrect solution before.")
                # Get elements in column major order
                value_ = np.ravel(value_, order='F')
            elif api=='new':
                # Get elements in column major order
                value_ = np.ravel(value_, order='F')
            else:
                raise Exception("Unknown api: '{}'".format(api))

        if value_shape != tuple(dims):
            raise Exception('AcadosOcpSolver.cost_set(): mismatching dimension' +
                f' for field "{field_}" at stage {stage} with dimension {tuple(dims)} (you have {value_shape})')

        value_data = cast(value_.ctypes.data, POINTER(c_double))
        value_data_p = cast((value_data), c_void_p)

        self.shared_lib.ocp_nlp_cost_model_set.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
        self.shared_lib.ocp_nlp_cost_model_set(self.nlp_config, \
            self.nlp_dims, self.nlp_in, stage, field, value_data_p)

        return


    def constraints_set(self, stage_, field_, value_, api='warn'):
        """
        Set numerical data in the constraint module of the solver.

            :param stage: integer corresponding to shooting node
            :param field: string in ['lbx', 'ubx', 'lbu', 'ubu', 'lg', 'ug', 'lh', 'uh', 'uphi', 'C', 'D']
            :param value: of appropriate size
        """
        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = value_.astype(float)

        field = field_
        field = field.encode('utf-8')

        stage = c_int(stage_)
        self.shared_lib.ocp_nlp_constraint_dims_get_from_attr.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p, POINTER(c_int)]
        self.shared_lib.ocp_nlp_constraint_dims_get_from_attr.restype = c_int

        dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
        dims_data = cast(dims.ctypes.data, POINTER(c_int))

        self.shared_lib.ocp_nlp_constraint_dims_get_from_attr(self.nlp_config, \
            self.nlp_dims, self.nlp_out, stage_, field, dims_data)

        value_shape = value_.shape
        if len(value_shape) == 1:
            value_shape = (value_shape[0], 0)
        elif len(value_shape) == 2:
            if api=='old':
                pass
            elif api=='warn':
                if not np.all(np.ravel(value_, order='F')==np.ravel(value_, order='K')):
                    raise Exception("Ambiguity in API detected.\n"
                                    "Are you making an acados model from scrach? Add api='new' to constraints_set and carry on.\n"
                                    "Are you seeing this error suddenly in previously running code? Read on.\n"
                                    f"  You are relying on a now-fixed bug in constraints_set for field '{field}'.\n" +
                                    "  acados_template now correctly passes on any matrices to acados in column major format.\n" +
                                    "  Two options to fix this error: \n" +
                                    "   * Add api='old' to constraints_set to restore old incorrect behaviour\n" +
                                    "   * Add api='new' to constraints_set and remove any unnatural manipulation of the value argument " +
                                    "such as non-mathematical transposes, reshaping, casting to fortran order, etc... " +
                                    "If there is no such manipulation, then you have probably been getting an incorrect solution before.")
                # Get elements in column major order
                value_ = np.ravel(value_, order='F')
            elif api=='new':
                # Get elements in column major order
                value_ = np.ravel(value_, order='F')
            else:
                raise Exception(f"Unknown api: '{api}'")

        if value_shape != tuple(dims):
            raise Exception(f'AcadosOcpSolver.constraints_set(): mismatching dimension' +
                f' for field "{field_}" at stage {stage} with dimension {tuple(dims)} (you have {value_shape})')

        value_data = cast(value_.ctypes.data, POINTER(c_double))
        value_data_p = cast((value_data), c_void_p)

        self.shared_lib.ocp_nlp_constraints_model_set.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
        self.shared_lib.ocp_nlp_constraints_model_set(self.nlp_config, \
            self.nlp_dims, self.nlp_in, stage, field, value_data_p)

        return


    def get_from_qp_in(self, stage_: int, field_: str):
        """
        Get numerical data from the current QP.

            :param stage: integer corresponding to shooting node
            :param field: string in ['A', 'B', 'b', 'Q', 'R', 'S', 'q', 'r', 'C', 'D', 'lg', 'ug', 'lbx', 'ubx', 'lbu', 'ubu']
        """
        # idx* should be added too..
        if not isinstance(stage_, int):
            raise TypeError("stage should be int")
        if stage_ > self.N:
            raise Exception("stage should be <= self.N")
        if field_ in self.__qp_dynamics_fields and stage_ >= self.N:
            raise ValueError(f"dynamics field {field_} not available at terminal stage")
        if field_ not in self.__qp_dynamics_fields + self.__qp_cost_fields + self.__qp_constraint_fields:
            raise Exception(f"field {field_} not supported.")

        field = field_.encode('utf-8')
        stage = c_int(stage_)

        # get dims
        self.shared_lib.ocp_nlp_qp_dims_get_from_attr.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p, POINTER(c_int)]
        self.shared_lib.ocp_nlp_qp_dims_get_from_attr.restype = c_int

        dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
        dims_data = cast(dims.ctypes.data, POINTER(c_int))

        self.shared_lib.ocp_nlp_qp_dims_get_from_attr(self.nlp_config, \
            self.nlp_dims, self.nlp_out, stage_, field, dims_data)

        # create output data
        out = np.ascontiguousarray(np.zeros((np.prod(dims),)), dtype=np.float64)
        out = out.reshape(dims[0], dims[1], order='F')

        out_data = cast(out.ctypes.data, POINTER(c_double))
        out_data_p = cast((out_data), c_void_p)

        # call getter
        self.shared_lib.ocp_nlp_get_at_stage.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
        self.shared_lib.ocp_nlp_get_at_stage(self.nlp_config, \
            self.nlp_dims, self.nlp_solver, stage, field, out_data_p)

        return out


    def options_set(self, field_, value_):
        """
        Set options of the solver.

            :param field: string, e.g. 'print_level', 'rti_phase', 'initialize_t_slacks', 'step_length', 'alpha_min', 'alpha_reduction', 'qp_warm_start', 'line_search_use_sufficient_descent', 'full_step_dual', 'globalization_use_SOC', 'qp_tol_stat', 'qp_tol_eq', 'qp_tol_ineq', 'qp_tol_comp', 'qp_tau_min', 'qp_mu0'

            :param value: of type int, float, string

            - qp_tol_stat: QP solver tolerance stationarity
            - qp_tol_eq: QP solver tolerance equalities
            - qp_tol_ineq: QP solver tolerance inequalities
            - qp_tol_comp: QP solver tolerance complementarity
            - qp_tau_min: for HPIPM QP solvers: minimum value of barrier parameter in HPIPM
            - qp_mu0: for HPIPM QP solvers: initial value for complementarity slackness
            - warm_start_first_qp: indicates if first QP in SQP is warm_started
        """
        int_fields = ['print_level', 'rti_phase', 'initialize_t_slacks', 'qp_warm_start',
                      'line_search_use_sufficient_descent', 'full_step_dual', 'globalization_use_SOC', 'warm_start_first_qp']
        double_fields = ['step_length', 'tol_eq', 'tol_stat', 'tol_ineq', 'tol_comp', 'alpha_min', 'alpha_reduction',
                         'eps_sufficient_descent', 'qp_tol_stat', 'qp_tol_eq', 'qp_tol_ineq', 'qp_tol_comp', 'qp_tau_min', 'qp_mu0']
        string_fields = ['globalization']

        # check field availability and type
        if field_ in int_fields:
            if not isinstance(value_, int):
                raise Exception(f'solver option \'{field_}\' must be of type int. You have {type(value_)}.')
            else:
                value_ctypes = c_int(value_)

        elif field_ in double_fields:
            if not isinstance(value_, float):
                raise Exception(f'solver option \'{field_}\' must be of type float. You have {type(value_)}.')
            else:
                value_ctypes = c_double(value_)

        elif field_ in string_fields:
            if not isinstance(value_, str):
                raise Exception(f'solver option \'{field_}\' must be of type str. You have {type(value_)}.')
            else:
                value_ctypes = value_.encode('utf-8')
        else:
            fields = ', '.join(int_fields + double_fields + string_fields)
            raise Exception(f'AcadosOcpSolver.options_set() does not support field \'{field_}\'.\n'\
                f' Possible values are {fields}.')


        if field_ == 'rti_phase':
            if value_ < 0 or value_ > 2:
                raise Exception('AcadosOcpSolver.options_set(): argument \'rti_phase\' can '
                    'take only values 0, 1, 2 for SQP-RTI-type solvers')
            if self.solver_options['nlp_solver_type'] != 'SQP_RTI' and value_ > 0:
                raise Exception('AcadosOcpSolver.options_set(): argument \'rti_phase\' can '
                    'take only value 0 for SQP-type solvers')

        # encode
        field = field_
        field = field.encode('utf-8')

        # call C interface
        if field_ in string_fields:
            self.shared_lib.ocp_nlp_solver_opts_set.argtypes = \
                [c_void_p, c_void_p, c_char_p, c_char_p]
            self.shared_lib.ocp_nlp_solver_opts_set(self.nlp_config, \
                self.nlp_opts, field, value_ctypes)
        else:
            self.shared_lib.ocp_nlp_solver_opts_set.argtypes = \
                [c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.ocp_nlp_solver_opts_set(self.nlp_config, \
                self.nlp_opts, field, byref(value_ctypes))
        return


    def set_params_sparse(self, stage_, idx_values_, param_values_):
        """
        set parameters of the solvers external function partially:
        Pseudo: solver.param[idx_values_] = param_values_;
        Parameters:
            :param stage_: integer corresponding to shooting node
            :param idx_values_: 0 based np array (or iterable) of integers: indices of parameter to be set
            :param param_values_: new parameter values as numpy array
        """

        # if not isinstance(idx_values_, np.ndarray) or not issubclass(type(idx_values_[0]), np.integer):
        #     raise Exception('idx_values_ must be np.array of integers.')

        if not isinstance(param_values_, np.ndarray):
            raise Exception('param_values_ must be np.array.')
        elif np.float64 != param_values_.dtype:
            raise TypeError('param_values_ must be np.array of float64.')

        if param_values_.shape[0] != len(idx_values_):
            raise Exception(f'param_values_ and idx_values_ must be of the same size.' +
                 f' Got sizes idx {param_values_.shape[0]}, param_values {len(idx_values_)}.')

        if any(idx_values_ >= self.acados_ocp.dims.np):
            raise Exception(f'idx_values_ contains value >= np = {self.acados_ocp.dims.np}')

        stage = c_int(stage_)
        n_update = c_int(len(param_values_))

        param_data = cast(param_values_.ctypes.data, POINTER(c_double))
        c_idx_values = np.ascontiguousarray(idx_values_, dtype=np.intc)
        idx_data = cast(c_idx_values.ctypes.data, POINTER(c_int))

        getattr(self.shared_lib, f"{self.model_name}_acados_update_params_sparse").argtypes = \
                        [c_void_p, c_int, POINTER(c_int), POINTER(c_double), c_int]
        getattr(self.shared_lib, f"{self.model_name}_acados_update_params_sparse").restype = c_int
        getattr(self.shared_lib, f"{self.model_name}_acados_update_params_sparse") \
                                    (self.capsule, stage, idx_data, param_data, n_update)

    def __del__(self):
        if self.solver_created:
            getattr(self.shared_lib, f"{self.model_name}_acados_free").argtypes = [c_void_p]
            getattr(self.shared_lib, f"{self.model_name}_acados_free").restype = c_int
            getattr(self.shared_lib, f"{self.model_name}_acados_free")(self.capsule)

            getattr(self.shared_lib, f"{self.model_name}_acados_free_capsule").argtypes = [c_void_p]
            getattr(self.shared_lib, f"{self.model_name}_acados_free_capsule").restype = c_int
            getattr(self.shared_lib, f"{self.model_name}_acados_free_capsule")(self.capsule)

            try:
                self.dlclose(self.shared_lib._handle)
            except:
                print(f"WARNING: acados Python interface could not close shared_lib handle of AcadosOcpSolver {self.model_name}.\n",
                     "Attempting to create a new one with the same name will likely result in the old one being used!")
                pass
