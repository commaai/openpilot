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
#   Author: Jonathan Frey: jonathanpaulfrey(at)gmail.com

from casadi import *
from ..utils import casadi_length, is_empty


def determine_input_nonlinearity_function(gnsf):

    ## Description
    # this function takes a structure gnsf and updates the matrices L_x,
    # L_xdot, L_z, L_u and CasADi vectors y, uhat of this structure as follows:

    # given a CasADi expression phi_expr, which may depend on the variables
    # (x1, x1dot, z, u), this function determines a vector y (uhat) consisting
    # of all components of (x1, x1dot, z) (respectively u) that enter phi_expr.
    # Additionally matrices L_x, L_xdot, L_z, L_u are determined such that
    #           y    = L_x * x + L_xdot * xdot + L_z * z
    #           uhat = L_u * u
    # Furthermore the dimensions ny, nuhat, n_out are updated

    ## y
    y = SX.sym('y', 0, 0)
    # components of x1
    for ii in range(gnsf["nx1"]):
        if which_depends(gnsf["phi_expr"], gnsf["x"][ii])[0]:
            y = vertcat(y, gnsf["x"][ii])
        # else:
        # x[ii] is not part of y
    # components of x1dot
    for ii in range(gnsf["nx1"]):
        if which_depends(gnsf["phi_expr"], gnsf["xdot"][ii])[0]:
            print(gnsf["phi_expr"], "depends on", gnsf["xdot"][ii])
            y = vertcat(y, gnsf["xdot"][ii])
        # else:
        # xdot[ii] is not part of y
    # components of z
    for ii in range(gnsf["nz1"]):
        if which_depends(gnsf["phi_expr"], gnsf["z"][ii])[0]:
            y = vertcat(y, gnsf["z"][ii])
        # else:
        # z[ii] is not part of y
    ## uhat
    uhat = SX.sym('uhat', 0, 0)
    # components of u
    for ii in range(gnsf["nu"]):
        if which_depends(gnsf["phi_expr"], gnsf["u"][ii])[0]:
            uhat = vertcat(uhat, gnsf["u"][ii])
        # else:
        # u[ii] is not part of uhat
    ## generate gnsf['phi_expr_fun']
    # linear input matrices
    if is_empty(y):
        gnsf["L_x"] = []
        gnsf["L_xdot"] = []
        gnsf["L_u"] = []
        gnsf["L_z"] = []
    else:
        dummy = SX.sym("dummy_input", 0)
        L_x_fun = Function(
            "L_x_fun", [dummy], [jacobian(y, gnsf["x"][range(gnsf["nx1"])])]
        )
        L_xdot_fun = Function(
            "L_xdot_fun", [dummy], [jacobian(y, gnsf["xdot"][range(gnsf["nx1"])])]
        )
        L_z_fun = Function(
            "L_z_fun", [dummy], [jacobian(y, gnsf["z"][range(gnsf["nz1"])])]
        )
        L_u_fun = Function("L_u_fun", [dummy], [jacobian(uhat, gnsf["u"])])

        gnsf["L_x"] = L_x_fun(0).full()
        gnsf["L_xdot"] = L_xdot_fun(0).full()
        gnsf["L_u"] = L_u_fun(0).full()
        gnsf["L_z"] = L_z_fun(0).full()
    gnsf["y"] = y
    gnsf["uhat"] = uhat

    gnsf["ny"] = casadi_length(y)
    gnsf["nuhat"] = casadi_length(uhat)
    gnsf["n_out"] = casadi_length(gnsf["phi_expr"])

    return gnsf
