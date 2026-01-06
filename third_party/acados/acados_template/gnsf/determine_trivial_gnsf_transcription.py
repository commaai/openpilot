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

from casadi import *
import numpy as np
from ..utils import casadi_length, idx_perm_to_ipiv
from .determine_input_nonlinearity_function import determine_input_nonlinearity_function
from .check_reformulation import check_reformulation


def determine_trivial_gnsf_transcription(acados_ocp, print_info):
    ## Description
    # this function takes a model of an implicit ODE/ index-1 DAE and sets up
    # an equivalent model in the GNSF structure, with empty linear output
    # system and trivial model matrices, i.e. A, B, E, c are zeros, and C is
    # eye. - no structure is exploited

    model = acados_ocp.model
    # initial print
    print("*****************************************************************")
    print(" ")
    print(f"******      Restructuring {model.name} model    ***********")
    print(" ")
    print("*****************************************************************")

    # load model
    f_impl_expr = model.f_impl_expr

    model_name_prefix = model.name

    # x
    x = model.x
    nx = acados_ocp.dims.nx
    # check type
    if isinstance(x[0], SX):
        isSX = True
    else:
        print("GNSF detection only works for SX CasADi type!!!")
        import pdb

        pdb.set_trace()
    # xdot
    xdot = model.xdot
    # u
    nu = acados_ocp.dims.nu
    if nu == 0:
        u = SX.sym("u", 0, 0)
    else:
        u = model.u

    nz = acados_ocp.dims.nz
    if nz == 0:
        z = SX.sym("z", 0, 0)
    else:
        z = model.z

    p = model.p
    nparam = acados_ocp.dims.np

    # avoid SX of size 0x1
    if casadi_length(u) == 0:
        u = SX.sym("u", 0, 0)
        nu = 0
    ## initialize gnsf struct
    # dimensions
    gnsf = {"nx": nx, "nu": nu, "nz": nz, "np": nparam}
    gnsf["nx1"] = nx
    gnsf["nx2"] = 0
    gnsf["nz1"] = nz
    gnsf["nz2"] = 0
    gnsf["nuhat"] = nu
    gnsf["ny"] = 2 * nx + nz

    gnsf["phi_expr"] = f_impl_expr
    gnsf["A"] = np.zeros((nx + nz, nx))
    gnsf["B"] = np.zeros((nx + nz, nu))
    gnsf["E"] = np.zeros((nx + nz, nx + nz))
    gnsf["c"] = np.zeros((nx + nz, 1))
    gnsf["C"] = np.eye(nx + nz)
    gnsf["name"] = model_name_prefix

    gnsf["x"] = x
    gnsf["xdot"] = xdot
    gnsf["z"] = z
    gnsf["u"] = u
    gnsf["p"] = p

    gnsf = determine_input_nonlinearity_function(gnsf)

    gnsf["A_LO"] = []
    gnsf["E_LO"] = []
    gnsf["B_LO"] = []
    gnsf["c_LO"] = []
    gnsf["f_lo_expr"] = []

    # permutation
    gnsf["idx_perm_x"] = range(nx)  # matlab-style)
    gnsf["ipiv_x"] = idx_perm_to_ipiv(gnsf["idx_perm_x"])  # blasfeo-style
    gnsf["idx_perm_z"] = range(nz)
    gnsf["ipiv_z"] = idx_perm_to_ipiv(gnsf["idx_perm_z"])
    gnsf["idx_perm_f"] = range((nx + nz))
    gnsf["ipiv_f"] = idx_perm_to_ipiv(gnsf["idx_perm_f"])

    gnsf["nontrivial_f_LO"] = 0

    check_reformulation(model, gnsf, print_info)
    if print_info:
        print(f"Success: Set up equivalent GNSF model with trivial matrices")
        print(" ")
    if print_info:
        print(
            "-----------------------------------------------------------------------------------"
        )
        print(" ")
        print(
            "reduced input ny    of phi from  ",
            str(2 * nx + nz),
            "   to  ",
            str(gnsf["ny"]),
        )
        print(
            "reduced input nuhat of phi from  ", str(nu), "   to  ", str(gnsf["nuhat"])
        )
        print(" ")
        print(
            "-----------------------------------------------------------------------------------"
        )
    return gnsf
