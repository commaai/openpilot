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

from casadi import n_nodes
import numpy as np


def structure_detection_print_summary(gnsf, acados_ocp):

    ## Description
    # this function prints the most important info after determining a GNSF
    # reformulation of the implicit model "initial_model" into "gnsf", which is
    # equivalent to the "reordered_model".
    model = acados_ocp.model
    # # GNSF
    # get dimensions
    nx = gnsf["nx"]
    nu = gnsf["nu"]
    nz = gnsf["nz"]

    nx1 = gnsf["nx1"]
    nx2 = gnsf["nx2"]

    nz1 = gnsf["nz1"]
    nz2 = gnsf["nz2"]

    # np = gnsf['np']
    n_out = gnsf["n_out"]
    ny = gnsf["ny"]
    nuhat = gnsf["nuhat"]

    #
    f_impl_expr = model.f_impl_expr
    n_nodes_initial = n_nodes(model.f_impl_expr)
    # x_old = model.x
    # f_impl_old = model.f_impl_expr

    x = gnsf["x"]
    z = gnsf["z"]

    phi_current = gnsf["phi_expr"]

    ## PRINT SUMMARY -- STRUCHTRE DETECTION
    print(" ")
    print(
        "*********************************************************************************************"
    )
    print(" ")
    print(
        "******************        SUCCESS: GNSF STRUCTURE DETECTION COMPLETE !!!      ***************"
    )
    print(" ")
    print(
        "*********************************************************************************************"
    )
    print(" ")
    print(
        f"========================= STRUCTURE DETECTION SUMMARY ===================================="
    )
    print(" ")
    print("-------- Nonlinear Static Feedback type system --------")
    print(" ")
    print(" successfully transcribed dynamic system model into GNSF structure ")
    print(" ")
    print(
        "reduced dimension of nonlinearity phi from        ",
        str(nx + nz),
        " to ",
        str(gnsf["n_out"]),
    )
    print(" ")
    print(
        "reduced input dimension of nonlinearity phi from  ",
        2 * nx + nz + nu,
        " to ",
        gnsf["ny"] + gnsf["nuhat"],
    )
    print(" ")
    print(f"reduced number of nodes in CasADi expression of nonlinearity phi from  {n_nodes_initial}  to  {n_nodes(phi_current)}\n")
    print("----------- Linear Output System (LOS) ---------------")
    if nx2 + nz2 > 0:
        print(" ")
        print(f"introduced Linear Output System of size           ", str(nx2 + nz2))
        print(" ")
        if nx2 > 0:
            print("consisting of the states:")
            print(" ")
            print(x[range(nx1, nx)])
            print(" ")
        if nz2 > 0:
            print("and algebraic variables:")
            print(" ")
            print(z[range(nz1, nz)])
            print(" ")
        if gnsf["purely_linear"] == 1:
            print(" ")
            print("Model is fully linear!")
            print(" ")
    if not all(gnsf["idx_perm_x"] == np.array(range(nx))):
        print(" ")
        print(
            "--------------------------------------------------------------------------------------------------"
        )
        print(
            "NOTE: permuted differential state vector x, such that x_gnsf = x(idx_perm_x) with idx_perm_x ="
        )
        print(" ")
        print(gnsf["idx_perm_x"])
    if nz != 0 and not all(gnsf["idx_perm_z"] == np.array(range(nz))):
        print(" ")
        print(
            "--------------------------------------------------------------------------------------------------"
        )
        print(
            "NOTE: permuted algebraic state vector z, such that z_gnsf = z(idx_perm_z) with idx_perm_z ="
        )
        print(" ")
        print(gnsf["idx_perm_z"])
    if not all(gnsf["idx_perm_f"] == np.array(range(nx + nz))):
        print(" ")
        print(
            "--------------------------------------------------------------------------------------------------"
        )
        print(
            "NOTE: permuted rhs expression vector f, such that f_gnsf = f(idx_perm_f) with idx_perm_f ="
        )
        print(" ")
        print(gnsf["idx_perm_f"])
    ## print GNSF dimensions
    print(
        "--------------------------------------------------------------------------------------------------------"
    )
    print(" ")
    print("The dimensions of the GNSF reformulated model read as:")
    print(" ")
    # T_dim = table(nx, nu, nz, np, nx1, nz1, n_out, ny, nuhat)
    # print( T_dim )
    print(f"nx    ", {nx})
    print(f"nu    ", {nu})
    print(f"nz    ", {nz})
    # print(f"np    ", {np})
    print(f"nx1   ", {nx1})
    print(f"nz1   ", {nz1})
    print(f"n_out ", {n_out})
    print(f"ny    ", {ny})
    print(f"nuhat ", {nuhat})
