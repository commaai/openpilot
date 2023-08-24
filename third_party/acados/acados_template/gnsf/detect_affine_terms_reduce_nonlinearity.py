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
from .check_reformulation import check_reformulation
from .determine_input_nonlinearity_function import determine_input_nonlinearity_function
from ..utils import casadi_length, print_casadi_expression


def detect_affine_terms_reduce_nonlinearity(gnsf, acados_ocp, print_info):

    ## Description
    # this function takes a gnsf structure with trivial model matrices (A, B,
    # E, c are zeros, and C is eye).
    # It detects all affine linear terms and sets up an equivalent model in the
    # GNSF structure, where all affine linear terms are modeled through the
    # matrices A, B, E, c and the linear output system (LOS) is empty.
    # NOTE: model is just taken as an argument to check equivalence of the
    # models within the function.

    model = acados_ocp.model
    if print_info:
        print(" ")
        print("====================================================================")
        print(" ")
        print("============  Detect affine-linear dependencies   ==================")
        print(" ")
        print("====================================================================")
        print(" ")
    # symbolics
    x = gnsf["x"]
    xdot = gnsf["xdot"]
    u = gnsf["u"]
    z = gnsf["z"]

    # dimensions
    nx = gnsf["nx"]
    nu = gnsf["nu"]
    nz = gnsf["nz"]

    ny_old = gnsf["ny"]
    nuhat_old = gnsf["nuhat"]

    ## Represent all affine dependencies through the model matrices A, B, E, c
    ## determine A
    n_nodes_current = n_nodes(gnsf["phi_expr"])

    for ii in range(casadi_length(gnsf["phi_expr"])):
        fii = gnsf["phi_expr"][ii]
        for ix in range(nx):
            var = x[ix]
            varname = var.name
            # symbolic jacobian of fii w.r.t. xi
            jac_fii_xi = jacobian(fii, var)
            if jac_fii_xi.is_constant():
                # jacobian value
                jac_fii_xi_fun = Function("jac_fii_xi_fun", [x[1]], [jac_fii_xi])
                # x[1] as input just to have a scalar input and call the function as follows:
                gnsf["A"][ii, ix] = jac_fii_xi_fun(0).full()
            else:
                gnsf["A"][ii, ix] = 0
                if print_info:
                    print(
                        "phi(",
                        str(ii),
                        ") is nonlinear in x(",
                        str(ix),
                        ") = ",
                        varname,
                    )
                    print(fii)
                    print("-----------------------------------------------------")
    f_next = gnsf["phi_expr"] - gnsf["A"] @ x
    f_next = simplify(f_next)
    n_nodes_next = n_nodes(f_next)

    if print_info:
        print("\n")
        print(f"determined matrix A:")
        print(gnsf["A"])
        print(f"reduced nonlinearity from  {n_nodes_current} to {n_nodes_next} nodes")
    # assert(n_nodes_current >= n_nodes_next,'n_nodes_current >= n_nodes_next FAILED')
    gnsf["phi_expr"] = f_next

    check_reformulation(model, gnsf, print_info)

    ## determine B
    n_nodes_current = n_nodes(gnsf["phi_expr"])

    for ii in range(casadi_length(gnsf["phi_expr"])):
        fii = gnsf["phi_expr"][ii]
        for iu in range(nu):
            var = u[iu]
            varname = var.name
            # symbolic jacobian of fii w.r.t. ui
            jac_fii_ui = jacobian(fii, var)
            if jac_fii_ui.is_constant():  # i.e. hessian is structural zero:
                # jacobian value
                jac_fii_ui_fun = Function("jac_fii_ui_fun", [x[1]], [jac_fii_ui])
                gnsf["B"][ii, iu] = jac_fii_ui_fun(0).full()
            else:
                gnsf["B"][ii, iu] = 0
                if print_info:
                    print(f"phi({ii}) is nonlinear in u(", str(iu), ") = ", varname)
                    print(fii)
                    print("-----------------------------------------------------")
    f_next = gnsf["phi_expr"] - gnsf["B"] @ u
    f_next = simplify(f_next)
    n_nodes_next = n_nodes(f_next)

    if print_info:
        print("\n")
        print(f"determined matrix B:")
        print(gnsf["B"])
        print(f"reduced nonlinearity from  {n_nodes_current} to {n_nodes_next} nodes")

    gnsf["phi_expr"] = f_next

    check_reformulation(model, gnsf, print_info)

    ## determine E
    n_nodes_current = n_nodes(gnsf["phi_expr"])
    k = vertcat(xdot, z)

    for ii in range(casadi_length(gnsf["phi_expr"])):
        fii = gnsf["phi_expr"][ii]
        for ik in range(casadi_length(k)):
            # symbolic jacobian of fii w.r.t. ui
            var = k[ik]
            varname = var.name
            jac_fii_ki = jacobian(fii, var)
            if jac_fii_ki.is_constant():
                # jacobian value
                jac_fii_ki_fun = Function("jac_fii_ki_fun", [x[1]], [jac_fii_ki])
                gnsf["E"][ii, ik] = -jac_fii_ki_fun(0).full()
            else:
                gnsf["E"][ii, ik] = 0
                if print_info:
                    print(f"phi( {ii}) is nonlinear in xdot_z({ik}) = ", varname)
                    print(fii)
                    print("-----------------------------------------------------")
    f_next = gnsf["phi_expr"] + gnsf["E"] @ k
    f_next = simplify(f_next)
    n_nodes_next = n_nodes(f_next)

    if print_info:
        print("\n")
        print(f"determined matrix E:")
        print(gnsf["E"])
        print(f"reduced nonlinearity from {n_nodes_current} to {n_nodes_next} nodes")

    gnsf["phi_expr"] = f_next
    check_reformulation(model, gnsf, print_info)

    ## determine constant term c

    n_nodes_current = n_nodes(gnsf["phi_expr"])
    for ii in range(casadi_length(gnsf["phi_expr"])):
        fii = gnsf["phi_expr"][ii]
        if fii.is_constant():
            # function value goes into c
            fii_fun = Function("fii_fun", [x[1]], [fii])
            gnsf["c"][ii] = fii_fun(0).full()
        else:
            gnsf["c"][ii] = 0
            if print_info:
                print(f"phi(", str(ii), ") is NOT constant")
                print(fii)
                print("-----------------------------------------------------")
    gnsf["phi_expr"] = gnsf["phi_expr"] - gnsf["c"]
    gnsf["phi_expr"] = simplify(gnsf["phi_expr"])
    n_nodes_next = n_nodes(gnsf["phi_expr"])

    if print_info:
        print("\n")
        print(f"determined vector c:")
        print(gnsf["c"])
        print(f"reduced nonlinearity from {n_nodes_current} to {n_nodes_next} nodes")

    check_reformulation(model, gnsf, print_info)

    ## determine nonlinearity & corresponding matrix C
    ## Reduce dimension of phi
    n_nodes_current = n_nodes(gnsf["phi_expr"])
    ind_non_zero = []
    for ii in range(casadi_length(gnsf["phi_expr"])):
        fii = gnsf["phi_expr"][ii]
        fii = simplify(fii)
        if not fii.is_zero():
            ind_non_zero = list(set.union(set(ind_non_zero), set([ii])))
    gnsf["phi_expr"] = gnsf["phi_expr"][ind_non_zero]

    # C
    gnsf["C"] = np.zeros((nx + nz, len(ind_non_zero)))
    for ii in range(len(ind_non_zero)):
        gnsf["C"][ind_non_zero[ii], ii] = 1
    gnsf = determine_input_nonlinearity_function(gnsf)
    n_nodes_next = n_nodes(gnsf["phi_expr"])

    if print_info:
        print(" ")
        print("determined matrix C:")
        print(gnsf["C"])
        print(
            "---------------------------------------------------------------------------------"
        )
        print(
            "------------- Success: Affine linear terms detected -----------------------------"
        )
        print(
            "---------------------------------------------------------------------------------"
        )
        print(
            f'reduced nonlinearity dimension n_out from  {nx+nz}  to  {gnsf["n_out"]}'
        )
        print(f"reduced nonlinearity from  {n_nodes_current} to {n_nodes_next} nodes")
        print(" ")
        print("phi now reads as:")
        print_casadi_expression(gnsf["phi_expr"])

    ## determine input of nonlinearity function
    check_reformulation(model, gnsf, print_info)

    gnsf["ny"] = casadi_length(gnsf["y"])
    gnsf["nuhat"] = casadi_length(gnsf["uhat"])

    if print_info:
        print(
            "-----------------------------------------------------------------------------------"
        )
        print(" ")
        print(
            f"reduced input ny    of phi from  ",
            str(ny_old),
            "   to  ",
            str(gnsf["ny"]),
        )
        print(
            f"reduced input nuhat of phi from  ",
            str(nuhat_old),
            "   to  ",
            str(gnsf["nuhat"]),
        )
        print(
            "-----------------------------------------------------------------------------------"
        )

    # if print_info:
    #     print(f"gnsf: {gnsf}")

    return gnsf
