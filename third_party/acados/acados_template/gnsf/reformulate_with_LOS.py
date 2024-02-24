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

from .determine_input_nonlinearity_function import determine_input_nonlinearity_function
from .check_reformulation import check_reformulation
from casadi import *
from ..utils import casadi_length, idx_perm_to_ipiv, is_empty


def reformulate_with_LOS(acados_ocp, gnsf, print_info):

    ## Description:
    # This function takes an intitial transcription of the implicit ODE model
    # "model" into "gnsf" and reformulates "gnsf" with a linear output system
    # (LOS), containing as many states of the model as possible.
    # Therefore it might be that the state vector and the implicit function
    # vector have to be reordered. This reordered model is part of the output,
    # namely reordered_model.

    ## import CasADi and load models
    model = acados_ocp.model

    # symbolics
    x = gnsf["x"]
    xdot = gnsf["xdot"]
    u = gnsf["u"]
    z = gnsf["z"]

    # dimensions
    nx = gnsf["nx"]
    nz = gnsf["nz"]

    # get model matrices
    A = gnsf["A"]
    B = gnsf["B"]
    C = gnsf["C"]
    E = gnsf["E"]
    c = gnsf["c"]

    A_LO = gnsf["A_LO"]

    y = gnsf["y"]

    phi_old = gnsf["phi_expr"]

    if print_info:
        print(" ")
        print("=================================================================")
        print(" ")
        print("================    Detect Linear Output System   ===============")
        print(" ")
        print("=================================================================")
        print(" ")
    ## build initial I_x1 and I_x2_candidates
    # I_xrange( all components of x for which either xii or xdot_ii enters y):
    # I_LOS_candidates: the remaining components

    I_nsf_components = set()
    I_LOS_candidates = set()

    if gnsf["ny"] > 0:
        for ii in range(nx):
            if which_depends(y, x[ii])[0] or which_depends(y, xdot[ii])[0]:
                # i.e. xii or xiidot are part of y, and enter phi_expr
                if print_info:
                    print(f"x_{ii} is part of x1")
                I_nsf_components = set.union(I_nsf_components, set([ii]))
            else:
                # i.e. neither xii nor xiidot are part of y, i.e. enter phi_expr
                I_LOS_candidates = set.union(I_LOS_candidates, set([ii]))
                if print_info:
                    print(" ")
        for ii in range(nz):
            if which_depends(y, z[ii])[0]:
                # i.e. xii or xiidot are part of y, and enter phi_expr
                if print_info:
                    print(f"z_{ii} is part of x1")
                I_nsf_components = set.union(I_nsf_components, set([ii + nx]))
            else:
                # i.e. neither xii nor xiidot are part of y, i.e. enter phi_expr
                I_LOS_candidates = set.union(I_LOS_candidates, set([ii + nx]))
    else:
        I_LOS_candidates = set(range((nx + nz)))
    if print_info:
        print(" ")
        print(f"I_LOS_candidates {I_LOS_candidates}")
    new_nsf_components = I_nsf_components
    I_nsf_eq = set([])
    unsorted_dyn = set(range(nx + nz))
    xdot_z = vertcat(xdot, z)

    ## determine components of Linear Output System
    # determine maximal index set I_x2
    # such that the components x(I_x2) can be written as a LOS
    Eq_map = []
    while True:
        ## find equations corresponding to new_nsf_components
        for ii in new_nsf_components:
            current_var = xdot_z[ii]
            var_name = current_var.name

            # print( unsorted_dyn)
            # print("np.nonzero(E[:,ii])[0]",np.nonzero(E[:,ii])[0])
            I_eq = set.intersection(set(np.nonzero(E[:, ii])[0]), unsorted_dyn)
            if len(I_eq) == 1:
                i_eq = I_eq.pop()
                if print_info:
                    print(f"component {i_eq} is associated with state {ii}")
            elif len(I_eq) > 1:  # x_ii_dot occurs in more than 1 eq linearly
                # find the equation with least linear dependencies on
                # I_LOS_cancidates
                number_of_eq = 0
                candidate_dependencies = np.zeros(len(I_eq), 1)
                I_x2_candidates = set.intersection(I_LOS_candidates, set(range(nx)))
                for eq in I_eq:
                    depending_candidates = set.union(
                        np.nonzero(E[eq, I_LOS_candidates])[0],
                        np.nonzero(A[eq, I_x2_candidates])[0],
                    )
                    candidate_dependencies[number_of_eq] = +len(depending_candidates)
                    number_of_eq += 1
                    number_of_eq = np.argmin(candidate_dependencies)
                i_eq = I_eq[number_of_eq]
            else:  ## x_ii_dot does not occur linearly in any of the unsorted dynamics
                for j in unsorted_dyn:
                    phi_eq_j = gnsf["phi_expr"][np.nonzero(C[j, :])[0]]
                    if which_depends(phi_eq_j, xdot_z(ii))[0]:
                        I_eq = set.union(I_eq, j)
                if is_empty(I_eq):
                    I_eq = unsorted_dyn
                # find the equation with least linear dependencies on I_LOS_cancidates
                number_of_eq = 0
                candidate_dependencies = np.zeros(len(I_eq), 1)
                I_x2_candidates = set.intersection(I_LOS_candidates, set(range(nx)))
                for eq in I_eq:
                    depending_candidates = set.union(
                        np.nonzero(E[eq, I_LOS_candidates])[0],
                        np.nonzero(A[eq, I_x2_candidates])[0],
                    )
                    candidate_dependencies[number_of_eq] = +len(depending_candidates)
                    number_of_eq += 1
                    number_of_eq = np.argmin(candidate_dependencies)
                i_eq = I_eq[number_of_eq]
                ## add 1 * [xdot,z](ii) to both sides of i_eq
                if print_info:
                    print(
                        "adding 1 * ",
                        var_name,
                        " to both sides of equation ",
                        i_eq,
                        ".",
                    )
                gnsf["E"][i_eq, ii] = 1
                i_phi = np.nonzero(gnsf["C"][i_eq, :])
                if is_empty(i_phi):
                    i_phi = len(gnsf["phi_expr"]) + 1
                    gnsf["C"][i_eq, i_phi] = 1  # add column to C with 1 entry
                    gnsf["phi_expr"] = vertcat(gnsf["phi_expr"], 0)
                    gnsf["phi_expr"][i_phi] = (
                        gnsf["phi_expr"](i_phi)
                        + gnsf["E"][i_eq, ii] / gnsf["C"][i_eq, i_phi] * xdot_z[ii]
                    )
                if print_info:
                    print(
                        "detected equation ",
                        i_eq,
                        " to correspond to variable ",
                        var_name,
                    )
            I_nsf_eq = set.union(I_nsf_eq, {i_eq})
            # remove i_eq from unsorted_dyn
            unsorted_dyn.remove(i_eq)
            Eq_map.append([ii, i_eq])

        ## add components to I_x1
        for eq in I_nsf_eq:
            I_linear_dependence = set.union(
                set(np.nonzero(A[eq, :])[0]), set(np.nonzero(E[eq, :])[0])
            )
            I_nsf_components = set.union(I_linear_dependence, I_nsf_components)
            # I_nsf_components = I_nsf_components[:]

        new_nsf_components = set.intersection(I_LOS_candidates, I_nsf_components)
        if is_empty(new_nsf_components):
            if print_info:
                print("new_nsf_components is empty")
            break
        # remove new_nsf_components from candidates
        I_LOS_candidates = set.difference(I_LOS_candidates, new_nsf_components)
    if not is_empty(Eq_map):
        # [~, new_eq_order] = sort(Eq_map(1,:))
        # I_nsf_eq = Eq_map(2, new_eq_order )
        for count, m in enumerate(Eq_map):
            m.append(count)
        sorted(Eq_map, key=lambda x: x[1])
        new_eq_order = [m[2] for m in Eq_map]
        Eq_map = [Eq_map[i] for i in new_eq_order]
        I_nsf_eq = [m[1] for m in Eq_map]

    else:
        I_nsf_eq = []

    I_LOS_components = I_LOS_candidates
    I_LOS_eq = sorted(set.difference(set(range(nx + nz)), I_nsf_eq))
    I_nsf_eq = sorted(I_nsf_eq)

    I_x1 = set.intersection(I_nsf_components, set(range(nx)))
    I_z1 = set.intersection(I_nsf_components, set(range(nx, nx + nz)))
    I_z1 = set([i - nx for i in I_z1])

    I_x2 = set.intersection(I_LOS_components, set(range(nx)))
    I_z2 = set.intersection(I_LOS_components, set(range(nx, nx + nz)))
    I_z2 = set([i - nx for i in I_z2])

    if print_info:
        print(f"I_x1 {I_x1}, I_x2 {I_x2}")

    ## permute x, xdot
    if is_empty(I_x1):
        x1 = []
        x1dot = []
    else:
        x1 = x[list(I_x1)]
        x1dot = xdot[list(I_x1)]
    if is_empty(I_x2):
        x2 = []
        x2dot = []
    else:
        x2 = x[list(I_x2)]
        x2dot = xdot[list(I_x2)]
    if is_empty(I_z1):
        z1 = []
    else:
        z1 = z(I_z1)
    if is_empty(I_z2):
        z2 = []
    else:
        z2 = z[list(I_z2)]

    I_x1 = sorted(I_x1)
    I_x2 = sorted(I_x2)
    I_z1 = sorted(I_z1)
    I_z2 = sorted(I_z2)
    gnsf["xdot"] = vertcat(x1dot, x2dot)
    gnsf["x"] = vertcat(x1, x2)
    gnsf["z"] = vertcat(z1, z2)
    gnsf["nx1"] = len(I_x1)
    gnsf["nx2"] = len(I_x2)
    gnsf["nz1"] = len(I_z1)
    gnsf["nz2"] = len(I_z2)

    # store permutations
    gnsf["idx_perm_x"] = I_x1 + I_x2
    gnsf["ipiv_x"] = idx_perm_to_ipiv(gnsf["idx_perm_x"])
    gnsf["idx_perm_z"] = I_z1 + I_z2
    gnsf["ipiv_z"] = idx_perm_to_ipiv(gnsf["idx_perm_z"])
    gnsf["idx_perm_f"] = I_nsf_eq + I_LOS_eq
    gnsf["ipiv_f"] = idx_perm_to_ipiv(gnsf["idx_perm_f"])

    f_LO = SX.sym("f_LO", 0, 0)

    ## rewrite I_LOS_eq as LOS
    if gnsf["n_out"] == 0:
        C_phi = np.zeros(gnsf["nx"] + gnsf["nz"], 1)
    else:
        C_phi = C @ phi_old
    if gnsf["nx1"] == 0:
        Ax1 = np.zeros(gnsf["nx"] + gnsf["nz"], 1)
    else:
        Ax1 = A[:, sorted(I_x1)] @ x1
    if gnsf["nx1"] + gnsf["nz1"] == 0:
        lhs_nsf = np.zeros(gnsf["nx"] + gnsf["nz"], 1)
    else:
        lhs_nsf = E[:, sorted(I_nsf_components)] @ vertcat(x1, z1)
    n_LO = len(I_LOS_eq)
    B_LO = np.zeros((n_LO, gnsf["nu"]))
    A_LO = np.zeros((gnsf["nx2"] + gnsf["nz2"], gnsf["nx2"]))
    E_LO = np.zeros((n_LO, n_LO))
    c_LO = np.zeros((n_LO, 1))

    I_LOS_eq = list(I_LOS_eq)
    for eq in I_LOS_eq:
        i_LO = I_LOS_eq.index(eq)
        f_LO = vertcat(f_LO, Ax1[eq] + C_phi[eq] - lhs_nsf[eq])
        print(f"eq {eq} I_LOS_components {I_LOS_components}, i_LO {i_LO}, f_LO {f_LO}")
        E_LO[i_LO, :] = E[eq, sorted(I_LOS_components)]
        A_LO[i_LO, :] = A[eq, I_x2]
        c_LO[i_LO, :] = c[eq]
        B_LO[i_LO, :] = B[eq, :]
    if casadi_length(f_LO) == 0:
        f_LO = SX.zeros((gnsf["nx2"] + gnsf["nz2"], 1))
    f_LO = simplify(f_LO)
    gnsf["A_LO"] = A_LO
    gnsf["E_LO"] = E_LO
    gnsf["B_LO"] = B_LO
    gnsf["c_LO"] = c_LO
    gnsf["f_lo_expr"] = f_LO

    ## remove I_LOS_eq from NSF type system
    gnsf["A"] = gnsf["A"][np.ix_(sorted(I_nsf_eq), sorted(I_x1))]
    gnsf["B"] = gnsf["B"][sorted(I_nsf_eq), :]
    gnsf["C"] = gnsf["C"][sorted(I_nsf_eq), :]
    gnsf["E"] = gnsf["E"][np.ix_(sorted(I_nsf_eq), sorted(I_nsf_components))]
    gnsf["c"] = gnsf["c"][sorted(I_nsf_eq), :]

    ## reduce phi, C
    I_nonzero = []
    for ii in range(gnsf["C"].shape[1]):  # n_colums of C:
        print(f"ii {ii}")
        if not all(gnsf["C"][:, ii] == 0):  # if column ~= 0
            I_nonzero.append(ii)
    gnsf["C"] = gnsf["C"][:, I_nonzero]
    gnsf["phi_expr"] = gnsf["phi_expr"][I_nonzero]

    gnsf = determine_input_nonlinearity_function(gnsf)

    check_reformulation(model, gnsf, print_info)

    gnsf["nontrivial_f_LO"] = 0
    if not is_empty(gnsf["f_lo_expr"]):
        for ii in range(casadi_length(gnsf["f_lo_expr"])):
            fii = gnsf["f_lo_expr"][ii]
            if not fii.is_zero():
                gnsf["nontrivial_f_LO"] = 1
            if not gnsf["nontrivial_f_LO"] and print_info:
                print("f_LO is fully trivial (== 0)")
    check_reformulation(model, gnsf, print_info)

    if print_info:
        print("")
        print(
            "---------------------------------------------------------------------------------"
        )
        print(
            "------------- Success: Linear Output System (LOS) detected ----------------------"
        )
        print(
            "---------------------------------------------------------------------------------"
        )
        print("")
        print(
            "==>>  moved  ",
            gnsf["nx2"],
            "differential states and ",
            gnsf["nz2"],
            " algebraic variables to the Linear Output System",
        )
        print(
            "==>>  recuced output dimension of phi from  ",
            casadi_length(phi_old),
            " to ",
            casadi_length(gnsf["phi_expr"]),
        )
        print(" ")
        print("Matrices defining the LOS read as")
        print(" ")
        print("E_LO =")
        print(gnsf["E_LO"])
        print("A_LO =")
        print(gnsf["A_LO"])
        print("B_LO =")
        print(gnsf["B_LO"])
        print("c_LO =")
        print(gnsf["c_LO"])

    return gnsf
