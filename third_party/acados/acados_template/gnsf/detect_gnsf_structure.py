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

from casadi import Function, jacobian, SX, vertcat, horzcat

from .determine_trivial_gnsf_transcription import determine_trivial_gnsf_transcription
from .detect_affine_terms_reduce_nonlinearity import (
    detect_affine_terms_reduce_nonlinearity,
)
from .reformulate_with_LOS import reformulate_with_LOS
from .reformulate_with_invertible_E_mat import reformulate_with_invertible_E_mat
from .structure_detection_print_summary import structure_detection_print_summary
from .check_reformulation import check_reformulation


def detect_gnsf_structure(acados_ocp, transcribe_opts=None):

    ## Description
    # This function takes a CasADi implicit ODE or index-1 DAE model "model"
    # consisting of a CasADi expression f_impl in the symbolic CasADi
    # variables x, xdot, u, z, (and possibly parameters p), which are also part
    # of the model, as well as a model name.
    # It will create a struct "gnsf" containing all information needed to use
    # it with the gnsf integrator in acados.
    # Additionally it will create the struct "reordered_model" which contains
    # the permuted state vector and permuted f_impl, in which additionally some
    # functions, which were made part of the linear output system of the gnsf,
    # have changed signs.

    # Options: transcribe_opts is a Matlab struct consisting of booleans:
    #   print_info: if extensive information on how the model is processed
    #       is printed to the console.
    #   generate_gnsf_model: if the neccessary C functions to simulate the gnsf
    #       model with the acados implementation of the GNSF exploiting
    #       integrator should be generated.
    #   generate_gnsf_model: if the neccessary C functions to simulate the
    #       reordered model with the acados implementation of the IRK
    #       integrator should be generated.
    #   check_E_invertibility: if the transcription method should check if the
    #       assumption that the main blocks of the matrix gnsf.E are invertible
    #       holds. If not, the method will try to reformulate the gnsf model
    #       with a different model, such that the assumption holds.

    # acados_root_dir = getenv('ACADOS_INSTALL_DIR')

    ## load transcribe_opts
    if transcribe_opts is None:
        print("WARNING: GNSF structure detection called without transcribe_opts")
        print(" using default settings")
        print("")
        transcribe_opts = dict()

    if "print_info" in transcribe_opts:
        print_info = transcribe_opts["print_info"]
    else:
        print_info = 1
        print("print_info option was not set - default is true")

    if "detect_LOS" in transcribe_opts:
        detect_LOS = transcribe_opts["detect_LOS"]
    else:
        detect_LOS = 1
        if print_info:
            print("detect_LOS option was not set - default is true")

    if "check_E_invertibility" in transcribe_opts:
        check_E_invertibility = transcribe_opts["check_E_invertibility"]
    else:
        check_E_invertibility = 1
        if print_info:
            print("check_E_invertibility option was not set - default is true")

    ## Reformulate implicit index-1 DAE into GNSF form
    # (Generalized nonlinear static feedback)
    gnsf = determine_trivial_gnsf_transcription(acados_ocp, print_info)
    gnsf = detect_affine_terms_reduce_nonlinearity(gnsf, acados_ocp, print_info)

    if detect_LOS:
        gnsf = reformulate_with_LOS(acados_ocp, gnsf, print_info)

    if check_E_invertibility:
        gnsf = reformulate_with_invertible_E_mat(gnsf, acados_ocp, print_info)

    # detect purely linear model
    if gnsf["nx1"] == 0 and gnsf["nz1"] == 0 and gnsf["nontrivial_f_LO"] == 0:
        gnsf["purely_linear"] = 1
    else:
        gnsf["purely_linear"] = 0

    structure_detection_print_summary(gnsf, acados_ocp)
    check_reformulation(acados_ocp.model, gnsf, print_info)

    ## copy relevant fields from gnsf to model
    acados_ocp.model.get_matrices_fun = Function()
    dummy = acados_ocp.model.x[0]
    model_name = acados_ocp.model.name

    get_matrices_fun = Function(
        f"{model_name}_gnsf_get_matrices_fun",
        [dummy],
        [
            gnsf["A"],
            gnsf["B"],
            gnsf["C"],
            gnsf["E"],
            gnsf["L_x"],
            gnsf["L_xdot"],
            gnsf["L_z"],
            gnsf["L_u"],
            gnsf["A_LO"],
            gnsf["c"],
            gnsf["E_LO"],
            gnsf["B_LO"],
            gnsf["nontrivial_f_LO"],
            gnsf["purely_linear"],
            gnsf["ipiv_x"] + 1,
            gnsf["ipiv_z"] + 1,
            gnsf["c_LO"],
        ],
    )

    phi = gnsf["phi_expr"]
    y = gnsf["y"]
    uhat = gnsf["uhat"]
    p = gnsf["p"]

    jac_phi_y = jacobian(phi, y)
    jac_phi_uhat = jacobian(phi, uhat)

    phi_fun = Function(f"{model_name}_gnsf_phi_fun", [y, uhat, p], [phi])
    acados_ocp.model.phi_fun = phi_fun
    acados_ocp.model.phi_fun_jac_y = Function(
        f"{model_name}_gnsf_phi_fun_jac_y", [y, uhat, p], [phi, jac_phi_y]
    )
    acados_ocp.model.phi_jac_y_uhat = Function(
        f"{model_name}_gnsf_phi_jac_y_uhat", [y, uhat, p], [jac_phi_y, jac_phi_uhat]
    )

    x1 = acados_ocp.model.x[gnsf["idx_perm_x"][: gnsf["nx1"]]]
    x1dot = acados_ocp.model.xdot[gnsf["idx_perm_x"][: gnsf["nx1"]]]
    if gnsf["nz1"] > 0:
        z1 = acados_ocp.model.z[gnsf["idx_perm_z"][: gnsf["nz1"]]]
    else:
        z1 = SX.sym("z1", 0, 0)
    f_lo = gnsf["f_lo_expr"]
    u = acados_ocp.model.u
    acados_ocp.model.f_lo_fun_jac_x1k1uz = Function(
        f"{model_name}_gnsf_f_lo_fun_jac_x1k1uz",
        [x1, x1dot, z1, u, p],
        [
            f_lo,
            horzcat(
                jacobian(f_lo, x1),
                jacobian(f_lo, x1dot),
                jacobian(f_lo, u),
                jacobian(f_lo, z1),
            ),
        ],
    )

    acados_ocp.model.get_matrices_fun = get_matrices_fun

    size_gnsf_A = gnsf["A"].shape
    acados_ocp.dims.gnsf_nx1 = size_gnsf_A[1]
    acados_ocp.dims.gnsf_nz1 = size_gnsf_A[0] - size_gnsf_A[1]
    acados_ocp.dims.gnsf_nuhat = max(phi_fun.size_in(1))
    acados_ocp.dims.gnsf_ny = max(phi_fun.size_in(0))
    acados_ocp.dims.gnsf_nout = max(phi_fun.size_out(0))

    # # dim
    # model['dim_gnsf_nx1'] = gnsf['nx1']
    # model['dim_gnsf_nx2'] = gnsf['nx2']
    # model['dim_gnsf_nz1'] = gnsf['nz1']
    # model['dim_gnsf_nz2'] = gnsf['nz2']
    # model['dim_gnsf_nuhat'] = gnsf['nuhat']
    # model['dim_gnsf_ny'] = gnsf['ny']
    # model['dim_gnsf_nout'] = gnsf['n_out']

    # # sym
    # model['sym_gnsf_y'] = gnsf['y']
    # model['sym_gnsf_uhat'] = gnsf['uhat']

    # # data
    # model['dyn_gnsf_A'] = gnsf['A']
    # model['dyn_gnsf_A_LO'] = gnsf['A_LO']
    # model['dyn_gnsf_B'] = gnsf['B']
    # model['dyn_gnsf_B_LO'] = gnsf['B_LO']
    # model['dyn_gnsf_E'] = gnsf['E']
    # model['dyn_gnsf_E_LO'] = gnsf['E_LO']
    # model['dyn_gnsf_C'] = gnsf['C']
    # model['dyn_gnsf_c'] = gnsf['c']
    # model['dyn_gnsf_c_LO'] = gnsf['c_LO']
    # model['dyn_gnsf_L_x'] = gnsf['L_x']
    # model['dyn_gnsf_L_xdot'] = gnsf['L_xdot']
    # model['dyn_gnsf_L_z'] = gnsf['L_z']
    # model['dyn_gnsf_L_u'] = gnsf['L_u']
    # model['dyn_gnsf_idx_perm_x'] = gnsf['idx_perm_x']
    # model['dyn_gnsf_ipiv_x'] = gnsf['ipiv_x']
    # model['dyn_gnsf_idx_perm_z'] = gnsf['idx_perm_z']
    # model['dyn_gnsf_ipiv_z'] = gnsf['ipiv_z']
    # model['dyn_gnsf_idx_perm_f'] = gnsf['idx_perm_f']
    # model['dyn_gnsf_ipiv_f'] = gnsf['ipiv_f']

    # # flags
    # model['dyn_gnsf_nontrivial_f_LO'] = gnsf['nontrivial_f_LO']
    # model['dyn_gnsf_purely_linear'] = gnsf['purely_linear']

    # # casadi expr
    # model['dyn_gnsf_expr_phi'] = gnsf['phi_expr']
    # model['dyn_gnsf_expr_f_lo'] = gnsf['f_lo_expr']

    return acados_ocp
