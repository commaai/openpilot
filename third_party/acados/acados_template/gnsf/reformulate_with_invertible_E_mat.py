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
from .determine_input_nonlinearity_function import determine_input_nonlinearity_function
from .check_reformulation import check_reformulation


def reformulate_with_invertible_E_mat(gnsf, model, print_info):
    ## Description
    # this function checks that the necessary condition to apply the gnsf
    # structure exploiting integrator to a model, namely that the matrices E11,
    # E22 are invertible holds.
    # if this is not the case, it will make these matrices invertible and add:
    # corresponding terms, to the term C * phi, such that the obtained model is
    # still equivalent

    # check invertibility of E11, E22 and reformulate if needed:
    ind_11 = range(gnsf["nx1"])
    ind_22 = range(gnsf["nx1"], gnsf["nx1"] + gnsf["nz1"])

    if print_info:
        print(" ")
        print("----------------------------------------------------")
        print("checking rank of E11 and E22")
        print("----------------------------------------------------")
    ## check if E11, E22 are invertible:
    z_check = False
    if gnsf["nz1"] > 0:
        z_check = (
            np.linalg.matrix_rank(gnsf["E"][np.ix_(ind_22, ind_22)]) != gnsf["nz1"]
        )

    if (
        np.linalg.matrix_rank(gnsf["E"][np.ix_(ind_11, ind_11)]) != gnsf["nx1"]
        or z_check
    ):
        # print warning (always)
        print(f"the rank of E11 or E22 is not full after the reformulation")
        print("")
        print(
            f"the script will try to reformulate the model with an invertible matrix instead"
        )
        print(
            f"NOTE: this feature is based on a heuristic, it should be used with care!!!"
        )

        ## load models
        xdot = gnsf["xdot"]
        z = gnsf["z"]

        # # GNSF
        # get dimensions
        nx1 = gnsf["nx1"]
        x1dot = xdot[range(nx1)]

        k = vertcat(x1dot, z)
        for i in [1, 2]:
            if i == 1:
                ind = range(gnsf["nx1"])
            else:
                ind = range(gnsf["nx1"], gnsf["nx1"] + gnsf["nz1"])
            mat = gnsf["E"][np.ix_(ind, ind)]
            import pdb

            pdb.set_trace()
            while np.linalg.matrix_rank(mat) < len(ind):
                # import pdb; pdb.set_trace()
                if print_info:
                    print(" ")
                    print(f"the rank of E", str(i), str(i), " is not full")
                    print(
                        f"the algorithm will try to reformulate the model with an invertible matrix instead"
                    )
                    print(
                        f"NOTE: this feature is not super stable and might need more testing!!!!!!"
                    )
                for sub_max in ind:
                    sub_ind = range(min(ind), sub_max)
                    # regard the submatrix mat(sub_ind, sub_ind)
                    sub_mat = gnsf["E"][sub_ind, sub_ind]
                    if np.linalg.matrix_rank(sub_mat) < len(sub_ind):
                        # reformulate the model by adding a 1 to last diagonal
                        # element and changing rhs respectively.
                        gnsf["E"][sub_max, sub_max] = gnsf["E"][sub_max, sub_max] + 1
                        # this means adding the term 1 * k(sub_max) to the sub_max
                        # row of the l.h.s
                        if len(np.nonzero(gnsf["C"][sub_max, :])[0]) == 0:
                            # if isempty(find(gnsf['C'](sub_max,:), 1)):
                            # add new nonlinearity entry
                            gnsf["C"][sub_max, gnsf["n_out"] + 1] = 1
                            gnsf["phi_expr"] = vertcat(gnsf["phi_expr"], k[sub_max])
                        else:
                            ind_f = np.nonzero(gnsf["C"][sub_max, :])[0]
                            if len(ind_f) != 1:
                                raise Exception("C is assumed to be a selection matrix")
                            else:
                                ind_f = ind_f[0]
                            # add term to corresponding nonlinearity entry
                            # note: herbey we assume that C is a selection matrix,
                            # i.e. gnsf['phi_expr'](ind_f) is only entering one equation

                            gnsf["phi_expr"][ind_f] = (
                                gnsf["phi_expr"][ind_f]
                                + k[sub_max] / gnsf["C"][sub_max, ind_f]
                            )
                            gnsf = determine_input_nonlinearity_function(gnsf)
                            check_reformulation(model, gnsf, print_info)
        print("successfully reformulated the model with invertible matrices E11, E22")
    else:
        if print_info:
            print(" ")
            print(
                "the rank of both E11 and E22 is naturally full after the reformulation "
            )
            print("==>  model reformulation finished")
            print(" ")
    if (gnsf['nx2'] > 0 or gnsf['nz2'] > 0) and det(gnsf["E_LO"]) == 0:
        print(
            "_______________________________________________________________________________________________________"
        )
        print(" ")
        print("TAKE CARE ")
        print("E_LO matrix is NOT regular after automatic transcription!")
        print("->> this means the model CANNOT be used with the gnsf integrator")
        print(
            "->> it probably means that one entry (of xdot or z) that was moved to the linear output type system"
        )
        print("    does not appear in the model at all (zero column in E_LO)")
        print(" OR: the columns of E_LO are linearly dependent ")
        print(" ")
        print(
            " SOLUTIONs: a) go through your model & check equations the method wanted to move to LOS"
        )
        print("            b) deactivate the detect_LOS option")
        print(
            "_______________________________________________________________________________________________________"
        )
    return gnsf
