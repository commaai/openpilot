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
# cython: language_level=3
# cython: profile=False
# distutils: language=c

cimport cython
from libc cimport string

cimport acados_solver_common
# TODO: make this import more clear? it is not a general solver, but problem specific.
cimport acados_solver

cimport numpy as cnp

import os
from datetime import datetime
import numpy as np


cdef class AcadosOcpSolverCython:
    """
    Class to interact with the acados ocp solver C object.
    """

    cdef acados_solver.nlp_solver_capsule *capsule
    cdef void *nlp_opts
    cdef acados_solver_common.ocp_nlp_dims *nlp_dims
    cdef acados_solver_common.ocp_nlp_config *nlp_config
    cdef acados_solver_common.ocp_nlp_out *nlp_out
    cdef acados_solver_common.ocp_nlp_out *sens_out
    cdef acados_solver_common.ocp_nlp_in *nlp_in
    cdef acados_solver_common.ocp_nlp_solver *nlp_solver

    cdef bint solver_created

    cdef str model_name
    cdef int N

    cdef str nlp_solver_type

    def __cinit__(self, model_name, nlp_solver_type, N):

        self.solver_created = False

        self.N = N
        self.model_name = model_name
        self.nlp_solver_type = nlp_solver_type

        # create capsule
        self.capsule = acados_solver.acados_create_capsule()

        # create solver
        assert acados_solver.acados_create(self.capsule) == 0
        self.solver_created = True

        # get pointers solver
        self.__get_pointers_solver()


    def __get_pointers_solver(self):
        """
        Private function to get the pointers for solver
        """
        # get pointers solver
        self.nlp_opts = acados_solver.acados_get_nlp_opts(self.capsule)
        self.nlp_dims = acados_solver.acados_get_nlp_dims(self.capsule)
        self.nlp_config = acados_solver.acados_get_nlp_config(self.capsule)
        self.nlp_out = acados_solver.acados_get_nlp_out(self.capsule)
        self.sens_out = acados_solver.acados_get_sens_out(self.capsule)
        self.nlp_in = acados_solver.acados_get_nlp_in(self.capsule)
        self.nlp_solver = acados_solver.acados_get_nlp_solver(self.capsule)


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
        return acados_solver.acados_solve(self.capsule)


    def reset(self, reset_qp_solver_mem=1):
        """
        Sets current iterate to all zeros.
        """
        return acados_solver.acados_reset(self.capsule, reset_qp_solver_mem)


    def custom_update(self, data_):
        """
        A custom function that can be implemented by a user to be called between solver calls.
        By default this does nothing.
        The idea is to have a convenient wrapper to do complex updates of parameters and numerical data efficiently in C,
        in a function that is compiled into the solver library and can be conveniently used in the Python environment.
        """
        data_len = len(data_)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] data = np.ascontiguousarray(data_, dtype=np.float64)

        return acados_solver.acados_custom_update(self.capsule, <double *> data.data, data_len)


    def set_new_time_steps(self, new_time_steps):
        """
        Set new time steps.
        Recreates the solver if N changes.

            :param new_time_steps: 1 dimensional np array of new time steps for the solver

            .. note:: This allows for different use-cases: either set a new size of time-steps or a new distribution of
                      the shooting nodes without changing the number, e.g., to reach a different final time. Both cases
                      do not require a new code export and compilation.
        """

        raise NotImplementedError("AcadosOcpSolverCython: does not support set_new_time_steps() since it is only a prototyping feature")
        # # unlikely but still possible
        # if not self.solver_created:
        #     raise Exception('Solver was not yet created!')

        # ## check if time steps really changed in value
        # # get time steps
        # cdef cnp.ndarray[cnp.float64_t, ndim=1] old_time_steps = np.ascontiguousarray(np.zeros((self.N,)), dtype=np.float64)
        # assert acados_solver.acados_get_time_steps(self.capsule, self.N, <double *> old_time_steps.data)

        # if np.array_equal(old_time_steps, new_time_steps):
        #     return

        # N = new_time_steps.size
        # cdef cnp.ndarray[cnp.float64_t, ndim=1] value = np.ascontiguousarray(new_time_steps, dtype=np.float64)

        # # check if recreation of acados is necessary (no need to recreate acados if sizes are identical)
        # if len(old_time_steps) == N:
        #     assert acados_solver.acados_update_time_steps(self.capsule, N, <double *> value.data) == 0

        # else:  # recreate the solver with the new time steps
        #     self.solver_created = False

        #     # delete old memory (analog to __del__)
        #     acados_solver.acados_free(self.capsule)

        #     # create solver with new time steps
        #     assert acados_solver.acados_create_with_discretization(self.capsule, N, <double *> value.data) == 0

        #     self.solver_created = True

        #     # get pointers solver
        #     self.__get_pointers_solver()

        # # store time_steps, N
        # self.time_steps = new_time_steps
        # self.N = N


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
        raise NotImplementedError("AcadosOcpSolverCython: does not support update_qp_solver_cond_N() since it is only a prototyping feature")

        # # unlikely but still possible
        # if not self.solver_created:
        #     raise Exception('Solver was not yet created!')
        # if self.N < qp_solver_cond_N:
        #     raise Exception('Setting qp_solver_cond_N to be larger than N does not work!')
        # if self.qp_solver_cond_N != qp_solver_cond_N:
        #     self.solver_created = False

        #     # recreate the solver
        #     acados_solver.acados_update_qp_solver_cond_N(self.capsule, qp_solver_cond_N)

        #     # store the new value
        #     self.qp_solver_cond_N = qp_solver_cond_N
        #     self.solver_created = True

        #     # get pointers solver
        #     self.__get_pointers_solver()


    def eval_param_sens(self, index, stage=0, field="ex"):
        """
        Calculate the sensitivity of the curent solution with respect to the initial state component of index

            :param index: integer corresponding to initial state index in range(nx)
        """

        field_ = field
        field = field_.encode('utf-8')

        # checks
        if not isinstance(index, int):
            raise Exception('AcadosOcpSolverCython.eval_param_sens(): index must be Integer.')

        cdef int nx = acados_solver_common.ocp_nlp_dims_get_from_attr(self.nlp_config, self.nlp_dims, self.nlp_out, 0, "x".encode('utf-8'))

        if index < 0 or index > nx:
            raise Exception(f'AcadosOcpSolverCython.eval_param_sens(): index must be in [0, nx-1], got: {index}.')

        # actual eval_param
        acados_solver_common.ocp_nlp_eval_param_sens(self.nlp_solver, field, stage, index, self.sens_out)

        return


    def get(self, int stage, str field_):
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
        field = field_.encode('utf-8')

        if field_ not in out_fields:
            raise Exception('AcadosOcpSolverCython.get(): {} is an invalid argument.\
                    \n Possible values are {}.'.format(field_, out_fields))

        if stage < 0 or stage > self.N:
            raise Exception('AcadosOcpSolverCython.get(): stage index must be in [0, N], got: {}.'.format(self.N))

        if stage == self.N and field_ == 'pi':
            raise Exception('AcadosOcpSolverCython.get(): field {} does not exist at final stage {}.'\
                .format(field_, stage))

        cdef int dims = acados_solver_common.ocp_nlp_dims_get_from_attr(self.nlp_config,
            self.nlp_dims, self.nlp_out, stage, field)

        cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.zeros((dims,))
        acados_solver_common.ocp_nlp_out_get(self.nlp_config, \
            self.nlp_dims, self.nlp_out, stage, field, <void *> out.data)

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
            - qp_res_stat: stationarity residual of the last QP solution
            - qp_res_eq: residual wrt equality constraints (dynamics) of the last QP solution
            - qp_res_ineq: residual wrt inequality constraints (constraints)  of the last QP solution
            - qp_res_comp: residual wrt complementarity conditions of the last QP solution
        """
        acados_solver.acados_print_stats(self.capsule)


    def store_iterate(self, filename='', overwrite=False):
        """
        Stores the current iterate of the ocp solver in a json file.

            :param filename: if not set, use model_name + timestamp + '.json'
            :param overwrite: if false and filename exists add timestamp to filename
        """
        import json
        if filename == '':
            filename += self.model_name + '_' + 'iterate' + '.json'

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
            json.dump(solution, f, default=lambda x: x.tolist(), indent=4, sort_keys=True)
        print("stored current iterate in ", os.path.join(os.getcwd(), filename))


    def load_iterate(self, filename):
        """
        Loads the iterate stored in json file with filename into the ocp solver.
        """
        import json
        if not os.path.isfile(filename):
            raise Exception('load_iterate: failed, file does not exist: ' + os.path.join(os.getcwd(), filename))

        with open(filename, 'r') as f:
            solution = json.load(f)

        for key in solution.keys():
            (field, stage) = key.split('_')
            self.set(int(stage), field, np.array(solution[key]))


    def get_stats(self, field_):
        """
        Get the information of the last solver call.

            :param field: string in ['statistics', 'time_tot', 'time_lin', 'time_sim', 'time_sim_ad', 'time_sim_la', 'time_qp', 'time_qp_solver_call', 'time_reg', 'sqp_iter']
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
            return self.__get_stat_int(field)

        elif field_ in double_fields:
            return self.__get_stat_double(field)

        elif field_ == 'statistics':
            sqp_iter = self.get_stats("sqp_iter")
            stat_m = self.get_stats("stat_m")
            stat_n = self.get_stats("stat_n")
            min_size = min([stat_m, sqp_iter+1])
            return self.__get_stat_matrix(field, stat_n+1, min_size)

        elif field_ == 'qp_iter':
            full_stats = self.get_stats('statistics')
            if self.nlp_solver_type == 'SQP':
                return full_stats[6, :]
            elif self.nlp_solver_type == 'SQP_RTI':
                return full_stats[2, :]

        elif field_ == 'alpha':
            full_stats = self.get_stats('statistics')
            if self.nlp_solver_type == 'SQP':
                return full_stats[7, :]
            else: # self.nlp_solver_type == 'SQP_RTI':
                raise Exception("alpha values are not available for SQP_RTI")

        elif field_ == 'residuals':
            return self.get_residuals()

        else:
            raise NotImplementedError("TODO!")


    def __get_stat_int(self, field):
        cdef int out
        acados_solver_common.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, <void *> &out)
        return out

    def __get_stat_double(self, field):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.zeros((1,))
        acados_solver_common.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, <void *> out.data)
        return out

    def __get_stat_matrix(self, field, n, m):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] out_mat = np.ascontiguousarray(np.zeros((n, m)), dtype=np.float64)
        acados_solver_common.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, <void *> out_mat.data)
        return out_mat


    def get_cost(self):
        """
        Returns the cost value of the current solution.
        """
        # compute cost internally
        acados_solver_common.ocp_nlp_eval_cost(self.nlp_solver, self.nlp_in, self.nlp_out)

        # create output
        cdef double out

        # call getter
        acados_solver_common.ocp_nlp_get(self.nlp_config, self.nlp_solver, "cost_value", <void *> &out)

        return out


    def get_residuals(self, recompute=False):
        """
        Returns an array of the form [res_stat, res_eq, res_ineq, res_comp].
        """
        # compute residuals if RTI
        if self.nlp_solver_type == 'SQP_RTI' or recompute:
            acados_solver_common.ocp_nlp_eval_residuals(self.nlp_solver, self.nlp_in, self.nlp_out)

        # create output array
        cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.ascontiguousarray(np.zeros((4,), dtype=np.float64))
        cdef double double_value

        field = "res_stat".encode('utf-8')
        acados_solver_common.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, <void *> &double_value)
        out[0] = double_value

        field = "res_eq".encode('utf-8')
        acados_solver_common.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, <void *> &double_value)
        out[1] = double_value

        field = "res_ineq".encode('utf-8')
        acados_solver_common.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, <void *> &double_value)
        out[2] = double_value

        field = "res_comp".encode('utf-8')
        acados_solver_common.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, <void *> &double_value)
        out[3] = double_value

        return out


    # Note: this function should not be used anymore, better use cost_set, constraints_set
    def set(self, int stage, str field_, value_):

        """
        Set numerical data inside the solver.

            :param stage: integer corresponding to shooting node
            :param field: string in ['x', 'u', 'pi', 'lam', 't', 'p']

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
        if not isinstance(value_, np.ndarray):
            raise Exception(f"set: value must be numpy array, got {type(value_)}.")
        cost_fields = ['y_ref', 'yref']
        constraints_fields = ['lbx', 'ubx', 'lbu', 'ubu']
        out_fields = ['x', 'u', 'pi', 'lam', 't', 'z', 'sl', 'su']
        mem_fields = ['xdot_guess', 'z_guess']

        field = field_.encode('utf-8')

        cdef cnp.ndarray[cnp.float64_t, ndim=1] value = np.ascontiguousarray(value_, dtype=np.float64)

        # treat parameters separately
        if field_ == 'p':
            assert acados_solver.acados_update_params(self.capsule, stage, <double *> value.data, value.shape[0]) == 0
        else:
            if field_ not in constraints_fields + cost_fields + out_fields:
                raise Exception("AcadosOcpSolverCython.set(): {} is not a valid argument.\
                    \nPossible values are {}.".format(field, \
                    constraints_fields + cost_fields + out_fields + ['p']))

            dims = acados_solver_common.ocp_nlp_dims_get_from_attr(self.nlp_config,
                self.nlp_dims, self.nlp_out, stage, field)

            if value_.shape[0] != dims:
                msg = 'AcadosOcpSolverCython.set(): mismatching dimension for field "{}" '.format(field_)
                msg += 'with dimension {} (you have {})'.format(dims, value_.shape[0])
                raise Exception(msg)

            if field_ in constraints_fields:
                acados_solver_common.ocp_nlp_constraints_model_set(self.nlp_config,
                    self.nlp_dims, self.nlp_in, stage, field, <void *> value.data)
            elif field_ in cost_fields:
                acados_solver_common.ocp_nlp_cost_model_set(self.nlp_config,
                    self.nlp_dims, self.nlp_in, stage, field, <void *> value.data)
            elif field_ in out_fields:
                acados_solver_common.ocp_nlp_out_set(self.nlp_config,
                    self.nlp_dims, self.nlp_out, stage, field, <void *> value.data)
            elif field_ in mem_fields:
                acados_solver_common.ocp_nlp_set(self.nlp_config, \
                    self.nlp_solver, stage, field, <void *> value.data)

            if field_ == 'z':
                field = 'z_guess'.encode('utf-8')
                acados_solver_common.ocp_nlp_set(self.nlp_config, \
                    self.nlp_solver, stage, field, <void *> value.data)
        return

    def cost_set(self, int stage, str field_, value_):
        """
        Set numerical data in the cost module of the solver.

            :param stage: integer corresponding to shooting node
            :param field: string, e.g. 'yref', 'W', 'ext_cost_num_hess'
            :param value: of appropriate size
        """
        if not isinstance(value_, np.ndarray):
            raise Exception(f"cost_set: value must be numpy array, got {type(value_)}.")
        field = field_.encode('utf-8')

        cdef int dims[2]
        acados_solver_common.ocp_nlp_cost_dims_get_from_attr(self.nlp_config, \
            self.nlp_dims, self.nlp_out, stage, field, &dims[0])

        cdef double[::1,:] value

        value_shape = value_.shape
        if len(value_shape) == 1:
            value_shape = (value_shape[0], 0)
            value = np.asfortranarray(value_[None,:])

        elif len(value_shape) == 2:
            # Get elements in column major order
            value = np.asfortranarray(value_)

        if value_shape[0] != dims[0] or value_shape[1] != dims[1]:
            raise Exception('AcadosOcpSolverCython.cost_set(): mismatching dimension' +
                f' for field "{field_}" at stage {stage} with dimension {tuple(dims)} (you have {value_shape})')

        acados_solver_common.ocp_nlp_cost_model_set(self.nlp_config, \
            self.nlp_dims, self.nlp_in, stage, field, <void *> &value[0][0])


    def constraints_set(self, int stage, str field_, value_):
        """
        Set numerical data in the constraint module of the solver.

            :param stage: integer corresponding to shooting node
            :param field: string in ['lbx', 'ubx', 'lbu', 'ubu', 'lg', 'ug', 'lh', 'uh', 'uphi', 'C', 'D']
            :param value: of appropriate size
        """
        if not isinstance(value_, np.ndarray):
            raise Exception(f"constraints_set: value must be numpy array, got {type(value_)}.")

        field = field_.encode('utf-8')

        cdef int dims[2]
        acados_solver_common.ocp_nlp_constraint_dims_get_from_attr(self.nlp_config, \
            self.nlp_dims, self.nlp_out, stage, field, &dims[0])

        cdef double[::1,:] value

        value_shape = value_.shape
        if len(value_shape) == 1:
            value_shape = (value_shape[0], 0)
            value = np.asfortranarray(value_[None,:])

        elif len(value_shape) == 2:
            # Get elements in column major order
            value = np.asfortranarray(value_)

        if value_shape != tuple(dims):
            raise Exception(f'AcadosOcpSolverCython.constraints_set(): mismatching dimension' +
                f' for field "{field_}" at stage {stage} with dimension {tuple(dims)} (you have {value_shape})')

        acados_solver_common.ocp_nlp_constraints_model_set(self.nlp_config, \
            self.nlp_dims, self.nlp_in, stage, field, <void *> &value[0][0])

        return


    def get_from_qp_in(self, int stage, str field_):
        """
        Get numerical data from the dynamics module of the solver:

            :param stage: integer corresponding to shooting node
            :param field: string, e.g. 'A'
        """
        field = field_.encode('utf-8')

        # get dims
        cdef int[2] dims
        acados_solver_common.ocp_nlp_qp_dims_get_from_attr(self.nlp_config, self.nlp_dims, self.nlp_out, stage, field, &dims[0])

        # create output data
        cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.zeros((dims[0], dims[1]), order='F')

        # call getter
        acados_solver_common.ocp_nlp_get_at_stage(self.nlp_config, self.nlp_dims, self.nlp_solver, stage, field, <void *> out.data)

        return out


    def options_set(self, str field_, value_):
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
        int_fields = ['print_level', 'rti_phase', 'initialize_t_slacks', 'qp_warm_start', 'line_search_use_sufficient_descent', 'full_step_dual', 'globalization_use_SOC', 'warm_start_first_qp']
        double_fields = ['step_length', 'tol_eq', 'tol_stat', 'tol_ineq', 'tol_comp', 'alpha_min', 'alpha_reduction', 'eps_sufficient_descent',
        'qp_tol_stat', 'qp_tol_eq', 'qp_tol_ineq', 'qp_tol_comp', 'qp_tau_min', 'qp_mu0']
        string_fields = ['globalization']

        # encode
        field = field_.encode('utf-8')

        cdef int int_value
        cdef double double_value
        cdef unsigned char[::1] string_value

        # check field availability and type
        if field_ in int_fields:
            if not isinstance(value_, int):
                raise Exception('solver option {} must be of type int. You have {}.'.format(field_, type(value_)))

            if field_ == 'rti_phase':
                if value_ < 0 or value_ > 2:
                    raise Exception('AcadosOcpSolverCython.solve(): argument \'rti_phase\' can '
                        'take only values 0, 1, 2 for SQP-RTI-type solvers')
                if self.nlp_solver_type != 'SQP_RTI' and value_ > 0:
                    raise Exception('AcadosOcpSolverCython.solve(): argument \'rti_phase\' can '
                        'take only value 0 for SQP-type solvers')

            int_value = value_
            acados_solver_common.ocp_nlp_solver_opts_set(self.nlp_config, self.nlp_opts, field, <void *> &int_value)

        elif field_ in double_fields:
            if not isinstance(value_, float):
                raise Exception('solver option {} must be of type float. You have {}.'.format(field_, type(value_)))

            double_value = value_
            acados_solver_common.ocp_nlp_solver_opts_set(self.nlp_config, self.nlp_opts, field, <void *> &double_value)

        elif field_ in string_fields:
            if not isinstance(value_, bytes):
                raise Exception('solver option {} must be of type str. You have {}.'.format(field_, type(value_)))

            string_value = value_.encode('utf-8')
            acados_solver_common.ocp_nlp_solver_opts_set(self.nlp_config, self.nlp_opts, field, <void *> &string_value[0])

        else:
            raise Exception('AcadosOcpSolverCython.options_set() does not support field {}.'\
                '\n Possible values are {}.'.format(field_, ', '.join(int_fields + double_fields + string_fields)))


    def set_params_sparse(self, int stage, idx_values_, param_values_):
        """
        set parameters of the solvers external function partially:
        Pseudo: solver.param[idx_values_] = param_values_;
        Parameters:
            :param stage_: integer corresponding to shooting node
            :param idx_values_: 0 based integer array corresponding to parameter indices to be set
            :param param_values_: new parameter values as numpy array
        """

        if not isinstance(param_values_, np.ndarray):
            raise Exception('param_values_ must be np.array.')

        if param_values_.shape[0] != len(idx_values_):
            raise Exception(f'param_values_ and idx_values_ must be of the same size.' +
                 f' Got sizes idx {param_values_.shape[0]}, param_values {len(idx_values_)}.')

        # n_update = c_int(len(param_values_))

        # param_data = cast(param_values_.ctypes.data, POINTER(c_double))
        # c_idx_values = np.ascontiguousarray(idx_values_, dtype=np.intc)
        # idx_data = cast(c_idx_values.ctypes.data, POINTER(c_int))

        # getattr(self.shared_lib, f"{self.model_name}_acados_update_params_sparse").argtypes = \
        #                 [c_void_p, c_int, POINTER(c_int), POINTER(c_double), c_int]
        # getattr(self.shared_lib, f"{self.model_name}_acados_update_params_sparse").restype = c_int
        # getattr(self.shared_lib, f"{self.model_name}_acados_update_params_sparse") \
        #                             (self.capsule, stage, idx_data, param_data, n_update)

        cdef cnp.ndarray[cnp.float64_t, ndim=1] value = np.ascontiguousarray(param_values_, dtype=np.float64)
        # cdef cnp.ndarray[cnp.intc, ndim=1] idx = np.ascontiguousarray(idx_values_, dtype=np.intc)

        # NOTE: this does throw an error somehow:
        # ValueError: Buffer dtype mismatch, expected 'int object' but got 'int'
        # cdef cnp.ndarray[cnp.int, ndim=1] idx = np.ascontiguousarray(idx_values_, dtype=np.intc)

        cdef cnp.ndarray[cnp.int32_t, ndim=1] idx = np.ascontiguousarray(idx_values_, dtype=np.int32)
        cdef int n_update = value.shape[0]
        # print(f"in set_params_sparse Cython n_update {n_update}")

        assert acados_solver.acados_update_params_sparse(self.capsule, stage, <int *> idx.data, <double *> value.data, n_update) == 0
        return


    def __del__(self):
        if self.solver_created:
            acados_solver.acados_free(self.capsule)
            acados_solver.acados_free_capsule(self.capsule)
