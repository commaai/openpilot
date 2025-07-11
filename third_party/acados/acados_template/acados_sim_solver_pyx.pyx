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
# from libc cimport bool as bool_t

cimport acados_sim_solver_common
cimport acados_sim_solver

cimport numpy as cnp

import os
from datetime import datetime
import numpy as np


cdef class AcadosSimSolverCython:
    """
    Class to interact with the acados sim solver C object.
    """

    cdef acados_sim_solver.sim_solver_capsule *capsule
    cdef void *sim_dims
    cdef acados_sim_solver_common.sim_opts *sim_opts
    cdef acados_sim_solver_common.sim_config *sim_config
    cdef acados_sim_solver_common.sim_out *sim_out
    cdef acados_sim_solver_common.sim_in *sim_in
    cdef acados_sim_solver_common.sim_solver *sim_solver

    cdef bint solver_created

    cdef str model_name

    cdef str sim_solver_type

    cdef list gettable_vectors
    cdef list gettable_matrices
    cdef list gettable_scalars

    def __cinit__(self, model_name):

        self.solver_created = False

        self.model_name = model_name

        # create capsule
        self.capsule = acados_sim_solver.acados_sim_solver_create_capsule()

        # create solver
        assert acados_sim_solver.acados_sim_create(self.capsule) == 0
        self.solver_created = True

        # get pointers solver
        self.__get_pointers_solver()

        self.gettable_vectors = ['x', 'u', 'z', 'S_adj']
        self.gettable_matrices = ['S_forw', 'Sx', 'Su', 'S_hess', 'S_algebraic']
        self.gettable_scalars = ['CPUtime', 'time_tot', 'ADtime', 'time_ad', 'LAtime', 'time_la']

    def __get_pointers_solver(self):
        """
        Private function to get the pointers for solver
        """
        # get pointers solver
        self.sim_opts = acados_sim_solver.acados_get_sim_opts(self.capsule)
        self.sim_dims = acados_sim_solver.acados_get_sim_dims(self.capsule)
        self.sim_config = acados_sim_solver.acados_get_sim_config(self.capsule)
        self.sim_out = acados_sim_solver.acados_get_sim_out(self.capsule)
        self.sim_in = acados_sim_solver.acados_get_sim_in(self.capsule)
        self.sim_solver = acados_sim_solver.acados_get_sim_solver(self.capsule)


    def simulate(self, x=None, u=None, z=None, p=None):
        """
        Simulate the system forward for the given x, u, z, p and return x_next.
        Wrapper around `solve()` taking care of setting/getting inputs/outputs.
        """
        if x is not None:
            self.set('x', x)
        if u is not None:
            self.set('u', u)
        if z is not None:
            self.set('z', z)
        if p is not None:
            self.set('p', p)

        status = self.solve()

        if status == 2:
            print("Warning: acados_sim_solver reached maximum iterations.")
        elif status != 0:
            raise Exception(f'acados_sim_solver for model {self.model_name} returned status {status}.')

        x_next = self.get('x')
        return x_next


    def solve(self):
        """
        Solve the sim with current input.
        """
        return acados_sim_solver.acados_sim_solve(self.capsule)


    def get(self, field_):
        """
        Get the last solution of the solver.

            :param str field: string in ['x', 'u', 'z', 'S_forw', 'Sx', 'Su', 'S_adj', 'S_hess', 'S_algebraic', 'CPUtime', 'time_tot', 'ADtime', 'time_ad', 'LAtime', 'time_la']
        """
        field = field_.encode('utf-8')

        if field_ in self.gettable_vectors:
            return self.__get_vector(field)
        elif field_ in self.gettable_matrices:
            return self.__get_matrix(field)
        elif field_ in self.gettable_scalars:
            return self.__get_scalar(field)
        else:
            raise Exception(f'AcadosSimSolver.get(): Unknown field {field_},' \
                f' available fields are {", ".join(self.gettable.keys())}')


    def __get_scalar(self, field):
        cdef double scalar
        acados_sim_solver_common.sim_out_get(self.sim_config, self.sim_dims, self.sim_out, field, <void *> &scalar)
        return scalar


    def __get_vector(self, field):
        cdef int[2] dims
        acados_sim_solver_common.sim_dims_get_from_attr(self.sim_config, self.sim_dims, field, &dims[0])
        # cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.ascontiguousarray(np.zeros((dims[0],), dtype=np.float64))
        cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.zeros((dims[0]),)
        acados_sim_solver_common.sim_out_get(self.sim_config, self.sim_dims, self.sim_out, field, <void *> out.data)
        return out


    def __get_matrix(self, field):
        cdef int[2] dims
        acados_sim_solver_common.sim_dims_get_from_attr(self.sim_config, self.sim_dims, field, &dims[0])
        cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.zeros((dims[0], dims[1]), order='F', dtype=np.float64)
        acados_sim_solver_common.sim_out_get(self.sim_config, self.sim_dims, self.sim_out, field, <void *> out.data)
        return out


    def set(self, field_: str, value_):
        """
        Set numerical data inside the solver.

            :param field: string in ['p', 'seed_adj', 'T', 'x', 'u', 'xdot', 'z']
            :param value: the value with appropriate size.
        """
        settable = ['seed_adj', 'T', 'x', 'u', 'xdot', 'z', 'p'] # S_forw

        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        # if len(value_.shape) > 1:
            # raise RuntimeError('AcadosSimSolverCython.set(): value_ should be 1 dimensional')

        cdef cnp.ndarray[cnp.float64_t, ndim=1] value = np.ascontiguousarray(value_, dtype=np.float64).flatten()

        field = field_.encode('utf-8')
        cdef int[2] dims

        # treat parameters separately
        if field_ == 'p':
            assert acados_sim_solver.acados_sim_update_params(self.capsule, <double *> value.data, value.shape[0]) == 0
            return
        else:
            acados_sim_solver_common.sim_dims_get_from_attr(self.sim_config, self.sim_dims, field, &dims[0])

            value_ = np.ravel(value_, order='F')

            value_shape = value_.shape
            if len(value_shape) == 1:
                value_shape = (value_shape[0], 0)

            if value_shape != tuple(dims):
                raise Exception(f'AcadosSimSolverCython.set(): mismatching dimension' \
                    f' for field "{field_}" with dimension {tuple(dims)} (you have {value_shape}).')

        # set
        if field_ in ['xdot', 'z']:
            acados_sim_solver_common.sim_solver_set(self.sim_solver, field, <void *> value.data)
        elif field_ in settable:
            acados_sim_solver_common.sim_in_set(self.sim_config, self.sim_dims, self.sim_in, field, <void *> value.data)
        else:
            raise Exception(f'AcadosSimSolverCython.set(): Unknown field {field_},' \
                f' available fields are {", ".join(settable)}')


    def options_set(self, field_: str, value_: bool):
        """
        Set solver options

            :param field: string in ['sens_forw', 'sens_adj', 'sens_hess']
            :param value: Boolean
        """
        fields = ['sens_forw', 'sens_adj', 'sens_hess']
        if field_ not in fields:
            raise Exception(f"field {field_} not supported. Supported values are {', '.join(fields)}.\n")

        field = field_.encode('utf-8')

        if not isinstance(value_, bool):
            raise TypeError("options_set: expected boolean for value")

        cdef bint bool_value = value_
        acados_sim_solver_common.sim_opts_set(self.sim_config, self.sim_opts, field, <void *> &bool_value)
        # TODO: only allow setting
        # if getattr(self.acados_sim.solver_options, field_) or value_ == False:
        #     acados_sim_solver_common.sim_opts_set(self.sim_config, self.sim_opts, field, <void *> &bool_value)
        # else:
        #     raise RuntimeError(f"Cannot set option {field_} to True, because it was False in original solver options.\n")

        return


    def __del__(self):
        if self.solver_created:
            acados_sim_solver.acados_sim_free(self.capsule)
            acados_sim_solver.acados_sim_solver_free_capsule(self.capsule)
