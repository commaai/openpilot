# -*- coding: future_fstrings -*-
#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
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

import numpy as np
import casadi as ca
import os
from .acados_model import AcadosModel
from .utils import get_acados_path

class AcadosSimDims:
    """
    Class containing the dimensions of the model to be simulated.
    """
    def __init__(self):
        self.__nx = None
        self.__nu = None
        self.__nz = 0
        self.__np = 0

    @property
    def nx(self):
        """:math:`n_x` - number of states. Type: int > 0"""
        return self.__nx

    @property
    def nz(self):
        """:math:`n_z` - number of algebraic variables. Type: int >= 0"""
        return self.__nz

    @property
    def nu(self):
        """:math:`n_u` - number of inputs. Type: int >= 0"""
        return self.__nu

    @property
    def np(self):
        """:math:`n_p` - number of parameters. Type: int >= 0"""
        return self.__np

    @nx.setter
    def nx(self, nx):
        if type(nx) == int and nx > 0:
            self.__nx = nx
        else:
            raise Exception('Invalid nx value, expected positive integer. Exiting.')

    @nz.setter
    def nz(self, nz):
        if type(nz) == int and nz > -1:
            self.__nz = nz
        else:
            raise Exception('Invalid nz value, expected nonnegative integer. Exiting.')

    @nu.setter
    def nu(self, nu):
        if type(nu) == int and nu > -1:
            self.__nu = nu
        else:
            raise Exception('Invalid nu value, expected nonnegative integer. Exiting.')

    @np.setter
    def np(self, np):
        if type(np) == int and np > -1:
            self.__np = np
        else:
            raise Exception('Invalid np value, expected nonnegative integer. Exiting.')

    def set(self, attr, value):
        setattr(self, attr, value)


class AcadosSimOpts:
    """
    class containing the solver options
    """
    def __init__(self):
        self.__integrator_type = 'ERK'
        self.__tf = None
        # ints
        self.__sim_method_num_stages = 1
        self.__sim_method_num_steps = 1
        self.__sim_method_newton_iter = 3
        # bools
        self.__sens_forw = True
        self.__sens_adj = False
        self.__sens_algebraic = False
        self.__sens_hess = False
        self.__output_z = False
        self.__sim_method_jac_reuse = False

    @property
    def integrator_type(self):
        """Integrator type. Default: 'ERK'."""
        return self.__integrator_type

    @property
    def num_stages(self):
        """Number of stages in the integrator. Default: 1"""
        return self.__sim_method_num_stages

    @property
    def num_steps(self):
        """Number of steps in the integrator. Default: 1"""
        return self.__sim_method_num_steps

    @property
    def newton_iter(self):
        """Number of Newton iterations in simulation method. Default: 3"""
        return self.__sim_method_newton_iter

    @property
    def sens_forw(self):
        """Boolean determining if forward sensitivities are computed. Default: True"""
        return self.__sens_forw

    @property
    def sens_adj(self):
        """Boolean determining if adjoint sensitivities are computed. Default: False"""
        return self.__sens_adj

    @property
    def sens_algebraic(self):
        """Boolean determining if sensitivities wrt algebraic variables are computed. Default: False"""
        return self.__sens_algebraic

    @property
    def sens_hess(self):
        """Boolean determining if hessians are computed. Default: False"""
        return self.__sens_hess

    @property
    def output_z(self):
        """Boolean determining if values for algebraic variables (corresponding to start of simulation interval) are computed. Default: False"""
        return self.__output_z

    @property
    def sim_method_jac_reuse(self):
        """Boolean determining if jacobians are reused. Default: False"""
        return self.__sim_method_jac_reuse

    @property
    def T(self):
        """Time horizon"""
        return self.__Tsim

    @integrator_type.setter
    def integrator_type(self, integrator_type):
        integrator_types = ('ERK', 'IRK', 'GNSF')
        if integrator_type in integrator_types:
            self.__integrator_type = integrator_type
        else:
            raise Exception('Invalid integrator_type value. Possible values are:\n\n' \
                    + ',\n'.join(integrator_types) + '.\n\nYou have: ' + integrator_type + '.\n\nExiting.')

    @T.setter
    def T(self, T):
        self.__Tsim = T

    @num_stages.setter
    def num_stages(self, num_stages):
        if isinstance(num_stages, int):
            self.__sim_method_num_stages = num_stages
        else:
            raise Exception('Invalid num_stages value. num_stages must be an integer.')

    @num_steps.setter
    def num_steps(self, num_steps):
        if isinstance(num_steps, int):
            self.__sim_method_num_steps = num_steps
        else:
            raise Exception('Invalid num_steps value. num_steps must be an integer.')

    @newton_iter.setter
    def newton_iter(self, newton_iter):
        if isinstance(newton_iter, int):
            self.__sim_method_newton_iter = newton_iter
        else:
            raise Exception('Invalid newton_iter value. newton_iter must be an integer.')

    @sens_forw.setter
    def sens_forw(self, sens_forw):
        if sens_forw in (True, False):
            self.__sens_forw = sens_forw
        else:
            raise Exception('Invalid sens_forw value. sens_forw must be a Boolean.')

    @sens_adj.setter
    def sens_adj(self, sens_adj):
        if sens_adj in (True, False):
            self.__sens_adj = sens_adj
        else:
            raise Exception('Invalid sens_adj value. sens_adj must be a Boolean.')

    @sens_hess.setter
    def sens_hess(self, sens_hess):
        if sens_hess in (True, False):
            self.__sens_hess = sens_hess
        else:
            raise Exception('Invalid sens_hess value. sens_hess must be a Boolean.')

    @sens_algebraic.setter
    def sens_algebraic(self, sens_algebraic):
        if sens_algebraic in (True, False):
            self.__sens_algebraic = sens_algebraic
        else:
            raise Exception('Invalid sens_algebraic value. sens_algebraic must be a Boolean.')

    @output_z.setter
    def output_z(self, output_z):
        if output_z in (True, False):
            self.__output_z = output_z
        else:
            raise Exception('Invalid output_z value. output_z must be a Boolean.')

    @sim_method_jac_reuse.setter
    def sim_method_jac_reuse(self, sim_method_jac_reuse):
        if sim_method_jac_reuse in (True, False):
            self.__sim_method_jac_reuse = sim_method_jac_reuse
        else:
            raise Exception('Invalid sim_method_jac_reuse value. sim_method_jac_reuse must be a Boolean.')

class AcadosSim:
    """
    The class has the following properties that can be modified to formulate a specific simulation problem, see below:

    :param acados_path: string with the path to acados. It is used to generate the include and lib paths.

    - :py:attr:`dims` of type :py:class:`acados_template.acados_ocp.AcadosSimDims` - are automatically detected from model
    - :py:attr:`model` of type :py:class:`acados_template.acados_model.AcadosModel`
    - :py:attr:`solver_options` of type :py:class:`acados_template.acados_sim.AcadosSimOpts`

    - :py:attr:`acados_include_path` (set automatically)
    - :py:attr:`acados_lib_path` (set automatically)
    """
    def __init__(self, acados_path=''):
        if acados_path == '':
            acados_path = get_acados_path()
        self.dims = AcadosSimDims()
        """Dimension definitions, automatically detected from :py:attr:`model`. Type :py:class:`acados_template.acados_sim.AcadosSimDims`"""
        self.model = AcadosModel()
        """Model definitions, type :py:class:`acados_template.acados_model.AcadosModel`"""
        self.solver_options = AcadosSimOpts()
        """Solver Options, type :py:class:`acados_template.acados_sim.AcadosSimOpts`"""

        self.acados_include_path = f'{acados_path}/include'
        """Path to acados include directors (set automatically), type: `string`"""
        self.acados_lib_path = f'{acados_path}/lib'
        """Path to where acados library is located (set automatically), type: `string`"""

        self.code_export_directory = 'c_generated_code'
        """Path to where code will be exported. Default: `c_generated_code`."""

    def set(self, attr, value):
        # tokenize string
        tokens = attr.split('_', 1)
        if len(tokens) > 1:
            setter_to_call = getattr(getattr(self, tokens[0]), 'set')
        else:
            setter_to_call = getattr(self, 'set')

        setter_to_call(tokens[1], value)

        return
