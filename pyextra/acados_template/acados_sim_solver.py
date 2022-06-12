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

import sys, os, json

import numpy as np

from ctypes import *
from copy import deepcopy

from .generate_c_code_explicit_ode import generate_c_code_explicit_ode
from .generate_c_code_implicit_ode import generate_c_code_implicit_ode
from .generate_c_code_gnsf import generate_c_code_gnsf
from .acados_sim import AcadosSim
from .acados_ocp import AcadosOcp
from .acados_model import acados_model_strip_casadi_symbolics
from .utils import is_column, render_template, format_class_dict, np_array_to_list,\
     make_model_consistent, set_up_imported_gnsf_model, get_python_interface_path
from .builders import CMakeBuilder


def make_sim_dims_consistent(acados_sim):
    dims = acados_sim.dims
    model = acados_sim.model
    # nx
    if is_column(model.x):
        dims.nx = model.x.shape[0]
    else:
        raise Exception("model.x should be column vector!")

    # nu
    if is_column(model.u):
        dims.nu = model.u.shape[0]
    elif model.u == None or model.u == []:
        dims.nu = 0
    else:
        raise Exception("model.u should be column vector or None!")

    # nz
    if is_column(model.z):
        dims.nz = model.z.shape[0]
    elif model.z == None or model.z == []:
        dims.nz = 0
    else:
        raise Exception("model.z should be column vector or None!")

    # np
    if is_column(model.p):
        dims.np = model.p.shape[0]
    elif model.p == None or model.p == []:
        dims.np = 0
    else:
        raise Exception("model.p should be column vector or None!")


def get_sim_layout():
    python_interface_path = get_python_interface_path()
    abs_path = os.path.join(python_interface_path, 'acados_sim_layout.json')
    with open(abs_path, 'r') as f:
        sim_layout = json.load(f)
    return sim_layout


def sim_formulation_json_dump(acados_sim, json_file='acados_sim.json'):
    # Load acados_sim structure description
    sim_layout = get_sim_layout()

    # Copy input sim object dictionary
    sim_dict = dict(deepcopy(acados_sim).__dict__)

    for key, v in sim_layout.items():
        # skip non dict attributes
        if not isinstance(v, dict): continue
        # Copy sim object attributes dictionaries
        sim_dict[key]=dict(getattr(acados_sim, key).__dict__)

    sim_dict['model'] = acados_model_strip_casadi_symbolics(sim_dict['model'])
    sim_json = format_class_dict(sim_dict)

    with open(json_file, 'w') as f:
        json.dump(sim_json, f, default=np_array_to_list, indent=4, sort_keys=True)


def sim_get_default_cmake_builder() -> CMakeBuilder:
    """
    If :py:class:`~acados_template.acados_sim_solver.AcadosSimSolver` is used with `CMake` this function returns a good first setting.
    :return: default :py:class:`~acados_template.builders.CMakeBuilder`
    """
    cmake_builder = CMakeBuilder()
    cmake_builder.options_on = ['BUILD_ACADOS_SIM_SOLVER_LIB']
    return cmake_builder


def sim_render_templates(json_file, model_name, code_export_dir, cmake_options: CMakeBuilder = None):
    # setting up loader and environment
    json_path = os.path.join(os.getcwd(), json_file)

    if not os.path.exists(json_path):
        raise Exception(f"{json_path} not found!")

    template_dir = code_export_dir

    ## Render templates
    in_file = 'acados_sim_solver.in.c'
    out_file = f'acados_sim_solver_{model_name}.c'
    render_template(in_file, out_file, template_dir, json_path)

    in_file = 'acados_sim_solver.in.h'
    out_file = f'acados_sim_solver_{model_name}.h'
    render_template(in_file, out_file, template_dir, json_path)

    # Builder
    if cmake_options is not None:
        in_file = 'CMakeLists.in.txt'
        out_file = 'CMakeLists.txt'
        render_template(in_file, out_file, template_dir, json_path)
    else:
        in_file = 'Makefile.in'
        out_file = 'Makefile'
        render_template(in_file, out_file, template_dir, json_path)

    in_file = 'main_sim.in.c'
    out_file = f'main_sim_{model_name}.c'
    render_template(in_file, out_file, template_dir, json_path)

    ## folder model
    template_dir = os.path.join(code_export_dir, model_name + '_model')

    in_file = 'model.in.h'
    out_file = f'{model_name}_model.h'
    render_template(in_file, out_file, template_dir, json_path)


def sim_generate_casadi_functions(acados_sim):
    model = acados_sim.model
    model = make_model_consistent(model)

    integrator_type = acados_sim.solver_options.integrator_type

    opts = dict(generate_hess = acados_sim.solver_options.sens_hess,
                code_export_directory = acados_sim.code_export_directory)
    # generate external functions
    if integrator_type == 'ERK':
        generate_c_code_explicit_ode(model, opts)
    elif integrator_type == 'IRK':
        generate_c_code_implicit_ode(model, opts)
    elif integrator_type == 'GNSF':
        generate_c_code_gnsf(model, opts)


class AcadosSimSolver:
    """
    Class to interact with the acados integrator C object.

        :param acados_sim: type :py:class:`~acados_template.acados_ocp.AcadosOcp` (takes values to generate an instance :py:class:`~acados_template.acados_sim.AcadosSim`) or :py:class:`~acados_template.acados_sim.AcadosSim`
        :param json_file: Default: 'acados_sim.json'
        :param build: Default: True
        :param cmake_builder: type :py:class:`~acados_template.utils.CMakeBuilder` generate a `CMakeLists.txt` and use
            the `CMake` pipeline instead of a `Makefile` (`CMake` seems to be the better option in conjunction with
            `MS Visual Studio`); default: `None`
    """
    def __init__(self, acados_sim_, json_file='acados_sim.json', build=True, cmake_builder: CMakeBuilder = None):

        self.solver_created = False

        if isinstance(acados_sim_, AcadosOcp):
            # set up acados_sim_
            acados_sim = AcadosSim()
            acados_sim.model = acados_sim_.model
            acados_sim.dims.nx = acados_sim_.dims.nx
            acados_sim.dims.nu = acados_sim_.dims.nu
            acados_sim.dims.nz = acados_sim_.dims.nz
            acados_sim.dims.np = acados_sim_.dims.np
            acados_sim.solver_options.integrator_type = acados_sim_.solver_options.integrator_type
            acados_sim.code_export_directory = acados_sim_.code_export_directory

        elif isinstance(acados_sim_, AcadosSim):
            acados_sim = acados_sim_

        acados_sim.__problem_class = 'SIM'

        model_name = acados_sim.model.name
        make_sim_dims_consistent(acados_sim)

        # reuse existing json and casadi functions, when creating integrator from ocp
        if isinstance(acados_sim_, AcadosSim):
            if acados_sim.solver_options.integrator_type == 'GNSF':
                set_up_imported_gnsf_model(acados_sim)

            sim_generate_casadi_functions(acados_sim)
            sim_formulation_json_dump(acados_sim, json_file)

        code_export_dir = acados_sim.code_export_directory
        if build:
            # render templates
            sim_render_templates(json_file, model_name, code_export_dir, cmake_builder)

            # Compile solver
            cwd = os.getcwd()
            code_export_dir = os.path.abspath(code_export_dir)
            os.chdir(code_export_dir)
            if cmake_builder is not None:
                cmake_builder.exec(code_export_dir)
            else:
                os.system('make sim_shared_lib')
            os.chdir(cwd)

        self.sim_struct = acados_sim
        model_name = self.sim_struct.model.name
        self.model_name = model_name

        # Load acados library to avoid unloading the library.
        # This is necessary if acados was compiled with OpenMP, since the OpenMP threads can't be destroyed.
        # Unloading a library which uses OpenMP results in a segfault (on any platform?).
        # see [https://stackoverflow.com/questions/34439956/vc-crash-when-freeing-a-dll-built-with-openmp]
        # or [https://python.hotexamples.com/examples/_ctypes/-/dlclose/python-dlclose-function-examples.html]
        libacados_name = 'libacados.so'
        libacados_filepath = os.path.join(acados_sim.acados_lib_path, libacados_name)
        self.__acados_lib = CDLL(libacados_filepath)
        # find out if acados was compiled with OpenMP
        try:
            self.__acados_lib_uses_omp = getattr(self.__acados_lib, 'omp_get_thread_num') is not None
        except AttributeError as e:
            self.__acados_lib_uses_omp = False
        if self.__acados_lib_uses_omp:
            print('acados was compiled with OpenMP.')
        else:
            print('acados was compiled without OpenMP.')

        # Ctypes
        lib_prefix = 'lib'
        lib_ext = '.so'
        if os.name == 'nt':
            lib_prefix = ''
            lib_ext = ''
        self.shared_lib_name = os.path.join(code_export_dir, f'{lib_prefix}acados_sim_solver_{model_name}{lib_ext}')
        print(f'self.shared_lib_name = "{self.shared_lib_name}"')
        
        self.shared_lib = CDLL(self.shared_lib_name)


        # create capsule
        getattr(self.shared_lib, f"{model_name}_acados_sim_solver_create_capsule").restype = c_void_p
        self.capsule = getattr(self.shared_lib, f"{model_name}_acados_sim_solver_create_capsule")()

        # create solver
        getattr(self.shared_lib, f"{model_name}_acados_sim_create").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_sim_create").restype = c_int
        assert getattr(self.shared_lib, f"{model_name}_acados_sim_create")(self.capsule)==0
        self.solver_created = True

        getattr(self.shared_lib, f"{model_name}_acados_get_sim_opts").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_sim_opts").restype = c_void_p
        self.sim_opts = getattr(self.shared_lib, f"{model_name}_acados_get_sim_opts")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_sim_dims").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_sim_dims").restype = c_void_p
        self.sim_dims = getattr(self.shared_lib, f"{model_name}_acados_get_sim_dims")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_sim_config").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_sim_config").restype = c_void_p
        self.sim_config = getattr(self.shared_lib, f"{model_name}_acados_get_sim_config")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_sim_out").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_sim_out").restype = c_void_p
        self.sim_out = getattr(self.shared_lib, f"{model_name}_acados_get_sim_out")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_sim_in").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_sim_in").restype = c_void_p
        self.sim_in = getattr(self.shared_lib, f"{model_name}_acados_get_sim_in")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_sim_solver").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_sim_solver").restype = c_void_p
        self.sim_solver = getattr(self.shared_lib, f"{model_name}_acados_get_sim_solver")(self.capsule)

        nu = self.sim_struct.dims.nu
        nx = self.sim_struct.dims.nx
        nz = self.sim_struct.dims.nz
        self.gettable = {
            'x': nx,
            'xn': nx,
            'u': nu,
            'z': nz,
            'S_forw': nx*(nx+nu),
            'Sx': nx*nx,
            'Su': nx*nu,
            'S_adj': nx+nu,
            'S_hess': (nx+nu)*(nx+nu),
            'S_algebraic': (nz)*(nx+nu),
        }

        self.settable = ['S_adj', 'T', 'x', 'u', 'xdot', 'z', 'p'] # S_forw


    def solve(self):
        """
        Solve the simulation problem with current input.
        """
        getattr(self.shared_lib, f"{self.model_name}_acados_sim_solve").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_sim_solve").restype = c_int

        status = getattr(self.shared_lib, f"{self.model_name}_acados_sim_solve")(self.capsule)
        return status


    def get(self, field_):
        """
        Get the last solution of the solver.

            :param str field: string in ['x', 'u', 'z', 'S_forw', 'Sx', 'Su', 'S_adj', 'S_hess', 'S_algebraic']
        """
        field = field_
        field = field.encode('utf-8')

        if field_ in self.gettable.keys():

            # allocate array
            dims = self.gettable[field_]
            out = np.ascontiguousarray(np.zeros((dims,)), dtype=np.float64)
            out_data = cast(out.ctypes.data, POINTER(c_double))

            self.shared_lib.sim_out_get.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.sim_out_get(self.sim_config, self.sim_dims, self.sim_out, field, out_data)

            if field_ == 'S_forw':
                nu = self.sim_struct.dims.nu
                nx = self.sim_struct.dims.nx
                out = out.reshape(nx, nx+nu, order='F')
            elif field_ == 'Sx':
                nx = self.sim_struct.dims.nx
                out = out.reshape(nx, nx, order='F')
            elif field_ == 'Su':
                nx = self.sim_struct.dims.nx
                nu = self.sim_struct.dims.nu
                out = out.reshape(nx, nu, order='F')
            elif field_ == 'S_hess':
                nx = self.sim_struct.dims.nx
                nu = self.sim_struct.dims.nu
                out = out.reshape(nx+nu, nx+nu, order='F')
            elif field_ == 'S_algebraic':
                nx = self.sim_struct.dims.nx
                nu = self.sim_struct.dims.nu
                nz = self.sim_struct.dims.nz
                out = out.reshape(nz, nx+nu, order='F')
        else:
            raise Exception(f'AcadosSimSolver.get(): Unknown field {field_},' \
                f' available fields are {", ".join(self.gettable.keys())}')

        return out


    def set(self, field_, value_):
        """
        Set numerical data inside the solver.

            :param field: string in ['p', 'S_adj', 'T', 'x', 'u', 'xdot', 'z']
            :param value: the value with appropriate size.
        """
        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])

        value_ = value_.astype(float)
        value_data = cast(value_.ctypes.data, POINTER(c_double))
        value_data_p = cast((value_data), c_void_p)

        field = field_
        field = field.encode('utf-8')

        # treat parameters separately
        if field_ == 'p':
            model_name = self.sim_struct.model.name
            getattr(self.shared_lib, f"{model_name}_acados_sim_update_params").argtypes = [c_void_p, POINTER(c_double), c_int]
            value_data = cast(value_.ctypes.data, POINTER(c_double))
            getattr(self.shared_lib, f"{model_name}_acados_sim_update_params")(self.capsule, value_data, value_.shape[0])
            return
        else:
            # dimension check
            dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
            dims_data = cast(dims.ctypes.data, POINTER(c_int))

            self.shared_lib.sim_dims_get_from_attr.argtypes = [c_void_p, c_void_p, c_char_p, POINTER(c_int)]
            self.shared_lib.sim_dims_get_from_attr(self.sim_config, self.sim_dims, field, dims_data)

            value_ = np.ravel(value_, order='F')

            value_shape = value_.shape
            if len(value_shape) == 1:
                value_shape = (value_shape[0], 0)

            if value_shape != tuple(dims):
                raise Exception('AcadosSimSolver.set(): mismatching dimension' \
                    ' for field "{}" with dimension {} (you have {})'.format(field_, tuple(dims), value_shape))

        # set
        if field_ in ['xdot', 'z']:
            self.shared_lib.sim_solver_set.argtypes = [c_void_p, c_char_p, c_void_p]
            self.shared_lib.sim_solver_set(self.sim_solver, field, value_data_p)
        elif field_ in self.settable:
            self.shared_lib.sim_in_set.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.sim_in_set(self.sim_config, self.sim_dims, self.sim_in, field, value_data_p)
        else:
            raise Exception(f'AcadosSimSolver.set(): Unknown field {field_},' \
                f' available fields are {", ".join(self.settable)}')

        return


    def __del__(self):

        if self.solver_created:
            getattr(self.shared_lib, f"{self.model_name}_acados_sim_free").argtypes = [c_void_p]
            getattr(self.shared_lib, f"{self.model_name}_acados_sim_free").restype = c_int
            getattr(self.shared_lib, f"{self.model_name}_acados_sim_free")(self.capsule)

            getattr(self.shared_lib, f"{self.model_name}_acados_sim_solver_free_capsule").argtypes = [c_void_p]
            getattr(self.shared_lib, f"{self.model_name}_acados_sim_solver_free_capsule").restype = c_int
            getattr(self.shared_lib, f"{self.model_name}_acados_sim_solver_free_capsule")(self.capsule)

            try:
                self.dlclose(self.shared_lib._handle)
            except:
                pass
