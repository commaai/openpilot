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

import sys
import os
import json
import importlib

import numpy as np

from subprocess import DEVNULL, call, STDOUT

from ctypes import POINTER, cast, CDLL, c_void_p, c_char_p, c_double, c_int, c_bool, byref
from copy import deepcopy

from .casadi_function_generation import generate_c_code_implicit_ode, generate_c_code_gnsf, generate_c_code_explicit_ode
from .acados_sim import AcadosSim
from .acados_ocp import AcadosOcp
from .utils import is_column, render_template, format_class_dict, make_object_json_dumpable,\
     make_model_consistent, set_up_imported_gnsf_model, get_python_interface_path, get_lib_ext,\
     casadi_length, is_empty, check_casadi_version
from .builders import CMakeBuilder
from .gnsf.detect_gnsf_structure import detect_gnsf_structure



def make_sim_dims_consistent(acados_sim: AcadosSim):
    dims = acados_sim.dims
    model = acados_sim.model
    # nx
    if is_column(model.x):
        dims.nx = casadi_length(model.x)
    else:
        raise Exception('model.x should be column vector!')

    # nu
    if is_empty(model.u):
        dims.nu = 0
    else:
        dims.nu = casadi_length(model.u)

    # nz
    if is_empty(model.z):
        dims.nz = 0
    else:
        dims.nz = casadi_length(model.z)

    # np
    if is_empty(model.p):
        dims.np = 0
    else:
        dims.np = casadi_length(model.p)
    if acados_sim.parameter_values.shape[0] != dims.np:
        raise Exception('inconsistent dimension np, regarding model.p and parameter_values.' + \
            f'\nGot np = {dims.np}, acados_sim.parameter_values.shape = {acados_sim.parameter_values.shape[0]}\n')


def get_sim_layout():
    python_interface_path = get_python_interface_path()
    abs_path = os.path.join(python_interface_path, 'acados_sim_layout.json')
    with open(abs_path, 'r') as f:
        sim_layout = json.load(f)
    return sim_layout


def sim_formulation_json_dump(acados_sim: AcadosSim, json_file='acados_sim.json'):
    # Load acados_sim structure description
    sim_layout = get_sim_layout()

    # Copy input sim object dictionary
    sim_dict = dict(deepcopy(acados_sim).__dict__)

    for key, v in sim_layout.items():
        # skip non dict attributes
        if not isinstance(v, dict): continue
        # Copy sim object attributes dictionaries
        sim_dict[key]=dict(getattr(acados_sim, key).__dict__)

    sim_json = format_class_dict(sim_dict)

    with open(json_file, 'w') as f:
        json.dump(sim_json, f, default=make_object_json_dumpable, indent=4, sort_keys=True)


def sim_get_default_cmake_builder() -> CMakeBuilder:
    """
    If :py:class:`~acados_template.acados_sim_solver.AcadosSimSolver` is used with `CMake` this function returns a good first setting.
    :return: default :py:class:`~acados_template.builders.CMakeBuilder`
    """
    cmake_builder = CMakeBuilder()
    cmake_builder.options_on = ['BUILD_ACADOS_SIM_SOLVER_LIB']
    return cmake_builder


def sim_render_templates(json_file, model_name: str, code_export_dir, cmake_options: CMakeBuilder = None):
    # setting up loader and environment
    json_path = os.path.join(os.getcwd(), json_file)

    if not os.path.exists(json_path):
        raise Exception(f"{json_path} not found!")

    # Render templates
    in_file = 'acados_sim_solver.in.c'
    out_file = f'acados_sim_solver_{model_name}.c'
    render_template(in_file, out_file, code_export_dir, json_path)

    in_file = 'acados_sim_solver.in.h'
    out_file = f'acados_sim_solver_{model_name}.h'
    render_template(in_file, out_file, code_export_dir, json_path)

    in_file = 'acados_sim_solver.in.pxd'
    out_file = f'acados_sim_solver.pxd'
    render_template(in_file, out_file, code_export_dir, json_path)

    # Builder
    if cmake_options is not None:
        in_file = 'CMakeLists.in.txt'
        out_file = 'CMakeLists.txt'
        render_template(in_file, out_file, code_export_dir, json_path)
    else:
        in_file = 'Makefile.in'
        out_file = 'Makefile'
        render_template(in_file, out_file, code_export_dir, json_path)

    in_file = 'main_sim.in.c'
    out_file = f'main_sim_{model_name}.c'
    render_template(in_file, out_file, code_export_dir, json_path)

    # folder model
    model_dir = os.path.join(code_export_dir, model_name + '_model')

    in_file = 'model.in.h'
    out_file = f'{model_name}_model.h'
    render_template(in_file, out_file, model_dir, json_path)


def sim_generate_external_functions(acados_sim: AcadosSim):
    model = acados_sim.model
    model = make_model_consistent(model)

    integrator_type = acados_sim.solver_options.integrator_type

    opts = dict(generate_hess = acados_sim.solver_options.sens_hess,
                code_export_directory = acados_sim.code_export_directory)

    # create code_export_dir, model_dir
    code_export_dir = acados_sim.code_export_directory
    opts['code_export_directory'] = code_export_dir
    model_dir = os.path.join(code_export_dir, model.name + '_model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # generate external functions
    check_casadi_version()
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
    if sys.platform=="win32":
        from ctypes import wintypes
        from ctypes import WinDLL
        dlclose = WinDLL('kernel32', use_last_error=True).FreeLibrary
        dlclose.argtypes = [wintypes.HMODULE]
    else:
        dlclose = CDLL(None).dlclose
        dlclose.argtypes = [c_void_p]


    @classmethod
    def generate(cls, acados_sim: AcadosSim, json_file='acados_sim.json', cmake_builder: CMakeBuilder = None):
        """
        Generates the code for an acados sim solver, given the description in acados_sim
        """

        acados_sim.code_export_directory = os.path.abspath(acados_sim.code_export_directory)

        # make dims consistent
        make_sim_dims_consistent(acados_sim)

        # module dependent post processing
        if acados_sim.solver_options.integrator_type == 'GNSF':
            if acados_sim.solver_options.sens_hess == True:
                raise Exception("AcadosSimSolver: GNSF does not support sens_hess = True.")
            if 'gnsf_model' in acados_sim.__dict__:
                set_up_imported_gnsf_model(acados_sim)
            else:
                detect_gnsf_structure(acados_sim)

        # generate external functions
        sim_generate_external_functions(acados_sim)

        # dump to json
        sim_formulation_json_dump(acados_sim, json_file)

        # render templates
        sim_render_templates(json_file, acados_sim.model.name, acados_sim.code_export_directory, cmake_builder)


    @classmethod
    def build(cls, code_export_dir, with_cython=False, cmake_builder: CMakeBuilder = None, verbose: bool = True):
        # Compile solver
        cwd = os.getcwd()
        os.chdir(code_export_dir)
        if with_cython:
            call(
                ['make', 'clean_sim_cython'],
                stdout=None if verbose else DEVNULL,
                stderr=None if verbose else STDOUT
            )
            call(
                ['make', 'sim_cython'],
                stdout=None if verbose else DEVNULL,
                stderr=None if verbose else STDOUT
            )
        else:
            if cmake_builder is not None:
                cmake_builder.exec(code_export_dir, verbose=verbose)
            else:
                call(
                    ['make', 'sim_shared_lib'],
                    stdout=None if verbose else DEVNULL,
                    stderr=None if verbose else STDOUT
                )
        os.chdir(cwd)


    @classmethod
    def create_cython_solver(cls, json_file):
        """
        """
        with open(json_file, 'r') as f:
            acados_sim_json = json.load(f)
        code_export_directory = acados_sim_json['code_export_directory']

        importlib.invalidate_caches()
        rel_code_export_directory = os.path.relpath(code_export_directory)
        acados_sim_solver_pyx = importlib.import_module(f'{rel_code_export_directory}.acados_sim_solver_pyx')

        AcadosSimSolverCython = getattr(acados_sim_solver_pyx, 'AcadosSimSolverCython')
        return AcadosSimSolverCython(acados_sim_json['model']['name'])

    def __init__(self, acados_sim, json_file='acados_sim.json', generate=True, build=True, cmake_builder: CMakeBuilder = None, verbose: bool = True):

        self.solver_created = False
        self.acados_sim = acados_sim
        model_name = acados_sim.model.name
        self.model_name = model_name

        code_export_dir = os.path.abspath(acados_sim.code_export_directory)

        # reuse existing json and casadi functions, when creating integrator from ocp
        if generate and not isinstance(acados_sim, AcadosOcp):
            self.generate(acados_sim, json_file=json_file, cmake_builder=cmake_builder)

        if build:
            self.build(code_export_dir, cmake_builder=cmake_builder, verbose=True)

        # prepare library loading
        lib_prefix = 'lib'
        lib_ext = get_lib_ext()
        if os.name == 'nt':
            lib_prefix = ''

        # Load acados library to avoid unloading the library.
        # This is necessary if acados was compiled with OpenMP, since the OpenMP threads can't be destroyed.
        # Unloading a library which uses OpenMP results in a segfault (on any platform?).
        # see [https://stackoverflow.com/questions/34439956/vc-crash-when-freeing-a-dll-built-with-openmp]
        # or [https://python.hotexamples.com/examples/_ctypes/-/dlclose/python-dlclose-function-examples.html]
        libacados_name = f'{lib_prefix}acados{lib_ext}'
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
        libacados_sim_solver_name = f'{lib_prefix}acados_sim_solver_{self.model_name}{lib_ext}'
        self.shared_lib_name = os.path.join(code_export_dir, libacados_sim_solver_name)

        # get shared_lib
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

        self.gettable_vectors = ['x', 'u', 'z', 'S_adj']
        self.gettable_matrices = ['S_forw', 'Sx', 'Su', 'S_hess', 'S_algebraic']
        self.gettable_scalars = ['CPUtime', 'time_tot', 'ADtime', 'time_ad', 'LAtime', 'time_la']


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
        Solve the simulation problem with current input.
        """
        getattr(self.shared_lib, f"{self.model_name}_acados_sim_solve").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{self.model_name}_acados_sim_solve").restype = c_int

        status = getattr(self.shared_lib, f"{self.model_name}_acados_sim_solve")(self.capsule)
        return status


    def get(self, field_):
        """
        Get the last solution of the solver.

            :param str field: string in ['x', 'u', 'z', 'S_forw', 'Sx', 'Su', 'S_adj', 'S_hess', 'S_algebraic', 'CPUtime', 'time_tot', 'ADtime', 'time_ad', 'LAtime', 'time_la']
        """
        field = field_.encode('utf-8')

        if field_ in self.gettable_vectors:
            # get dims
            dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
            dims_data = cast(dims.ctypes.data, POINTER(c_int))

            self.shared_lib.sim_dims_get_from_attr.argtypes = [c_void_p, c_void_p, c_char_p, POINTER(c_int)]
            self.shared_lib.sim_dims_get_from_attr(self.sim_config, self.sim_dims, field, dims_data)

            # allocate array
            out = np.ascontiguousarray(np.zeros((dims[0],)), dtype=np.float64)
            out_data = cast(out.ctypes.data, POINTER(c_double))

            self.shared_lib.sim_out_get.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.sim_out_get(self.sim_config, self.sim_dims, self.sim_out, field, out_data)

        elif field_ in self.gettable_matrices:
            # get dims
            dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
            dims_data = cast(dims.ctypes.data, POINTER(c_int))

            self.shared_lib.sim_dims_get_from_attr.argtypes = [c_void_p, c_void_p, c_char_p, POINTER(c_int)]
            self.shared_lib.sim_dims_get_from_attr(self.sim_config, self.sim_dims, field, dims_data)

            out = np.zeros((dims[0], dims[1]), order='F')
            out_data = cast(out.ctypes.data, POINTER(c_double))

            self.shared_lib.sim_out_get.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.sim_out_get(self.sim_config, self.sim_dims, self.sim_out, field, out_data)

        elif field_ in self.gettable_scalars:
            scalar = c_double()
            scalar_data = byref(scalar)
            self.shared_lib.sim_out_get.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.sim_out_get(self.sim_config, self.sim_dims, self.sim_out, field, scalar_data)

            out = scalar.value
        else:
            raise Exception(f'AcadosSimSolver.get(): Unknown field {field_},' \
                f' available fields are {", ".join(self.gettable_vectors+self.gettable_matrices)}, {", ".join(self.gettable_scalars)}')

        return out



    def set(self, field_: str, value_):
        """
        Set numerical data inside the solver.

            :param field: string in ['x', 'u', 'p', 'xdot', 'z', 'seed_adj', 'T']
            :param value: the value with appropriate size.
        """
        settable = ['x', 'u', 'p', 'xdot', 'z', 'seed_adj', 'T'] # S_forw

        # TODO: check and throw error here. then remove checks in Cython for speed
        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])

        value_ = value_.astype(float)
        value_data = cast(value_.ctypes.data, POINTER(c_double))
        value_data_p = cast((value_data), c_void_p)

        field = field_.encode('utf-8')

        # treat parameters separately
        if field_ == 'p':
            model_name = self.acados_sim.model.name
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
                raise Exception(f'AcadosSimSolver.set(): mismatching dimension' \
                    f' for field "{field_}" with dimension {tuple(dims)} (you have {value_shape}).')

        # set
        if field_ in ['xdot', 'z']:
            self.shared_lib.sim_solver_set.argtypes = [c_void_p, c_char_p, c_void_p]
            self.shared_lib.sim_solver_set(self.sim_solver, field, value_data_p)
        elif field_ in settable:
            self.shared_lib.sim_in_set.argtypes = [c_void_p, c_void_p, c_void_p, c_char_p, c_void_p]
            self.shared_lib.sim_in_set(self.sim_config, self.sim_dims, self.sim_in, field, value_data_p)
        else:
            raise Exception(f'AcadosSimSolver.set(): Unknown field {field_},' \
                f' available fields are {", ".join(settable)}')

        return


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
        value_ctypes = c_bool(value_)

        if not isinstance(value_, bool):
            raise TypeError("options_set: expected boolean for value")

        # only allow setting
        if getattr(self.acados_sim.solver_options, field_) or value_ == False:
            self.shared_lib.sim_opts_set.argtypes = [c_void_p, c_void_p, c_char_p, POINTER(c_bool)]
            self.shared_lib.sim_opts_set(self.sim_config, self.sim_opts, field, value_ctypes)
        else:
            raise RuntimeError(f"Cannot set option {field_} to True, because it was False in original solver options.\n")

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
                print(f"WARNING: acados Python interface could not close shared_lib handle of AcadosSimSolver {self.model_name}.\n",
                     "Attempting to create a new one with the same name will likely result in the old one being used!")
                pass
