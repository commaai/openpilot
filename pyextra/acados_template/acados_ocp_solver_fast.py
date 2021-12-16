import sys
import os
import json
import numpy as np
from datetime import datetime

from ctypes import POINTER, CDLL, c_void_p, c_int, cast, c_double, c_char_p

from copy import deepcopy

from .generate_c_code_explicit_ode import generate_c_code_explicit_ode
from .generate_c_code_implicit_ode import generate_c_code_implicit_ode
from .generate_c_code_gnsf import generate_c_code_gnsf
from .generate_c_code_discrete_dynamics import generate_c_code_discrete_dynamics
from .generate_c_code_constraint import generate_c_code_constraint
from .generate_c_code_nls_cost import generate_c_code_nls_cost
from .generate_c_code_external_cost import generate_c_code_external_cost
from .acados_ocp import AcadosOcp
from .acados_model import acados_model_strip_casadi_symbolics
from .utils import is_column, is_empty, casadi_length, render_template, acados_class2dict,\
     format_class_dict, ocp_check_against_layout, np_array_to_list, make_model_consistent,\
     set_up_imported_gnsf_model, get_acados_path


class AcadosOcpSolverFast:
    dlclose = CDLL(None).dlclose
    dlclose.argtypes = [c_void_p]

    def __init__(self, model_name, N, code_export_dir):

        self.solver_created = False
        self.N = N
        self.model_name = model_name

        self.shared_lib_name = f'{code_export_dir}/libacados_ocp_solver_{model_name}.so'

        # get shared_lib
        self.shared_lib = CDLL(self.shared_lib_name)

        # create capsule
        getattr(self.shared_lib, f"{model_name}_acados_create_capsule").restype = c_void_p
        self.capsule = getattr(self.shared_lib, f"{model_name}_acados_create_capsule")()

        # create solver
        getattr(self.shared_lib, f"{model_name}_acados_create").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_create").restype = c_int
        assert getattr(self.shared_lib, f"{model_name}_acados_create")(self.capsule)==0
        self.solver_created = True

        # get pointers solver
        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_opts").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_opts").restype = c_void_p
        self.nlp_opts = getattr(self.shared_lib, f"{model_name}_acados_get_nlp_opts")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_dims").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_dims").restype = c_void_p
        self.nlp_dims = getattr(self.shared_lib, f"{model_name}_acados_get_nlp_dims")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_config").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_config").restype = c_void_p
        self.nlp_config = getattr(self.shared_lib, f"{model_name}_acados_get_nlp_config")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_out").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_out").restype = c_void_p
        self.nlp_out = getattr(self.shared_lib, f"{model_name}_acados_get_nlp_out")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_in").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_in").restype = c_void_p
        self.nlp_in = getattr(self.shared_lib, f"{model_name}_acados_get_nlp_in")(self.capsule)

        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_solver").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_get_nlp_solver").restype = c_void_p
        self.nlp_solver = getattr(self.shared_lib, f"{model_name}_acados_get_nlp_solver")(self.capsule)


    def solve(self):
        """
        Solve the ocp with current input.
        """
        model_name = self.model_name

        getattr(self.shared_lib, f"{model_name}_acados_solve").argtypes = [c_void_p]
        getattr(self.shared_lib, f"{model_name}_acados_solve").restype = c_int
        status = getattr(self.shared_lib, f"{model_name}_acados_solve")(self.capsule)
        return status

    def cost_set(self, start_stage_, field_, value_, api='warn'):
      self.cost_set_slice(start_stage_, start_stage_+1, field_, value_[None], api='warn')
      return

    def cost_set_slice(self, start_stage_, end_stage_, field_, value_, api='warn'):
        """
        Set numerical data in the cost module of the solver.

            :param stage: integer corresponding to shooting node
            :param field: string, e.g. 'yref', 'W', 'ext_cost_num_hess'
            :param value: of appropriate size
        """
        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = np.ascontiguousarray(np.copy(value_), dtype=np.float64)
        field = field_
        field = field.encode('utf-8')
        dim = np.product(value_.shape[1:])

        start_stage = c_int(start_stage_)
        end_stage = c_int(end_stage_)
        self.shared_lib.ocp_nlp_cost_dims_get_from_attr.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p, POINTER(c_int)]
        self.shared_lib.ocp_nlp_cost_dims_get_from_attr.restype = c_int

        dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
        dims_data = cast(dims.ctypes.data, POINTER(c_int))

        self.shared_lib.ocp_nlp_cost_dims_get_from_attr(self.nlp_config,
            self.nlp_dims, self.nlp_out, start_stage_, field, dims_data)

        value_shape = value_.shape
        expected_shape = tuple(np.concatenate([np.array([end_stage_ - start_stage_]), dims]))
        if len(value_shape) == 2:
            value_shape = (value_shape[0], value_shape[1], 0)

        elif len(value_shape) == 3:
            if api=='old':
                pass
            elif api=='warn':
                if not np.all(np.ravel(value_, order='F')==np.ravel(value_, order='K')):
                    raise Exception("Ambiguity in API detected.\n"
                                    "Are you making an acados model from scrach? Add api='new' to cost_set and carry on.\n"
                                    "Are you seeing this error suddenly in previously running code? Read on.\n"
                                    "  You are relying on a now-fixed bug in cost_set for field '{}'.\n".format(field_) +
                                    "  acados_template now correctly passes on any matrices to acados in column major format.\n" +
                                    "  Two options to fix this error: \n" +
                                    "   * Add api='old' to cost_set to restore old incorrect behaviour\n" +
                                    "   * Add api='new' to cost_set and remove any unnatural manipulation of the value argument " +
                                    "such as non-mathematical transposes, reshaping, casting to fortran order, etc... " +
                                    "If there is no such manipulation, then you have probably been getting an incorrect solution before.")
                # Get elements in column major order
                value_ = np.ravel(value_, order='F')
            elif api=='new':
                # Get elements in column major order
                value_ = np.ravel(value_, order='F')
            else:
                raise Exception(f"Unknown api: '{api}'")

        if value_shape != expected_shape:
            raise Exception('AcadosOcpSolver.cost_set(): mismatching dimension',
                            ' for field "{}" with dimension {} (you have {})'.format(
                               field_, expected_shape, value_shape))


        value_data = cast(value_.ctypes.data, POINTER(c_double))
        value_data_p = cast((value_data), c_void_p)

        self.shared_lib.ocp_nlp_cost_model_set_slice.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_int, c_char_p, c_void_p, c_int]
        self.shared_lib.ocp_nlp_cost_model_set_slice(self.nlp_config,
            self.nlp_dims, self.nlp_in, start_stage, end_stage, field, value_data_p, dim)
        return

    def constraints_set(self, start_stage_, field_, value_, api='warn'):
      self.constraints_set_slice(start_stage_, start_stage_+1, field_, value_[None], api='warn')
      return

    def constraints_set_slice(self, start_stage_, end_stage_, field_, value_, api='warn'):
        """
        Set numerical data in the constraint module of the solver.

            :param stage: integer corresponding to shooting node
            :param field: string in ['lbx', 'ubx', 'lbu', 'ubu', 'lg', 'ug', 'lh', 'uh', 'uphi']
            :param value: of appropriate size
        """
        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = value_.astype(float)

        field = field_
        field = field.encode('utf-8')
        dim = np.product(value_.shape[1:])

        start_stage = c_int(start_stage_)
        end_stage = c_int(end_stage_)
        self.shared_lib.ocp_nlp_constraint_dims_get_from_attr.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p, POINTER(c_int)]
        self.shared_lib.ocp_nlp_constraint_dims_get_from_attr.restype = c_int

        dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
        dims_data = cast(dims.ctypes.data, POINTER(c_int))

        self.shared_lib.ocp_nlp_constraint_dims_get_from_attr(self.nlp_config, \
            self.nlp_dims, self.nlp_out, start_stage_, field, dims_data)

        value_shape = value_.shape
        expected_shape = tuple(np.concatenate([np.array([end_stage_ - start_stage_]), dims]))
        if len(value_shape) == 2:
            value_shape = (value_shape[0], value_shape[1], 0)
        elif len(value_shape) == 3:
            if api=='old':
                pass
            elif api=='warn':
                if not np.all(np.ravel(value_, order='F')==np.ravel(value_, order='K')):
                    raise Exception("Ambiguity in API detected.\n"
                                    "Are you making an acados model from scrach? Add api='new' to constraints_set and carry on.\n"
                                    "Are you seeing this error suddenly in previously running code? Read on.\n"
                                    "  You are relying on a now-fixed bug in constraints_set for field '{}'.\n".format(field_) +
                                    "  acados_template now correctly passes on any matrices to acados in column major format.\n" +
                                    "  Two options to fix this error: \n" +
                                    "   * Add api='old' to constraints_set to restore old incorrect behaviour\n" +
                                    "   * Add api='new' to constraints_set and remove any unnatural manipulation of the value argument " +
                                    "such as non-mathematical transposes, reshaping, casting to fortran order, etc... " +
                                    "If there is no such manipulation, then you have probably been getting an incorrect solution before.")
                # Get elements in column major order
                value_ = np.ravel(value_, order='F')
            elif api=='new':
                # Get elements in column major order
                value_ = np.ravel(value_, order='F')
            else:
                raise Exception(f"Unknown api: '{api}'")
        if value_shape != expected_shape:
            raise Exception('AcadosOcpSolver.constraints_set(): mismatching dimension' \
                ' for field "{}" with dimension {} (you have {})'.format(field_, expected_shape, value_shape))

        value_data = cast(value_.ctypes.data, POINTER(c_double))
        value_data_p = cast((value_data), c_void_p)

        self.shared_lib.ocp_nlp_constraints_model_set_slice.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_int, c_char_p, c_void_p, c_int]
        self.shared_lib.ocp_nlp_constraints_model_set_slice(self.nlp_config, \
            self.nlp_dims, self.nlp_in, start_stage, end_stage, field, value_data_p, dim)
        return

    # Note: this function should not be used anymore, better use cost_set, constraints_set
    def set(self, stage_, field_, value_):
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
        cost_fields = ['y_ref', 'yref']
        constraints_fields = ['lbx', 'ubx', 'lbu', 'ubu']
        out_fields = ['x', 'u', 'pi', 'lam', 't', 'z']
        mem_fields = ['sl', 'su']

        # cast value_ to avoid conversion issues
        if isinstance(value_, (float, int)):
            value_ = np.array([value_])
        value_ = value_.astype(float)

        model_name = self.model_name

        field = field_
        field = field.encode('utf-8')

        stage = c_int(stage_)

        # treat parameters separately
        if field_ == 'p':
            getattr(self.shared_lib, f"{model_name}_acados_update_params").argtypes = [c_void_p, c_int, POINTER(c_double)]
            getattr(self.shared_lib, f"{model_name}_acados_update_params").restype = c_int

            value_data = cast(value_.ctypes.data, POINTER(c_double))

            assert getattr(self.shared_lib, f"{model_name}_acados_update_params")(self.capsule, stage, value_data, value_.shape[0])==0
        else:
            if field_ not in constraints_fields + cost_fields + out_fields + mem_fields:
                raise Exception("AcadosOcpSolver.set(): {} is not a valid argument.\
                    \nPossible values are {}. Exiting.".format(field, \
                    constraints_fields + cost_fields + out_fields + ['p']))

            self.shared_lib.ocp_nlp_dims_get_from_attr.argtypes = \
                [c_void_p, c_void_p, c_void_p, c_int, c_char_p]
            self.shared_lib.ocp_nlp_dims_get_from_attr.restype = c_int

            dims = self.shared_lib.ocp_nlp_dims_get_from_attr(self.nlp_config, \
                self.nlp_dims, self.nlp_out, stage_, field)

            if value_.shape[0] != dims:
                msg = f'AcadosOcpSolver.set(): mismatching dimension for field "{field_}" '
                msg += f'with dimension {dims} (you have {value_.shape})'
                raise Exception(msg)

            value_data = cast(value_.ctypes.data, POINTER(c_double))
            value_data_p = cast((value_data), c_void_p)

            if field_ in constraints_fields:
                self.shared_lib.ocp_nlp_constraints_model_set.argtypes = \
                    [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_constraints_model_set(self.nlp_config, \
                    self.nlp_dims, self.nlp_in, stage, field, value_data_p)
            elif field_ in cost_fields:
                self.shared_lib.ocp_nlp_cost_model_set.argtypes = \
                    [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_cost_model_set(self.nlp_config, \
                    self.nlp_dims, self.nlp_in, stage, field, value_data_p)
            elif field_ in out_fields:
                self.shared_lib.ocp_nlp_out_set.argtypes = \
                    [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_out_set(self.nlp_config, \
                    self.nlp_dims, self.nlp_out, stage, field, value_data_p)
            elif field_ in mem_fields:
                self.shared_lib.ocp_nlp_set.argtypes = \
                    [c_void_p, c_void_p, c_int, c_char_p, c_void_p]
                self.shared_lib.ocp_nlp_set(self.nlp_config, \
                    self.nlp_solver, stage, field, value_data_p)
        return


    def get_slice(self, start_stage_, end_stage_, field_):
        """
        Get the last solution of the solver:

            :param start_stage: integer corresponding to shooting node that indicates start of slice
            :param end_stage: integer corresponding to shooting node that indicates end of slice
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
        out_fields = ['x', 'u', 'z', 'pi', 'lam', 't']
        mem_fields = ['sl', 'su']
        field = field_
        field = field.encode('utf-8')

        if (field_ not in out_fields + mem_fields):
            raise Exception('AcadosOcpSolver.get_slice(): {} is an invalid argument.\
                    \n Possible values are {}. Exiting.'.format(field_, out_fields))

        if not isinstance(start_stage_, int):
            raise Exception('AcadosOcpSolver.get_slice(): stage index must be Integer.')

        if not isinstance(end_stage_, int):
            raise Exception('AcadosOcpSolver.get_slice(): stage index must be Integer.')

        if start_stage_ >= end_stage_:
            raise Exception('AcadosOcpSolver.get_slice(): end stage index must be larger than start stage index')

        if start_stage_ < 0 or end_stage_ > self.N + 1:
            raise Exception(f'AcadosOcpSolver.get_slice(): stage index must be in [0, N], got: {self.N}.')
        self.shared_lib.ocp_nlp_dims_get_from_attr.argtypes = \
            [c_void_p, c_void_p, c_void_p, c_int, c_char_p]
        self.shared_lib.ocp_nlp_dims_get_from_attr.restype = c_int

        dims = self.shared_lib.ocp_nlp_dims_get_from_attr(self.nlp_config, \
            self.nlp_dims, self.nlp_out, start_stage_, field)

        out = np.ascontiguousarray(np.zeros((end_stage_ - start_stage_, dims)), dtype=np.float64)
        out_data = cast(out.ctypes.data, POINTER(c_double))

        if (field_ in out_fields):
            self.shared_lib.ocp_nlp_out_get_slice.argtypes = \
                [c_void_p, c_void_p, c_void_p, c_int, c_int, c_char_p, c_void_p]
            self.shared_lib.ocp_nlp_out_get_slice(self.nlp_config, \
                self.nlp_dims, self.nlp_out, start_stage_, end_stage_, field, out_data)
        elif field_ in mem_fields:
            self.shared_lib.ocp_nlp_get_at_stage.argtypes = \
                [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
            self.shared_lib.ocp_nlp_get_at_stage(self.nlp_config, \
                self.nlp_dims, self.nlp_solver, start_stage_, end_stage_, field, out_data)

        return out

    def get_cost(self):
        """
        Returns the cost value of the current solution.
        """
        # compute cost internally
        self.shared_lib.ocp_nlp_eval_cost.argtypes = [c_void_p, c_void_p, c_void_p]
        self.shared_lib.ocp_nlp_eval_cost(self.nlp_solver, self.nlp_in, self.nlp_out)

        # create output array
        out = np.ascontiguousarray(np.zeros((1,)), dtype=np.float64)
        out_data = cast(out.ctypes.data, POINTER(c_double))

        # call getter
        self.shared_lib.ocp_nlp_get.argtypes = [c_void_p, c_void_p, c_char_p, c_void_p]

        field = "cost_value".encode('utf-8')
        self.shared_lib.ocp_nlp_get(self.nlp_config, self.nlp_solver, field, out_data)

        return out[0]
