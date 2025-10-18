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

import os, sys, json
import urllib.request
import shutil
import numpy as np
from casadi import SX, MX, DM, Function, CasadiMeta

ALLOWED_CASADI_VERSIONS = ('3.5.6', '3.5.5', '3.5.4', '3.5.3', '3.5.2', '3.5.1', '3.4.5', '3.4.0')

TERA_VERSION = "0.0.34"

PLATFORM2TERA = {
    "linux": "linux",
    "darwin": "osx",
    "win32": "windows"
}


def get_acados_path():
    ACADOS_PATH = os.environ.get('ACADOS_SOURCE_DIR')
    if not ACADOS_PATH:
        acados_template_path = os.path.dirname(os.path.abspath(__file__))
        acados_path = os.path.join(acados_template_path, '..','..','..')
        ACADOS_PATH = os.path.realpath(acados_path)
        msg = 'Warning: Did not find environment variable ACADOS_SOURCE_DIR, '
        msg += 'guessed ACADOS_PATH to be {}.\n'.format(ACADOS_PATH)
        msg += 'Please export ACADOS_SOURCE_DIR to avoid this warning.'
        print(msg)
    return ACADOS_PATH


def get_python_interface_path():
    ACADOS_PYTHON_INTERFACE_PATH = os.environ.get('ACADOS_PYTHON_INTERFACE_PATH')
    if not ACADOS_PYTHON_INTERFACE_PATH:
        acados_path = get_acados_path()
        ACADOS_PYTHON_INTERFACE_PATH = os.path.join(acados_path, 'interfaces', 'acados_template', 'acados_template')
    return ACADOS_PYTHON_INTERFACE_PATH


def get_tera_exec_path():
    TERA_PATH = os.environ.get('TERA_PATH')
    if not TERA_PATH:
        TERA_PATH = os.path.join(get_acados_path(), 'bin', 't_renderer')
        if os.name == 'nt':
            TERA_PATH += '.exe'
    return TERA_PATH


def check_casadi_version():
    casadi_version = CasadiMeta.version()
    if casadi_version in ALLOWED_CASADI_VERSIONS:
        return
    else:
        msg =  'Warning: Please note that the following versions of CasADi  are '
        msg += 'officially supported: {}.\n '.format(" or ".join(ALLOWED_CASADI_VERSIONS))
        msg += 'If there is an incompatibility with the CasADi generated code, '
        msg += 'please consider changing your CasADi version.\n'
        msg += 'Version {} currently in use.'.format(casadi_version)
        print(msg)


def is_column(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return True
        elif x.ndim == 2 and x.shape[1] == 1:
            return True
        else:
            return False
    elif isinstance(x, (MX, SX, DM)):
        if x.shape[1] == 1:
            return True
        elif x.shape[0] == 0 and x.shape[1] == 0:
            return True
        else:
            return False
    elif x == None or x == []:
        return False
    else:
        raise Exception("is_column expects one of the following types: np.ndarray, casadi.MX, casadi.SX."
                        + " Got: " + str(type(x)))


def is_empty(x):
    if isinstance(x, (MX, SX, DM)):
        return x.is_empty()
    elif isinstance(x, np.ndarray):
        if np.prod(x.shape) == 0:
            return True
        else:
            return False
    elif x == None:
        return True
    elif isinstance(x, (set, list)):
        if len(x)==0:
            return True
        else:
            return False
    else:
        raise Exception("is_empty expects one of the following types: casadi.MX, casadi.SX, "
                        + "None, numpy array empty list, set. Got: " + str(type(x)))


def casadi_length(x):
    if isinstance(x, (MX, SX, DM)):
        return int(np.prod(x.shape))
    else:
        raise Exception("casadi_length expects one of the following types: casadi.MX, casadi.SX."
                        + " Got: " + str(type(x)))


def make_model_consistent(model):
    x = model.x
    xdot = model.xdot
    u = model.u
    z = model.z
    p = model.p

    if isinstance(x, MX):
        symbol = MX.sym
    elif isinstance(x, SX):
        symbol = SX.sym
    else:
        raise Exception("model.x must be casadi.SX or casadi.MX, got {}".format(type(x)))

    if is_empty(p):
        model.p = symbol('p', 0, 0)

    if is_empty(z):
        model.z = symbol('z', 0, 0)

    return model

def get_lib_ext():
    lib_ext = '.so'
    if sys.platform == 'darwin':
        lib_ext = '.dylib'
    elif os.name == 'nt':
        lib_ext = ''

    return lib_ext

def get_tera():
    tera_path = get_tera_exec_path()
    acados_path = get_acados_path()

    if os.path.exists(tera_path) and os.access(tera_path, os.X_OK):
        return tera_path

    repo_url = "https://github.com/acados/tera_renderer/releases"
    url = "{}/download/v{}/t_renderer-v{}-{}".format(
        repo_url, TERA_VERSION, TERA_VERSION, PLATFORM2TERA[sys.platform])

    manual_install = 'For manual installation follow these instructions:\n'
    manual_install += '1 Download binaries from {}\n'.format(url)
    manual_install += '2 Copy them in {}/bin\n'.format(acados_path)
    manual_install += '3 Strip the version and platform from the binaries: '
    manual_install += 'as t_renderer-v0.0.34-X -> t_renderer)\n'
    manual_install += '4 Enable execution privilege on the file "t_renderer" with:\n'
    manual_install += '"chmod +x {}"\n\n'.format(tera_path)

    msg = "\n"
    msg += 'Tera template render executable not found, '
    msg += 'while looking in path:\n{}\n'.format(tera_path)
    msg += 'In order to be able to render the templates, '
    msg += 'you need to download the tera renderer binaries from:\n'
    msg += '{}\n\n'.format(repo_url)
    msg += 'Do you wish to set up Tera renderer automatically?\n'
    msg += 'y/N? (press y to download tera or any key for manual installation)\n'

    if input(msg) == 'y':
        print("Dowloading {}".format(url))
        with urllib.request.urlopen(url) as response, open(tera_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Successfully downloaded t_renderer.")
        os.chmod(tera_path, 0o755)
        return tera_path

    msg_cancel = "\nYou cancelled automatic download.\n\n"
    msg_cancel += manual_install
    msg_cancel += "Once installed re-run your script.\n\n"
    print(msg_cancel)

    sys.exit(1)


def render_template(in_file, out_file, output_dir, json_path, template_glob=None):

    acados_path = os.path.dirname(os.path.abspath(__file__))
    if template_glob is None:
        template_glob = os.path.join(acados_path, 'c_templates_tera', '**', '*')
    cwd = os.getcwd()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    tera_path = get_tera()

    # call tera as system cmd
    os_cmd = f"{tera_path} '{template_glob}' '{in_file}' '{json_path}' '{out_file}'"
    # Windows cmd.exe can not cope with '...', so use "..." instead:
    if os.name == 'nt':
        os_cmd = os_cmd.replace('\'', '\"')

    status = os.system(os_cmd)
    if (status != 0):
        raise Exception(f'Rendering of {in_file} failed!\n\nAttempted to execute OS command:\n{os_cmd}\n\n')

    os.chdir(cwd)


## Conversion functions
def make_object_json_dumpable(input):
    if isinstance(input, (np.ndarray)):
        return input.tolist()
    elif isinstance(input, (SX)):
        return input.serialize()
    elif isinstance(input, (MX)):
        # NOTE: MX expressions can not be serialized, only Functions.
        return input.__str__()
    elif isinstance(input, (DM)):
        return input.full()
    else:
        raise TypeError(f"Cannot make input of type {type(input)} dumpable.")


def format_class_dict(d):
    """
    removes the __ artifact from class to dict conversion
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = format_class_dict(v)

        out_key = k.split('__', 1)[-1]
        out[k.replace(k, out_key)] = v
    return out


def get_ocp_nlp_layout() -> dict:
    python_interface_path = get_python_interface_path()
    abs_path = os.path.join(python_interface_path, 'acados_layout.json')
    with open(abs_path, 'r') as f:
        ocp_nlp_layout = json.load(f)
    return ocp_nlp_layout


def get_default_simulink_opts() -> dict:
    python_interface_path = get_python_interface_path()
    abs_path = os.path.join(python_interface_path, 'simulink_default_opts.json')
    with open(abs_path, 'r') as f:
        simulink_opts = json.load(f)
    return simulink_opts


def J_to_idx(J):
    nrows = J.shape[0]
    idx = np.zeros((nrows, ))
    for i in range(nrows):
        this_idx = np.nonzero(J[i,:])[0]
        if len(this_idx) != 1:
            raise Exception('Invalid J matrix structure detected, ' \
                'must contain one nonzero element per row.')
        if this_idx.size > 0 and J[i,this_idx[0]] != 1:
            raise Exception('J matrices can only contain 1s.')
        idx[i] = this_idx[0]
    return idx


def J_to_idx_slack(J):
    nrows = J.shape[0]
    ncol = J.shape[1]
    idx = np.zeros((ncol, ))
    i_idx = 0
    for i in range(nrows):
        this_idx = np.nonzero(J[i,:])[0]
        if len(this_idx) == 1:
            idx[i_idx] = i
            i_idx = i_idx + 1
        elif len(this_idx) > 1:
            raise Exception('J_to_idx_slack: Invalid J matrix. ' \
                'Found more than one nonzero in row ' + str(i))
        if this_idx.size > 0 and J[i,this_idx[0]] != 1:
            raise Exception('J_to_idx_slack: J matrices can only contain 1s, ' \
                 'got J(' + str(i) + ', ' + str(this_idx[0]) + ') = ' + str(J[i,this_idx[0]]) )
    if not i_idx == ncol:
            raise Exception('J_to_idx_slack: J must contain a 1 in every column!')
    return idx


def acados_dae_model_json_dump(model):

    # load model
    x = model.x
    xdot = model.xdot
    u = model.u
    z = model.z
    p = model.p

    f_impl = model.f_impl_expr
    model_name = model.name

    # create struct with impl_dae_fun, casadi_version
    fun_name = model_name + '_impl_dae_fun'
    impl_dae_fun = Function(fun_name, [x, xdot, u, z, p], [f_impl])

    casadi_version = CasadiMeta.version()
    str_impl_dae_fun = impl_dae_fun.serialize()

    dae_dict = {"str_impl_dae_fun": str_impl_dae_fun, "casadi_version": casadi_version}

    # dump
    json_file = model_name + '_acados_dae.json'
    with open(json_file, 'w') as f:
        json.dump(dae_dict, f, default=make_object_json_dumpable, indent=4, sort_keys=True)
    print("dumped ", model_name, " dae to file:", json_file, "\n")


def set_up_imported_gnsf_model(acados_ocp):

    gnsf = acados_ocp.gnsf_model

    # check CasADi version
    # dump_casadi_version = gnsf['casadi_version']
    # casadi_version = CasadiMeta.version()

    # if not casadi_version == dump_casadi_version:
    #     print("WARNING: GNSF model was dumped with another CasADi version.\n"
    #             + "This might yield errors. Please use the same version for compatibility, serialize version: "
    #             + dump_casadi_version + " current Python CasADi verison: " + casadi_version)
    #     input("Press any key to attempt to continue...")

    # load model
    phi_fun = Function.deserialize(gnsf['phi_fun'])
    phi_fun_jac_y = Function.deserialize(gnsf['phi_fun_jac_y'])
    phi_jac_y_uhat = Function.deserialize(gnsf['phi_jac_y_uhat'])
    get_matrices_fun = Function.deserialize(gnsf['get_matrices_fun'])

    # obtain gnsf dimensions
    size_gnsf_A = get_matrices_fun.size_out(0)
    acados_ocp.dims.gnsf_nx1 = size_gnsf_A[1]
    acados_ocp.dims.gnsf_nz1 = size_gnsf_A[0] - size_gnsf_A[1]
    acados_ocp.dims.gnsf_nuhat = max(phi_fun.size_in(1))
    acados_ocp.dims.gnsf_ny = max(phi_fun.size_in(0))
    acados_ocp.dims.gnsf_nout = max(phi_fun.size_out(0))

    # save gnsf functions in model
    acados_ocp.model.phi_fun = phi_fun
    acados_ocp.model.phi_fun_jac_y = phi_fun_jac_y
    acados_ocp.model.phi_jac_y_uhat = phi_jac_y_uhat
    acados_ocp.model.get_matrices_fun = get_matrices_fun

    # get_matrices_fun = Function([model_name,'_gnsf_get_matrices_fun'], {dummy},...
    #  {A, B, C, E, L_x, L_xdot, L_z, L_u, A_LO, c, E_LO, B_LO,...
    #   nontrivial_f_LO, purely_linear, ipiv_x, ipiv_z, c_LO});
    get_matrices_out = get_matrices_fun(0)
    acados_ocp.model.gnsf['nontrivial_f_LO'] = int(get_matrices_out[12])
    acados_ocp.model.gnsf['purely_linear'] = int(get_matrices_out[13])

    if "f_lo_fun_jac_x1k1uz" in gnsf:
        f_lo_fun_jac_x1k1uz = Function.deserialize(gnsf['f_lo_fun_jac_x1k1uz'])
        acados_ocp.model.f_lo_fun_jac_x1k1uz = f_lo_fun_jac_x1k1uz
    else:
        dummy_var_x1 = SX.sym('dummy_var_x1', acados_ocp.dims.gnsf_nx1)
        dummy_var_x1dot = SX.sym('dummy_var_x1dot', acados_ocp.dims.gnsf_nx1)
        dummy_var_z1 = SX.sym('dummy_var_z1', acados_ocp.dims.gnsf_nz1)
        dummy_var_u = SX.sym('dummy_var_z1', acados_ocp.dims.nu)
        dummy_var_p = SX.sym('dummy_var_z1', acados_ocp.dims.np)
        empty_var = SX.sym('empty_var', 0, 0)

        empty_fun = Function('empty_fun', \
            [dummy_var_x1, dummy_var_x1dot, dummy_var_z1, dummy_var_u, dummy_var_p],
                [empty_var])
        acados_ocp.model.f_lo_fun_jac_x1k1uz = empty_fun

    del acados_ocp.gnsf_model


def idx_perm_to_ipiv(idx_perm):
    n = len(idx_perm)
    vec = list(range(n))
    ipiv = np.zeros(n)

    print(n, idx_perm)
    # import pdb; pdb.set_trace()
    for ii in range(n):
        idx0 = idx_perm[ii]
        for jj in range(ii,n):
            if vec[jj]==idx0:
                idx1 = jj
                break
        tmp = vec[ii]
        vec[ii] = vec[idx1]
        vec[idx1] = tmp
        ipiv[ii] = idx1

    ipiv = ipiv-1 # C 0-based indexing
    return ipiv


def print_casadi_expression(f):
    for ii in range(casadi_length(f)):
        print(f[ii,:])
