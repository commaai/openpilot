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

import os, sys, json
import urllib.request
import shutil
import numpy as np
from casadi import SX, MX, DM, Function, CasadiMeta

ALLOWED_CASADI_VERSIONS = ('3.5.5', '3.5.4', '3.5.3', '3.5.2', '3.5.1', '3.4.5', '3.4.0')

TERA_VERSION = "0.0.34"

def get_acados_path():
    ACADOS_PATH = os.environ.get('ACADOS_SOURCE_DIR')
    if not ACADOS_PATH:
        acados_template_path = os.path.dirname(os.path.abspath(__file__))
        acados_path = os.path.join(acados_template_path, '../../../')
        ACADOS_PATH = os.path.realpath(acados_path)
        msg = 'Warning: Did not find environment variable ACADOS_SOURCE_DIR, '
        msg += 'guessed ACADOS_PATH to be {}.\n'.format(ACADOS_PATH)
        msg += 'Please export ACADOS_SOURCE_DIR to avoid this warning.'
        print(msg)
    return ACADOS_PATH


def get_tera_exec_path():
    TERA_PATH = os.environ.get('TERA_PATH')
    if not TERA_PATH:
        TERA_PATH = os.path.join(get_acados_path(), 'bin/t_renderer')
    return TERA_PATH


platform2tera = {
    "linux": "linux",
    "darwin": "osx",
    "win32": "window.exe"
}


def casadi_version_warning(casadi_version):
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
    elif x == None or x == []:
        return True
    else:
        raise Exception("is_empty expects one of the following types: casadi.MX, casadi.SX, "
                        + "None, numpy array empty list. Got: " + str(type(x)))


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


def get_tera():
    tera_path = get_tera_exec_path()
    acados_path = get_acados_path()

    if os.path.exists(tera_path) and os.access(tera_path, os.X_OK):
        return tera_path

    repo_url = "https://github.com/acados/tera_renderer/releases"
    url = "{}/download/v{}/t_renderer-v{}-{}".format(
        repo_url, TERA_VERSION, TERA_VERSION, platform2tera[sys.platform])

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


def render_template(in_file, out_file, template_dir, json_path):
    cwd = os.getcwd()
    if not os.path.exists(template_dir):
        os.mkdir(template_dir)
    os.chdir(template_dir)

    tera_path = get_tera()

    # setting up loader and environment
    acados_path = os.path.dirname(os.path.abspath(__file__))

    template_glob = acados_path + '/c_templates_tera/*'
    acados_template_path = acados_path + '/c_templates_tera'

    # call tera as system cmd
    os_cmd = "{tera_path} '{template_glob}' '{in_file}' '{json_path}' '{out_file}'".format(
        tera_path=tera_path,
        template_glob=template_glob,
        json_path=json_path,
        in_file=in_file,
        out_file=out_file
    )
    status = os.system(os_cmd)
    if (status != 0):
        raise Exception('Rendering of {} failed! Exiting.\n'.format(in_file))

    os.chdir(cwd)


## Conversion functions
def np_array_to_list(np_array):
    if isinstance(np_array, (np.ndarray)):
        return np_array.tolist()
    elif isinstance(np_array, (SX)):
        return DM(np_array).full()
    elif isinstance(np_array, (DM)):
        return np_array.full()
    else:
        raise(Exception(
            "Cannot convert to list type {}".format(type(np_array))
        ))


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


def acados_class2dict(class_instance):
    """
    removes the __ artifact from class to dict conversion
    """

    d = dict(class_instance.__dict__)
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = format_class_dict(v)

        out_key = k.split('__', 1)[-1]
        out[k.replace(k, out_key)] = v
    return out


def ocp_check_against_layout(ocp_nlp, ocp_dims):
    """
    Check dimensions against layout
    Parameters
    ---------
    ocp_nlp : dict
        dictionary loaded from JSON to be post-processed.

    ocp_dims : instance of AcadosOcpDims
    """

    # load JSON layout
    current_module = sys.modules[__name__]
    acados_path = os.path.dirname(current_module.__file__)
    with open(acados_path + '/acados_layout.json', 'r') as f:
        ocp_nlp_layout = json.load(f)

    ocp_check_against_layout_recursion(ocp_nlp, ocp_dims, ocp_nlp_layout)
    return


def ocp_check_against_layout_recursion(ocp_nlp, ocp_dims, layout):

    for key, item in ocp_nlp.items():

        try:
            layout_of_key = layout[key]
        except KeyError:
            raise Exception("ocp_check_against_layout_recursion: field" \
                            " '{0}' is not in layout but in OCP description.".format(key))

        if isinstance(item, dict):
            ocp_check_against_layout_recursion(item, ocp_dims, layout_of_key)

        if 'ndarray' in layout_of_key:
            if isinstance(item, int) or isinstance(item, float):
                item = np.array([item])
        if isinstance(item, (list, np.ndarray)) and (layout_of_key[0] != 'str'):
            dim_layout = []
            dim_names = layout_of_key[1]

            for dim_name in dim_names:
                dim_layout.append(ocp_dims[dim_name])

            dims = tuple(dim_layout)

            item = np.array(item)
            item_dims = item.shape
            if len(item_dims) != len(dims):
                raise Exception('Mismatching dimensions for field {0}. ' \
                    'Expected {1} dimensional array, got {2} dimensional array.' \
                        .format(key, len(dims), len(item_dims)))

            if np.prod(item_dims) != 0 or np.prod(dims) != 0:
                if dims != item_dims:
                    raise Exception('acados -- mismatching dimensions for field {0}. ' \
                        'Provided data has dimensions {1}, ' \
                        'while associated dimensions {2} are {3}' \
                            .format(key, item_dims, dim_names, dims))
    return


def J_to_idx(J):
    nrows = J.shape[0]
    idx = np.zeros((nrows, ))
    for i in range(nrows):
        this_idx = np.nonzero(J[i,:])[0]
        if len(this_idx) != 1:
            raise Exception('Invalid J matrix structure detected, ' \
                'must contain one nonzero element per row. Exiting.')
        if this_idx.size > 0 and J[i,this_idx[0]] != 1:
            raise Exception('J matrices can only contain 1s. Exiting.')
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
            raise Exception('J_to_idx_slack: Invalid J matrix. Exiting. ' \
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
        json.dump(dae_dict, f, default=np_array_to_list, indent=4, sort_keys=True)
    print("dumped ", model_name, " dae to file:", json_file, "\n")


def set_up_imported_gnsf_model(acados_formulation):

    gnsf = acados_formulation.gnsf_model

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
    acados_formulation.dims.gnsf_nx1 = size_gnsf_A[1]
    acados_formulation.dims.gnsf_nz1 = size_gnsf_A[0] - size_gnsf_A[1]
    acados_formulation.dims.gnsf_nuhat = max(phi_fun.size_in(1))
    acados_formulation.dims.gnsf_ny = max(phi_fun.size_in(0))
    acados_formulation.dims.gnsf_nout = max(phi_fun.size_out(0))

    # save gnsf functions in model
    acados_formulation.model.phi_fun = phi_fun
    acados_formulation.model.phi_fun_jac_y = phi_fun_jac_y
    acados_formulation.model.phi_jac_y_uhat = phi_jac_y_uhat
    acados_formulation.model.get_matrices_fun = get_matrices_fun

    if "f_lo_fun_jac_x1k1uz" in gnsf:
        f_lo_fun_jac_x1k1uz = Function.deserialize(gnsf['f_lo_fun_jac_x1k1uz'])
        acados_formulation.model.f_lo_fun_jac_x1k1uz = f_lo_fun_jac_x1k1uz
    else:
        dummy_var_x1 = SX.sym('dummy_var_x1', acados_formulation.dims.gnsf_nx1)
        dummy_var_x1dot = SX.sym('dummy_var_x1dot', acados_formulation.dims.gnsf_nx1)
        dummy_var_z1 = SX.sym('dummy_var_z1', acados_formulation.dims.gnsf_nz1)
        dummy_var_u = SX.sym('dummy_var_z1', acados_formulation.dims.nu)
        dummy_var_p = SX.sym('dummy_var_z1', acados_formulation.dims.np)
        empty_var = SX.sym('empty_var', 0, 0)

        empty_fun = Function('empty_fun', \
            [dummy_var_x1, dummy_var_x1dot, dummy_var_z1, dummy_var_u, dummy_var_p],
                [empty_var])
        acados_formulation.model.f_lo_fun_jac_x1k1uz = empty_fun

    del acados_formulation.gnsf_model
