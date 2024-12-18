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

from .acados_model import AcadosModel
from .acados_ocp import AcadosOcp, AcadosOcpConstraints, AcadosOcpCost, AcadosOcpDims, AcadosOcpOptions
from .acados_sim import AcadosSim, AcadosSimDims, AcadosSimOpts
from .acados_ocp_solver import AcadosOcpSolver, get_simulink_default_opts, ocp_get_default_cmake_builder
from .acados_sim_solver import AcadosSimSolver, sim_get_default_cmake_builder
from .utils import print_casadi_expression, get_acados_path, get_python_interface_path, \
    get_tera_exec_path, get_tera, check_casadi_version, acados_dae_model_json_dump, \
    casadi_length, make_object_json_dumpable, J_to_idx, get_default_simulink_opts

from .zoro_description import ZoroDescription, process_zoro_description
