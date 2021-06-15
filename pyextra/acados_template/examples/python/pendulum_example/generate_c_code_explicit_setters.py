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

from acados_template import *
import acados_template as at
from export_ode_model import *
import numpy as np
import scipy.linalg
from ctypes import *

# create render arguments
ocp = AcadosOcp()

# export model
model = export_ode_model()

# set model_name
ocp.model_name = model.name

Tf = 1.0
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu
ny_e = nx
N = 100

# set ocp_nlp_dimensions

ocp.set('dims_nx', nx)
ocp.set('dims_ny', ny)
ocp.set('dims_ny_e', ny_e)
ocp.set('dims_nbx', 0)
ocp.set('dims_nbu', nu)
ocp.set('dims_nu', model.u.size()[0])
ocp.set('dims_N', N)

# set weighting matrices
Q = np.eye(4)
Q[0,0] = 1e3
Q[1,1] = 1e-2
Q[2,2] = 1e3
Q[3,3] = 1e-2

R = np.eye(1)
R[0,0] = 1e-2

ocp.set('cost_W', scipy.linalg.block_diag(Q, R))

Vx = np.zeros((ny, nx))
Vx[0,0] = 1.0
Vx[1,1] = 1.0
Vx[2,2] = 1.0
Vx[3,3] = 1.0

ocp.set('cost_Vx', Vx)

Vu = np.zeros((ny, nu))
Vu[4,0] = 1.0
ocp.set('cost_Vu', Vu)

ocp.set('cost_W_e', Q)

Vx_e = np.zeros((ny_e, nx))
Vx_e[0,0] = 1.0
Vx_e[1,1] = 1.0
Vx_e[2,2] = 1.0
Vx_e[3,3] = 1.0

ocp.set('cost_Vx_e', Vx_e)

ocp.set('cost_yref', np.zeros((ny, )))
ocp.set('cost_yref_e', np.zeros((ny_e, )))

# setting bounds
Fmax = 80.0
ocp.set('constraints_lbu', np.array([-Fmax]))
ocp.set('constraints_ubu', np.array([-Fmax]))
ocp.set('constraints_x0', np.array([0.0, 0.0, 3.14, 0.0])
ocp.set('constraints_idxbu', np.array([0])

# set constants
# ocp.constants['PI'] = 3.1415926535897932

# set QP solver
# ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.set('solver_options_qp_solver', 'FULL_CONDENSING_QPOASES')
ocp.set('solver_options_hessian_approx', 'GAUSS_NEWTON')
ocp.set('solver_options_integrator_type', 'ERK')

# set prediction horizon
ocp.set('solver_options_tf', Tf)
ocp.set('solver_options_nlp_solver_type', 'SQP')

# set header path
ocp.set('acados_include_path', '/usr/local/include')
ocp.set('acados_lib_path', '/usr/local/lib')

# json_layout = acados_ocp2json_layout(ocp)
# with open('acados_layout.json', 'w') as f:
#     json.dump(json_layout, f, default=np_array_to_list)
# exit()

acados_solver = generate_solver(model, ocp, json_file = 'acados_ocp.json')

Nsim = 100

simX = np.ndarray((Nsim, nx))
simU = np.ndarray((Nsim, nu))

for i in range(Nsim):
    status = acados_solver.solve()

    # get solution
    x0 = acados_solver.get(0, "x")
    u0 = acados_solver.get(0, "u")

    for j in range(nx):
        simX[i,j] = x0[j]

    for j in range(nu):
        simU[i,j] = u0[j]

    # update initial condition
    x0 = acados_solver.get(1, "x")

    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)

# plot results
import matplotlib
import matplotlib.pyplot as plt
t = np.linspace(0.0, Tf/N, Nsim)
plt.subplot(2, 1, 1)
plt.step(t, simU, 'r')
plt.title('closed-loop simulation')
plt.ylabel('u')
plt.xlabel('t')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(t, simX[:,2])
plt.ylabel('theta')
plt.xlabel('t')
plt.grid(True)
plt.show()
