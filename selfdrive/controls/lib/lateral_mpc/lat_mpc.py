#!/usr/bin/env python3
import os
import sys
import json
from common.basedir import BASEDIR

# TODO: clean this up
acados_path = os.path.join(BASEDIR, "phonelibs/acados/x86_64")
os.environ["TERA_PATH"] = os.path.join(acados_path, "t_renderer")
sys.path.append(os.path.join(BASEDIR, "pyextra"))

import pyextra.acados_template as acados_template
import numpy as np
from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, sin, cos

from selfdrive.controls.lib.drive_helpers import LAT_MPC_N as N
from selfdrive.modeld.constants import index_function


IDX_N = 33
T_IDXS = np.array([index_function(idx, max_val=10.0) for idx in range(IDX_N)], dtype=np.float64)

LAT_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = "acados_ocp_lat.json"


# TODO: fix this up in AcadosOcpSolver
def get_default_simulink_options():
  template_dir = os.path.dirname(acados_template.__file__)
  with open(os.path.join(template_dir, 'simulink_default_opts.json')) as f:
    return json.load(f)


# TODO: move to acados_template?
def generate_code(acados_ocp, json_file):
  from pyextra.acados_template.acados_ocp_solver import make_ocp_dims_consistent, set_up_imported_gnsf_model, \
                                                remove_x0_elimination, ocp_generate_external_functions, \
                                                ocp_formulation_json_dump, ocp_render_templates

  # make dims consistent
  make_ocp_dims_consistent(acados_ocp)

  # module dependent post processing
  if acados_ocp.solver_options.integrator_type == 'GNSF':
    set_up_imported_gnsf_model(acados_ocp)

  if acados_ocp.solver_options.qp_solver == 'PARTIAL_CONDENSING_QPDUNES':
    remove_x0_elimination(acados_ocp)

  # set integrator time automatically
  acados_ocp.solver_options.Tsim = acados_ocp.solver_options.time_steps[0]

  # generate external functions
  ocp_generate_external_functions(acados_ocp, acados_ocp.model)

  # dump to json
  ocp_formulation_json_dump(acados_ocp, get_default_simulink_options(), json_file)

  # render templates
  ocp_render_templates(acados_ocp, json_file)


def gen_lat_model():
  model = AcadosModel()
  model.name = 'lat'

  # set up states & controls
  x_ego = SX.sym('x_ego')
  y_ego = SX.sym('y_ego')
  psi_ego = SX.sym('psi_ego')
  curv_ego = SX.sym('curv_ego')
  model.x = vertcat(x_ego, y_ego, psi_ego, curv_ego)

  # controls
  curv_rate = SX.sym('curv_rate')
  model.u = vertcat(curv_rate)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  y_ego_dot = SX.sym('y_ego_dot')
  psi_ego_dot = SX.sym('psi_ego_dot')
  curv_ego_dot = SX.sym('curv_ego_dot')
  model.xdot = vertcat(x_ego_dot, y_ego_dot, psi_ego_dot, curv_ego_dot)

  # live parameters
  rotation_radius = SX.sym('rotation_radius')
  v_ego = SX.sym('v_ego')
  model.p = vertcat(v_ego, rotation_radius)

  # dynamics model
  f_expl = vertcat(v_ego * cos(psi_ego) - rotation_radius * sin(psi_ego) * (v_ego * curv_ego),
                   v_ego * sin(psi_ego) + rotation_radius * cos(psi_ego) * (v_ego * curv_ego),
                   v_ego * curv_ego,
                   curv_rate)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_lat_mpc_solver():
  ocp = AcadosOcp()
  ocp.model = gen_lat_model()

  N = 16
  Tf = T_IDXS[N]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  Q = np.diag([1., 1.])
  QR = np.diag([1., 1., 1.])

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  y_ego, psi_ego = ocp.model.x[1], ocp.model.x[2]
  curv_rate = ocp.model.u[0]
  v_ego = ocp.model.p[0]


  ocp.cost.yref = np.zeros((3, ))
  ocp.cost.yref_e = np.zeros((2, ))
  # TODO hacky weights to keep behavior the same
  ocp.model.cost_y_expr = vertcat(y_ego,
                                  ((v_ego +5.0) * psi_ego),
                                  ((v_ego +5.0) * 4 * curv_rate))
  ocp.model.cost_y_expr_e = vertcat(y_ego,
                                    ((2.*v_ego +5.0) * psi_ego))
  ocp.parameter_values = np.array([0., .0])

  # set constraints
  ocp.constraints.constr_type = 'BGH'
  ocp.constraints.idxbx = np.array([2,3])
  ocp.constraints.ubx = np.array([np.radians(90), np.radians(50)])
  ocp.constraints.lbx = np.array([-np.radians(90), -np.radians(50)])
  x0 = np.array([0.0, -1.0, 0.0, 0.0])
  ocp.constraints.x0 = x0

  ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'
  ocp.solver_options.qp_solver_iter_max = 100
  ocp.solver_options.qp_solver_cond_N = N

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS[:N+1]

  ocp.code_export_directory = os.path.join(LAT_MPC_DIR, "c_generated_code")
  return ocp


class LateralMpc():
  def __init__(self):
    ocp = gen_lat_mpc_solver()
    self.solver = AcadosOcpSolver(ocp, json_file=JSON_FILE, build=False,
                                  simulink_opts=get_default_simulink_options())
    self.x_sol = np.zeros((N+1, 4))
    self.u_sol = np.zeros((N))

  def set_weights(self, path_weight, heading_weight, steer_rate_weight):
    W = np.diag([path_weight, heading_weight, steer_rate_weight])
    for i in range(N):
      self.solver.cost_set(i, 'W', W)
    # TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3/20.)*W[:2,:2])

  def run(self, x0, v_ego, car_rotation_radius, y_pts, heading_pts):
    self.solver.constraints_set(0, "lbx", x0)
    self.solver.constraints_set(0, "ubx", x0)
    yref = np.column_stack([y_pts, heading_pts, np.zeros(N+1)])
    p = np.array([v_ego, car_rotation_radius])
    for i in range(N):
      self.solver.set(i, "p", p)
      self.solver.set(i, "yref", yref[i])
    self.solver.set(N, "yref", yref[N][:2])

    #status = self.solver.solve()
    self.solver.solve()

    self.x_sol = np.array([self.solver.get(i, 'x') for i in range(N+1)])
    self.u_sol = np.array([self.solver.get(i, 'u') for i in range(N)])
    self.cost = self.solver.get_cost()

if __name__ == "__main__":
  ocp = gen_lat_mpc_solver()
  generate_code(ocp, json_file=JSON_FILE)
