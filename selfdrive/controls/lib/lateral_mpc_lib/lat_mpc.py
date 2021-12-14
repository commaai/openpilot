#!/usr/bin/env python3
import os
import numpy as np

from casadi import SX, vertcat, sin, cos
from selfdrive.controls.lib.drive_helpers import LAT_MPC_N as N
from selfdrive.controls.lib.drive_helpers import T_IDXS

if __name__ == '__main__':  # generating code
  from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  # from pyextra.acados_template import AcadosOcpSolverFast
  from selfdrive.controls.lib.lateral_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverFast  # pylint: disable=no-name-in-module, import-error

LAT_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LAT_MPC_DIR, "c_generated_code")
JSON_FILE = "acados_ocp_lat.json"
X_DIM = 6

def gen_lat_model():
  model = AcadosModel()
  model.name = 'lat'

  # set up states & controls
  x_ego = SX.sym('x_ego')
  y_ego = SX.sym('y_ego')
  psi_ego = SX.sym('psi_ego')
  curv_ego = SX.sym('curv_ego')
  v_ego = SX.sym('v_ego')
  rotation_radius = SX.sym('rotation_radius')
  model.x = vertcat(x_ego, y_ego, psi_ego, curv_ego, v_ego, rotation_radius)

  # controls
  curv_rate = SX.sym('curv_rate')
  model.u = vertcat(curv_rate)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  y_ego_dot = SX.sym('y_ego_dot')
  psi_ego_dot = SX.sym('psi_ego_dot')
  curv_ego_dot = SX.sym('curv_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  rotation_radius_dot = SX.sym('rotation_radius_dot')
  model.xdot = vertcat(x_ego_dot, y_ego_dot, psi_ego_dot, curv_ego_dot,
                       v_ego_dot, rotation_radius_dot)

  # dynamics model
  f_expl = vertcat(v_ego * cos(psi_ego) - rotation_radius * sin(psi_ego) * (v_ego * curv_ego),
                   v_ego * sin(psi_ego) + rotation_radius * cos(psi_ego) * (v_ego * curv_ego),
                   v_ego * curv_ego,
                   curv_rate,
                   0.0,
                   0.0)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_lat_mpc_solver():
  ocp = AcadosOcp()
  ocp.model = gen_lat_model()

  Tf = np.array(T_IDXS)[N]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  Q = np.diag([0.0, 0.0])
  QR = np.diag([0.0, 0.0, 0.0])

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  y_ego, psi_ego = ocp.model.x[1], ocp.model.x[2]
  curv_rate = ocp.model.u[0]
  v_ego = ocp.model.x[4]


  ocp.cost.yref = np.zeros((3, ))
  ocp.cost.yref_e = np.zeros((2, ))
  # TODO hacky weights to keep behavior the same
  ocp.model.cost_y_expr = vertcat(y_ego,
                                  ((v_ego +5.0) * psi_ego),
                                  ((v_ego +5.0) * 4 * curv_rate))
  ocp.model.cost_y_expr_e = vertcat(y_ego,
                                    ((v_ego +5.0) * psi_ego))

  # set constraints
  ocp.constraints.constr_type = 'BGH'
  ocp.constraints.idxbx = np.array([2,3])
  ocp.constraints.ubx = np.array([np.radians(90), np.radians(50)])
  ocp.constraints.lbx = np.array([-np.radians(90), -np.radians(50)])
  x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  ocp.constraints.x0 = x0

  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'
  ocp.solver_options.qp_solver_iter_max = 1
  ocp.solver_options.qp_solver_cond_N = N//4

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = np.array(T_IDXS)[:N+1]

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LateralMpc():
  def __init__(self, x0=np.zeros(X_DIM)):
    self.solver = AcadosOcpSolverFast('lat', N, EXPORT_DIR)
    self.reset(x0)

  def reset(self, x0=np.zeros(X_DIM)):
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.yref = np.zeros((N+1, 3))
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:2])

    # Somehow needed for stable init
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
    self.solver.constraints_set(0, "lbx", x0)
    self.solver.constraints_set(0, "ubx", x0)
    self.solver.solve()
    self.solution_status = 0
    self.cost = 0

  def set_weights(self, path_weight, heading_weight, steer_rate_weight):
    W = np.asfortranarray(np.diag([path_weight, heading_weight, steer_rate_weight]))
    for i in range(N):
      self.solver.cost_set(i, 'W', W)
    #TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3/20.)*W[:2,:2])

  def run(self, x0, v_ego, car_rotation_radius, y_pts, heading_pts):
    x0_cp = np.copy(x0)
    self.solver.constraints_set(0, "lbx", x0_cp)
    self.solver.constraints_set(0, "ubx", x0_cp)
    self.yref[:,0] = y_pts
    self.yref[:,1] = heading_pts*(v_ego+5.0)
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:2])

    self.solution_status = self.solver.solve()
    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')
    self.cost = self.solver.get_cost()


if __name__ == "__main__":
  ocp = gen_lat_mpc_solver()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE, build=False)
