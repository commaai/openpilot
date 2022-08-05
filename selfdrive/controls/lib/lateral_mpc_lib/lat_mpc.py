#!/usr/bin/env python3
import os
import numpy as np

from casadi import SX, vertcat, sin, cos

from common.realtime import sec_since_boot
from selfdrive.controls.lib.drive_helpers import LAT_MPC_N as N
from selfdrive.modeld.constants import T_IDXS

if __name__ == '__main__':  # generating code
  from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from selfdrive.controls.lib.lateral_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython  # pylint: disable=no-name-in-module, import-error

LAT_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LAT_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LAT_MPC_DIR, "acados_ocp_lat.json")
X_DIM = 4
P_DIM = 2
MODEL_NAME = 'lat'
ACADOS_SOLVER_TYPE = 'SQP_RTI'

def gen_lat_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # set up states & controls
  x_ego = SX.sym('x_ego')
  y_ego = SX.sym('y_ego')
  psi_ego = SX.sym('psi_ego')
  curv_ego = SX.sym('curv_ego')
  model.x = vertcat(x_ego, y_ego, psi_ego, curv_ego)

  # parameters
  v_ego = SX.sym('v_ego')
  rotation_radius = SX.sym('rotation_radius')
  model.p = vertcat(v_ego, rotation_radius)

  # controls
  curv_rate = SX.sym('curv_rate')
  model.u = vertcat(curv_rate)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  y_ego_dot = SX.sym('y_ego_dot')
  psi_ego_dot = SX.sym('psi_ego_dot')
  curv_ego_dot = SX.sym('curv_ego_dot')

  model.xdot = vertcat(x_ego_dot, y_ego_dot, psi_ego_dot, curv_ego_dot)

  # dynamics model
  f_expl = vertcat(v_ego * cos(psi_ego) - rotation_radius * sin(psi_ego) * (v_ego * curv_ego),
                   v_ego * sin(psi_ego) + rotation_radius * cos(psi_ego) * (v_ego * curv_ego),
                   v_ego * curv_ego,
                   curv_rate)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_lat_ocp():
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
  v_ego = ocp.model.p[0]

  ocp.parameter_values = np.zeros((P_DIM, ))

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
  x0 = np.zeros((X_DIM,))
  ocp.constraints.x0 = x0

  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_iter_max = 1
  ocp.solver_options.qp_solver_cond_N = 1

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = np.array(T_IDXS)[:N+1]

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LateralMpc():
  def __init__(self, x0=np.zeros(X_DIM)):
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
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
      self.solver.set(i, 'p', np.zeros(P_DIM))
    self.solver.constraints_set(0, "lbx", x0)
    self.solver.constraints_set(0, "ubx", x0)
    self.solver.solve()
    self.solution_status = 0
    self.solve_time = 0.0
    self.cost = 0

  def set_weights(self, path_weight, heading_weight, steer_rate_weight):
    W = np.asfortranarray(np.diag([path_weight, heading_weight, steer_rate_weight]))
    for i in range(N):
      self.solver.cost_set(i, 'W', W)
    #TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3/20.)*W[:2,:2])

  def run(self, x0, p, y_pts, heading_pts):
    x0_cp = np.copy(x0)
    p_cp = np.copy(p)
    self.solver.constraints_set(0, "lbx", x0_cp)
    self.solver.constraints_set(0, "ubx", x0_cp)
    self.yref[:,0] = y_pts
    v_ego = p_cp[0]
    # rotation_radius = p_cp[1]
    self.yref[:,1] = heading_pts*(v_ego+5.0)
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
      self.solver.set(i, "p", p_cp)
    self.solver.set(N, "p", p_cp)
    self.solver.cost_set(N, "yref", self.yref[N][:2])

    t = sec_since_boot()
    self.solution_status = self.solver.solve()
    self.solve_time = sec_since_boot() - t

    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')
    self.cost = self.solver.get_cost()


if __name__ == "__main__":
  ocp = gen_lat_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
