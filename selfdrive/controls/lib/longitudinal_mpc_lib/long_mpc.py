#!/usr/bin/env python3
import os
import numpy as np

from common.numpy_fast import clip
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.drive_helpers import LON_MPC_N as N
from selfdrive.modeld.constants import T_IDXS

from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat




LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = "acados_ocp_long.json"


def gen_long_model():
  model = AcadosModel()
  model.name = 'long'

  # set up states & controls
  x_ego = SX.sym('x_ego')
  v_ego = SX.sym('v_ego')
  a_ego = SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_mpc_solver():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = np.array(T_IDXS)[N]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.diag([0.0, 0.0, 0.0, 0.0])
  Q = np.diag([0.0, 0.0, 0.0])

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  ocp.cost.yref = np.zeros((4, ))
  ocp.cost.yref_e = np.zeros((3, ))
  # TODO hacky weights to keep behavior the same
  ocp.model.cost_y_expr = vertcat(x_ego, v_ego, a_ego, j_ego)
  ocp.model.cost_y_expr_e = vertcat(x_ego, v_ego, a_ego)

  # set constraints
  ocp.constraints.constr_type = 'BGH'
  ocp.constraints.idxbx = np.array([0, 1,2])
  ocp.constraints.lbx = np.array([0., 0, -1.2])
  ocp.constraints.ubx = np.array([10000, 100., 1.2])
  ocp.constraints.Jsbx = np.eye(3)
  x0 = np.array([0.0, 0.0, 0.0])
  ocp.constraints.x0 = x0

  l2_penalty = 1.0
  l1_penalty = 0.0
  weights = np.array([0.0, 1e4, 1e4])
  ocp.cost.Zl = l2_penalty * weights
  ocp.cost.Zu = l2_penalty * weights
  ocp.cost.zl = l1_penalty * weights
  ocp.cost.zu = l1_penalty * weights

  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'
  ocp.solver_options.qp_solver_iter_max = 2

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = np.array(T_IDXS)[:N+1]

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc():
  def __init__(self):
    self.solver = AcadosOcpSolver('long', N, EXPORT_DIR)
    self.x_sol = np.zeros((N+1, 3))
    self.u_sol = np.zeros((N, 1))
    self.set_weights()

    self.v_solution = [0.0 for i in range(len(T_IDXS))]
    self.a_solution = [0.0 for i in range(len(T_IDXS))]
    self.j_solution = [0.0 for i in range(len(T_IDXS)-1)]
    self.yref = np.zeros((N+1, 4))
    self.solver.cost_set_slice(0, N, "yref", self.yref[:N])
    self.solver.cost_set(N, "yref", self.yref[N][:3])
    self.T_IDXS = np.array(T_IDXS[:N+1])
    self.min_a = -1.2
    self.max_a = 1.2
    self.mins = np.tile(np.array([0.0, 0.0, self.min_a])[None], reps=(N-1,1))
    self.maxs = np.tile(np.array([0.0, 100.0, self.max_a])[None], reps=(N-1,1))
    self.x0 = np.zeros(3)
    self.reset()

  def reset(self):
    self.last_cloudlog_t = 0
    self.status = True
    self.solution_status = 0
    for i in range(N+1):
      self.solver.set(i, 'x', self.x0)

  def set_weights(self):
    W = np.diag([0.0, 1.0, 0.0, 50.0])
    Ws = np.tile(W[None], reps=(N,1,1))
    self.solver.cost_set_slice(0, N, 'W', Ws, api='old')
    #TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3/20.)*W[:3,:3])

  def set_accel_limits(self, min_a, max_a):
    self.min_a = min_a
    self.max_a = max_a
    self.mins[:,2] = self.min_a
    self.maxs[:,2] = self.max_a
    self.solver.constraints_set_slice(1, N, "lbx", self.mins, api='old')
    self.solver.constraints_set_slice(1, N, "ubx", self.maxs, api='old')

  def set_cur_state(self, v, a):
    self.x0[1] = v
    self.x0[2] = a
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

  def update(self, carstate, model, v_cruise):
    v_cruise_clipped = clip(v_cruise, self.x0[1] - 10., self.x0[1] + 10.0)
    position = v_cruise_clipped * self.T_IDXS
    speed = v_cruise_clipped * np.ones(N+1)
    accel = np.zeros(N+1)
    self.update_with_xva(position, speed, accel)

  def update_with_xva(self, position, speed, accel):
    self.yref[:,0] = position
    self.yref[:,1] = speed
    self.yref[:,2] = accel
    self.solver.cost_set_slice(0, N, "yref", self.yref[:N])
    self.solver.cost_set(N, "yref", self.yref[N][:3])

    self.solution_status = self.solver.solve()
    self.solver.fill_in_slice(0, N+1, 'x', self.x_sol)
    self.solver.fill_in_slice(0, N, 'u', self.u_sol)
    #self.solver.print_statistics()

    self.v_solution = list(self.x_sol[:,1])
    self.a_solution = list(self.x_sol[:,2])
    self.j_solution = list(self.u_sol[:,0])

    t = sec_since_boot()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f'Longitudinal model mpc reset, solution status: {self.solution_status}')
      self.reset()


if __name__ == "__main__":
  ocp = gen_long_mpc_solver()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE, build=False)
