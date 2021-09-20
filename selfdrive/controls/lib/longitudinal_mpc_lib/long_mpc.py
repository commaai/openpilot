#!/usr/bin/env python3
import os
import numpy as np

from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.drive_helpers import LON_MPC_N as N
from selfdrive.modeld.constants import T_IDXS

from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat


X_DIM = 3

LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = "acados_ocp_long.json"


def desired_follow_distance(v_ego, v_lead):
  TR = 1.8
  G = 9.81
  return (v_ego * TR - (v_lead - v_ego) * TR + v_ego * v_ego / (2 * G) - v_lead * v_lead / (2 * G)) + 4.0


def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, v_ego):
  MIN_ACCEL = -3.5
  min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-MIN_ACCEL * 2)
  if x_lead < min_x_lead:
    x_lead = min_x_lead

  output = np.zeros((N+1, 2))
  output[0,0] = x_lead
  output[0,1] = v_lead
  a_lead_0 = a_lead
  for i in range(1, N+1):
    dt = T_IDXS[i] - T_IDXS[i-1]
    output[i, 0] = output[i-1, 0] + dt * output[i-1, 1]
    output[i, 1] = max(0.0, output[i-1, 1] + dt * a_lead)
    a_lead = a_lead_0 * np.exp(-a_lead_tau * (T_IDXS[i]**2)/2.)
  return output


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

  # live parameters
  x_lead_0 = SX.sym('x_lead_0')
  v_lead_0 = SX.sym('v_lead_0')
  x_lead_1 = SX.sym('x_lead_1')
  v_lead_1 = SX.sym('v_lead_1')
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  model.p = vertcat(a_min, a_max,
                    x_lead_0, v_lead_0,
                    x_lead_1, v_lead_1)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_mpc_solver():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[N]

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
  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_lead_0, v_lead_0 = ocp.model.p[2], ocp.model.p[3]
  x_lead_1, v_lead_1 = ocp.model.p[4], ocp.model.p[5]

  ocp.cost.yref = np.zeros((4, ))
  ocp.cost.yref_e = np.zeros((3, ))
  ocp.model.cost_y_expr = vertcat(x_ego, v_ego, a_ego, j_ego)
  ocp.model.cost_y_expr_e = vertcat(x_ego, v_ego, a_ego)

  lead_0_x_err = x_lead_0 - x_ego - desired_follow_distance(v_ego, v_lead_0)
  lead_1_x_err = x_lead_1 - x_ego - desired_follow_distance(v_ego, v_lead_1)


  constraints = vertcat(v_ego,
                        a_ego - a_min,
                        a_max - a_ego,
                        lead_0_x_err/ (.05 + v_ego/20.),
                        lead_1_x_err/ (.05 + v_ego/20.))
  ocp.model.con_h_expr = constraints
  ocp.model.con_h_expr_e = constraints

  x0 = np.array([0.0, 0.0, 0.0])
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, 0.0, 0.0])

  l2_penalty = 1.0
  l1_penalty = 0.0
  weights = np.array([1e4, 1e4, 1e4, .1, .1])
  ocp.cost.Zl = l2_penalty * weights
  ocp.cost.zl = l1_penalty * weights
  ocp.cost.Zu = 0.0 * weights
  ocp.cost.zu = 0.0 * weights

  ocp.constraints.lh = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
  ocp.constraints.lh_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
  ocp.constraints.uh = np.array([1e3, 1e3, 1e3, 1e6, 1e6])
  ocp.constraints.uh_e = np.array([1e3, 1e3, 1e3, 1e6, 1e6])
  ocp.constraints.idxsh = np.array([0,1,2,3,4])

  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'
  ocp.solver_options.qp_solver_iter_max = 10

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
    self.accel_limit_arr = np.zeros((N+1, 2))
    self.accel_limit_arr[:,0] = -1.2
    self.accel_limit_arr[:,1] = 1.2
    self.x0 = np.zeros(3)
    self.constraint_cost = np.tile(np.array([1e4, 1e4, 1e4, .0, .0]), (N+1,1))
    self.reset()
    self.new_lead = False
    self.prev_lead_status = False
    self.prev_lead_x = 0.0

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
    self.solver.cost_set(N, 'W', (3/20.)*W[:X_DIM,:X_DIM])

  def set_accel_limits(self, min_a, max_a):
    self.accel_limit_arr[:,0] = min_a
    self.accel_limit_arr[:,1] = max_a

  def set_cur_state(self, v, a):
    if abs(self.x0[1] - v) > 1.:
      self.x0 = np.array([0, v, a])
      self.reset()
    else:
      self.x0 = np.array([0, v, a])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

  def update(self, carstate, radarstate, v_cruise):
    v_ego = carstate.vEgo
    v_cruise_clipped = np.clip(v_cruise, self.x0[1] - 10., self.x0[1] + 10.0)
    poss = v_cruise_clipped * np.array(T_IDXS[:N+1])
    speeds = v_cruise_clipped * np.ones(N+1)
    accels = np.zeros(N+1)
    yref = np.column_stack([poss, speeds, accels, np.zeros(N+1)])

    lead_0 = radarstate.leadOne
    if lead_0.status:
      lead_0_arr = extrapolate_lead(lead_0.dRel, lead_0.vLead, lead_0.aLeadK, lead_0.aLeadTau, v_ego)
      if not self.prev_lead_status or abs(lead_0.dRel - self.prev_lead_x) > 2.5:
        self.new_lead = True
      else:
        self.new_lead = False
      self.prev_lead_x = lead_0.dRel
    else:
      lead_0_arr = extrapolate_lead(100, v_ego + 10, 0.0, 0.0, v_ego)
    self.prev_lead_status = lead_0.status

    lead_1 = radarstate.leadTwo
    if lead_1.status:
      lead_1_arr = extrapolate_lead(lead_1.dRel, lead_1.vLead, lead_1.aLeadK, lead_1.aLeadTau, v_ego)
    else:
      lead_1_arr = extrapolate_lead(100, v_ego + 10, 0.0, 0.0, v_ego)

    self.lead_status = lead_0.status or lead_1.status
    if self.lead_status:
      self.accel_limit_arr[:,0] = -3.5

    params = np.concatenate([self.accel_limit_arr, lead_0_arr, lead_1_arr], axis=1)
    for i in range(N+1):
      self.solver.set_param(i, params[i])

    self.solver.cost_set_slice(0, N, "yref", yref[:N])
    self.solver.cost_set(N, "yref", yref[N][:3])

    self.solution_status = self.solver.solve()
    self.solver.fill_in_slice(0, N+1, 'x', self.x_sol)
    self.solver.fill_in_slice(0, N, 'u', self.u_sol)
    self.cost = self.solver.get_cost()
    #self.solver.print_statistics()

    self.v_solution = list(self.x_sol[:,1])
    self.a_solution = list(self.x_sol[:,2])
    self.j_solution = list(self.u_sol[:,0])

    # Reset if NaN or goes through lead car
    nans = np.any(np.isnan(self.x_sol))

    t = sec_since_boot()
    if nans or self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal model mpc reset - nans")
      self.reset()


if __name__ == "__main__":
  ocp = gen_long_mpc_solver()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE, build=False)
