#!/usr/bin/env python3

'''
import os
import numpy as np

from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
#from selfdrive.controls.lib.drive_helpers import LON_MPC_N as N
from selfdrive.modeld.constants import T_IDXS

from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, exp, sqrt


X_DIM = 3
COST_DIM = 4

LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = "acados_ocp_long.json"


MPC_T = list(np.arange(0,1.,.2)) + list(np.arange(1.,10.6,.6))
N = len(MPC_T) - 1


def desired_follow_distance(v_ego, v_lead, TR=1.8):
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
    dt = MPC_T[i] - MPC_T[i-1]
    output[i, 0] = output[i-1, 0] + dt * output[i-1, 1]
    output[i, 1] = max(0.0, output[i-1, 1] + dt * a_lead)
    a_lead = a_lead_0 * np.exp(-a_lead_tau * (MPC_T[i]**2)/2.)
  print(output)
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
  v_max = SX.sym('v_max')
  model.p = vertcat(a_min, a_max, v_max,
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

  Tf = MPC_T[N]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM+1, COST_DIM+1))
  Q = QR[:COST_DIM, :COST_DIM]

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]
  a_min, a_max, v_max = ocp.model.p[0], ocp.model.p[1], ocp.model.p[2]
  x_lead_0, v_lead_0 = ocp.model.p[3], ocp.model.p[4]
  x_lead_1, v_lead_1 = ocp.model.p[5], ocp.model.p[6]
  lead_0_x_err = x_lead_0 - x_ego - desired_follow_distance(v_ego, v_lead_0, TR=1.8)
  lead_1_x_err = x_lead_1 - x_ego - desired_follow_distance(v_ego, v_lead_1, TR=1.8)
  dist_err = (desired_follow_distance(v_ego, v_lead_0, TR=1.8) - (x_lead_0 - x_ego))/(sqrt(v_ego + 0.5) + 0.1)


  ocp.cost.yref = np.zeros((COST_DIM+1, ))
  ocp.cost.yref_e = np.zeros((COST_DIM, ))
  #costs = [(x_ego - x_desired)/10,
  costs = [(lead_0_x_err) / (0.05*v_ego + .5),
           (lead_1_x_err) / (0.05*v_ego + .5),
           exp(.3 * dist_err) - 1.,
           a_ego / (.1*v_ego + 1.),
           j_ego / (.1*v_ego + 1.)]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:COST_DIM])

  constraints = vertcat((v_ego),
                        (a_ego - a_min),
                        (a_max - a_ego),
                        (v_max - v_ego),
                        0.0*lead_0_x_err/ (10.0),
                        0.0*lead_1_x_err/ (10.0))
  ocp.model.con_h_expr = constraints
  ocp.model.con_h_expr_e = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 100.0, 0.0, 0.0, 0.0, 0.0])

  l2_penalty = 0.0
  l1_penalty = 0.0
  weights = np.array([1e4, 1e4, 1e4, 1e3, 0., 0.])
  ocp.cost.Zl = l2_penalty * weights
  ocp.cost.zl = l1_penalty * weights
  ocp.cost.Zu = 0.0 * weights
  ocp.cost.zu = 0.0 * weights

  ocp.constraints.lh = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  ocp.constraints.lh_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  ocp.constraints.uh = np.array([1e3, 1e3, 1e3, 1e3, 1e4, 1e4])
  ocp.constraints.uh_e = np.array([1e3, 1e3, 1e3, 1e3, 1e6, 1e6])
  ocp.constraints.idxsh = np.array([0,1,2,3,4,5])

  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'
  ocp.solver_options.qp_solver_iter_max = 10

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = np.array(MPC_T)[:N+1]

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc():
  def __init__(self):
    self.solver = AcadosOcpSolver('long', N, EXPORT_DIR)
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.set_weights()

    self.v_solution = [0.0 for i in range(len(T_IDXS))]
    self.a_solution = [0.0 for i in range(len(T_IDXS))]
    self.j_solution = [0.0 for i in range(len(T_IDXS)-1)]
    self.yref = np.zeros((N+1, COST_DIM+1))
    self.solver.cost_set_slice(0, N, "yref", self.yref[:N])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_DIM])
    self.T_IDXS = np.array(T_IDXS[:N+1])
    self.accel_limit_arr = np.zeros((N+1, 2))
    self.accel_limit_arr[:,0] = -1.2
    self.accel_limit_arr[:,1] = 1.2
    self.x0 = np.zeros(X_DIM)
    #self.constraint_cost = np.tile(np.array([1e4, 1e4, 1e4, .0, .0]), (N+1,1))
    self.reset()

  def reset(self):
    self.new_lead = False
    self.new_lead = False
    self.prev_lead_status = False
    self.prev_lead_x = 0.0
    self.new_lead1 = False
    self.prev_lead_status1 = False
    self.prev_lead_x1 = 0.0
    self.last_cloudlog_t = 0
    self.status = True
    self.solution_status = 0
    for i in range(N+1):
      self.solver.set(i, 'x', self.x0)

  def set_weights(self):
    W = np.diag([.1, .0, 5.0, 20.0, 40.0])
    Ws = np.tile(W[None], reps=(N,1,1))
    self.solver.cost_set_slice(0, N, 'W', Ws, api='old')
    #TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3/5.)*W[:COST_DIM,:COST_DIM])

  def set_cur_state(self, v, a):
    if abs(self.x0[1] - v) > 1.:
      self.x0 = np.array([0, v, a])
      self.reset()
    else:
      self.x0 = np.array([0, v, a])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

  def init_with_sim(self):
    x_ego, v_ego, a_ego, _ = self.x0
    j_ego = -1.0
    for i in range(N+1):
      self.solver.set(i, 'x', np.array([x_ego, v_ego, a_ego, MPC_T[i]]))
      if i < N:
        self.solver.set(i, 'u', np.array([j_ego]))
        dt = T_IDXS[i+1] - T_IDXS[i]
        a_ego += -dt
        v_ego += a_ego*dt
        x_ego += v_ego*dt
        if a_ego <= self.accel_limit_arr[0,0]:
          a_ego = self.accel_limit_arr[0,0]
          j_ego = 0.
        if v_ego <= 0.0:
          v_ego = 0.0
          a_ego = 0.0
          j_ego = 0.0
'''


import os
import math
import numpy as np

from common.realtime import sec_since_boot
from common.numpy_fast import clip
from selfdrive.swaglog import cloudlog
from selfdrive.modeld.constants import T_IDXS
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from selfdrive.controls.lib.drive_helpers import LON_MPC_N as N
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU

from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat, sqrt, exp

LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = "acados_ocp_long.json"


def desired_follow_distance(v_ego, v_lead):
  TR = 1.8
  G = 9.81
  return (v_ego * TR - (v_lead - v_ego) * TR + v_ego * v_ego / (2 * G) - v_lead * v_lead / (2 * G)) + 4.0


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
  x_lead = SX.sym('x_lead')
  v_lead = SX.sym('v_lead')
  model.p = vertcat(x_lead, v_lead)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_mpc_solver():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = np.array(T_IDXS)[-1]

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

  x_lead, v_lead = ocp.model.p[0], ocp.model.p[1]
  desired_dist = desired_follow_distance(v_ego, v_lead)
  dist_err = (desired_dist - (x_lead - x_ego))/(sqrt(v_ego + 0.5) + 0.1)

  # TODO hacky weights to keep behavior the same
  ocp.model.cost_y_expr = vertcat(exp(.3 * dist_err) - 1.,
                                  ((x_lead - x_ego) - (desired_dist)) / (0.05 * v_ego + 0.5),
                                  a_ego * (.1 * v_ego + 1.0),
                                  j_ego * (.1 * v_ego + 1.0))
  ocp.model.cost_y_expr_e = vertcat(exp(.3 * dist_err) - 1.,
                                  ((x_lead - x_ego) - (desired_dist)) / (0.05 * v_ego + 0.5),
                                  a_ego * (.1 * v_ego + 1.0))
  ocp.parameter_values = np.array([0., .0])

  # set constraints
  ocp.constraints.constr_type = 'BGH'
  ocp.constraints.idxbx = np.array([1,])
  ocp.constraints.lbx = np.array([0,])
  ocp.constraints.ubx = np.array([100.,])
  x0 = np.array([0.0, 0.0, 0.0])
  ocp.constraints.x0 = x0


  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'

  ocp.solver_options.qp_solver_iter_max = 10

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = np.array(T_IDXS)

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc():
  def __init__(self):
    self.solver = AcadosOcpSolver('long', N, EXPORT_DIR)
    self.v_solution = [0.0 for i in range(N)]
    self.a_solution = [0.0 for i in range(N)]
    self.j_solution = [0.0 for i in range(N-1)]
    yref = np.zeros((N+1,4))
    self.solver.cost_set_slice(0, N, "yref", yref[:N])
    self.solver.set(N, "yref", yref[N][:3])
    self.x_sol = np.zeros((N+1, 3))
    self.u_sol = np.zeros((N,1))
    self.lead_xv = np.zeros((N+1,2))
    self.reset()
    self.set_weights()

  def reset(self):
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(3))
    self.last_cloudlog_t = 0
    self.status = False
    self.new_lead = False
    self.prev_lead_status = False
    self.crashing = False
    self.prev_lead_x = 10
    self.solution_status = 0
    self.new_lead1 = False
    self.prev_lead_status1 = False
    self.prev_lead_x1 = 0.0
    self.x0 = np.zeros(3)

  def set_weights(self):
    W = np.diag([MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                 MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK])
    Ws = np.tile(W[None], reps=(N,1,1))
    self.solver.cost_set_slice(0, N, 'W', Ws, api='old')
    #TODO hacky weights to keep behavior the same
    self.solver.cost_set(N, 'W', (3./5.)*W[:3,:3])

  def set_cur_state(self, v, a):
    self.x0[1] = v
    self.x0[2] = a

  def extrapolate_lead(self, x_lead, v_lead, a_lead_0, a_lead_tau):
    self.lead_xv[0, 0], self.lead_xv[0, 1] = x_lead, v_lead
    for i in range(1, N+1):
      dt = T_IDXS[i] - T_IDXS[i-1]
      a_lead = a_lead_0 * math.exp(-a_lead_tau * (T_IDXS[i]**2)/2.)
      x_lead += v_lead * dt
      v_lead += a_lead * dt
      if v_lead < 0.0:
        a_lead = 0.0
        v_lead = 0.0
      self.lead_xv[i, 0], self.lead_xv[i, 1] = x_lead, v_lead

  def init_with_sim(self, v_ego, lead_xv, a_lead_0):
    a_ego = min(0.0, -2 * (v_ego - lead_xv[0,1]) * (v_ego - lead_xv[0,1]) / (2.0 * lead_xv[0,0] + 0.01) + a_lead_0)
    dt =.2
    x_ego = 0.0
    self.solver.set(0, 'x', np.array([x_ego, v_ego, a_ego]))
    for i in range(1, N+1):
      dt = T_IDXS[i] - T_IDXS[i-1]
      v_ego += a_ego * dt
      if v_ego <= 0.0:
        v_ego = 0.0
        a_ego = 0.0
      x_ego += v_ego * dt
      self.solver.set(i, 'x', np.array([x_ego, v_ego, a_ego]))

  def set_accel_limits(self, a, b):
    pass 

  def update(self, carstate, radarstate, v_cruise):
    self.crashing = False
    v_ego = self.x0[1]
    lead = radarstate.leadOne
    self.status = lead.status
    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = max(0.0, lead.vLead)
      a_lead = clip(lead.aLeadK, -5.0, 5.0)

      # MPC will not converge if immidiate crash is expected
      # Clip lead distance to what is still possible to brake for
      MIN_ACCEL = -3.5
      min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-MIN_ACCEL * 2)
      if x_lead < min_x_lead:
        x_lead = min_x_lead
        self.crashing = True

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0

      self.a_lead_tau = lead.aLeadTau
      self.new_lead = False
      self.extrapolate_lead(x_lead, v_lead, a_lead, self.a_lead_tau)
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        self.init_with_sim(v_ego, self.lead_xv, a_lead)
        self.new_lead = True

      self.prev_lead_status = True
      self.prev_lead_x = x_lead
    else:
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU
      self.extrapolate_lead(x_lead, v_lead, a_lead, self.a_lead_tau)
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)
    for i in range(N+1):
      self.solver.set_param(i, self.lead_xv[i])

    self.solution_status = self.solver.solve()
    self.solver.fill_in_slice(0, N+1, 'x', self.x_sol)
    self.solver.fill_in_slice(0, N, 'u', self.u_sol)
    #self.solver.print_statistics()

    self.v_solution = list(self.x_sol[:,1])
    self.a_solution = list(self.x_sol[:,2])
    self.j_solution = list(self.u_sol[:,0])

    # Reset if goes through lead car
    self.crashing = self.crashing or np.sum(self.lead_xv[:,0] - self.x_sol[:,0] < 0) > 0

    t = sec_since_boot()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Long mpc reset, solution_status: %s" % (
                          self.solution_status))
      self.prev_lead_status = False
      self.reset()


if __name__ == "__main__":
  ocp = gen_long_mpc_solver()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE, build=False)
