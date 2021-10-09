#!/usr/bin/env python3
import os
import numpy as np

from common.realtime import sec_since_boot
from common.numpy_fast import clip, interp
from common.numpy_helpers import deep_interp_np
from selfdrive.swaglog import cloudlog
from selfdrive.modeld.constants import T_IDXS as T_IDXS_LST_ORIG
from selfdrive.modeld.constants import index_function
from selfdrive.controls.lib.drive_helpers import LON_MPC_N as N
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU

from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import SX, vertcat

LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = "acados_ocp_long.json"

SOURCES = ['lead0', 'lead1', 'cruise']

X_DIM = 3
U_DIM = 1
COST_E_DIM = 3
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_COST = 3.
X_EGO_E2E_COST = 10.
A_EGO_COST = 1.
J_EGO_COST = 10.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .5
LIMIT_COST = 1e6


T_IDXS_LST = [index_function(idx, max_val=10.0) for idx in range(33)]

T_IDXS = np.array(T_IDXS_LST)
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
N = len(T_IDXS) - 1
MIN_ACCEL = -3.5
T_REACT = 1.8
MAX_BRAKE = 9.81


def get_stopped_equivalence_factor(v_lead):
  return T_REACT * v_lead + (v_lead*v_lead) / (2 * MAX_BRAKE)

def get_safe_obstacle_distance(v_ego):
  return 2 * T_REACT * v_ego + (v_ego*v_ego) / (2 * MAX_BRAKE) + 4.0

def desired_follow_distance(v_ego, v_lead):
  return get_safe_obstacle_distance(v_ego) - get_stopped_equivalence_factor(v_lead)


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
  x_obstacle = SX.sym('x_obstacle')
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  model.p = vertcat(a_min, a_max, x_obstacle)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_mpc_solver():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  costs = [((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.),
           x_ego,
           a_ego,
           j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an assymetrical cost.
  constraints = vertcat((v_ego),
                        (a_ego - a_min),
                        (a_max - a_ego),
                        ((x_obstacle - x_ego) - (3/4) * (desired_dist_comfort)) / (v_ego + 10.))
  ocp.model.con_h_expr = constraints
  ocp.model.con_h_expr_e = vertcat(np.zeros(CONSTR_DIM))

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0])

  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.lh_e = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.uh_e = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where the compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = 'SQP_RTI'

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 4

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc():
  def __init__(self, e2e=False):
    self.e2e = e2e
    self.reset()
    self.accel_limit_arr = np.zeros((N+1, 2))
    self.accel_limit_arr[:,0] = -1.2
    self.accel_limit_arr[:,1] = 1.2
    self.source = SOURCES[2]

  def reset(self):
    self.solver = AcadosOcpSolver('long', N, EXPORT_DIR)
    self.v_solution = [0.0 for i in range(N+1)]
    self.a_solution = [0.0 for i in range(N+1)]
    self.j_solution = [0.0 for i in range(N)]
    self.yref = np.zeros((N+1, COST_DIM))
    self.solver.cost_set_slice(0, N, "yref", self.yref[:N])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N,1))
    self.params = np.zeros((N+1,3))
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
    self.last_cloudlog_t = 0
    self.status = False
    self.crash_cnt = 0.0
    self.solution_status = 0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def set_weights(self):
    if self.e2e:
      self.set_weights_for_xva_policy()
    else:
      self.set_weights_for_lead_policy()

  def set_weights_for_lead_policy(self):
    W = np.diag([X_EGO_COST, 0.0, A_EGO_COST, J_EGO_COST])
    Ws = np.tile(W[None], reps=(N,1,1))
    self.solver.cost_set_slice(0, N, 'W', Ws, api='old')
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array([LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST])
    Zls = np.tile(Zl[None], reps=(N+1,1,1))
    self.solver.cost_set_slice(0, N+1, 'Zl', Zls, api='old')

  def set_weights_for_xva_policy(self):
    W = np.diag([0.0, X_EGO_E2E_COST, 0., J_EGO_COST])
    Ws = np.tile(W[None], reps=(N,1,1))
    self.solver.cost_set_slice(0, N, 'W', Ws, api='old')
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array([LIMIT_COST, LIMIT_COST, LIMIT_COST, 0.0])
    Zls = np.tile(Zl[None], reps=(N+1,1,1))
    self.solver.cost_set_slice(0, N+1, 'Zl', Zls, api='old')

  def set_cur_state(self, v, a):
    if abs(self.x0[1] - v) > 1.:
      self.x0[1] = v
      self.x0[2] = a
      for i in range(0, N+1):
        self.solver.set(i, 'x', self.x0)
    else:
      self.x0[1] = v
      self.x0[2] = a

  def extrapolate_lead(self, x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-MIN_ACCEL * 2)
    x_lead = clip(x_lead, min_x_lead, 1e8)
    v_lead = clip(v_lead, 0.0, 1e8)
    a_lead = clip(a_lead, -10., 5.)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv

  def set_accel_limits(self, min_a, max_a):
    self.cruise_min_a = min_a
    self.cruise_max_a = max_a

  def update(self, carstate, radarstate, v_cruise):
    v_ego = self.x0[1]
    self.status = radarstate.leadOne.status or radarstate.leadTwo.status

    lead_xv_0 = self.process_lead(radarstate.leadOne)
    lead_xv_1 = self.process_lead(radarstate.leadTwo)

    # set accel limits in params
    self.params[:,0] = interp(float(self.status), [0.0, 1.0], [self.cruise_min_a, MIN_ACCEL])
    self.params[:,1] = self.cruise_max_a

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])

    # Fake an obstacle for cruise, this ensures smooth acceleration to set speed
    # when the leads are no factor.
    cruise_lower_bound = v_ego + (3/4) * self.cruise_min_a * T_IDXS
    cruise_upper_bound = v_ego + (3/4) * self.cruise_max_a * T_IDXS
    v_cruise_clipped = np.clip(v_cruise * np.ones(N+1),
                               cruise_lower_bound,
                               cruise_upper_bound)
    cruise_obstacle = T_IDXS*v_cruise_clipped + get_safe_obstacle_distance(v_cruise_clipped)

    x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
    self.source = SOURCES[np.argmin(x_obstacles[0])]
    self.params[:,2] = np.min(x_obstacles, axis=1)

    self.run()
    if (np.any(lead_xv_0[:,0] - self.x_sol[:,0] < CRASH_DISTANCE) and
            radarstate.leadOne.modelProb > 0.9):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0
    #next_state = deep_interp_np(T_IDXS+0.05, T_IDXS, self.x_sol)
    #for i in range(0, N+1):
    #  self.solver.set(i, 'x', next_state[i])

  def update_with_xva(self, x, v, a):
    self.yref[:,1] = x
    self.solver.cost_set_slice(0, N, "yref", self.yref[:N], api='old')
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])
    self.accel_limit_arr[:,0] = -10.
    self.accel_limit_arr[:,1] = 10.
    x_obstacle = 1e5*np.ones((N+1))
    self.params = np.concatenate([self.accel_limit_arr,
                             x_obstacle[:,None]], axis=1)
    self.run()


  def run(self):
    for i in range(N+1):
      self.solver.set_param(i, self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)
    self.solution_status = self.solver.solve()
    self.solver.fill_in_slice(0, N+1, 'x', self.x_sol)
    self.solver.fill_in_slice(0, N, 'u', self.u_sol)

    self.v_solution = np.interp(T_IDXS_LST_ORIG, T_IDXS, self.x_sol[:,1])
    self.a_solution = np.interp(T_IDXS_LST_ORIG, T_IDXS, self.x_sol[:,2])
    self.j_solution = np.interp(T_IDXS_LST_ORIG, T_IDXS[:-1], self.u_sol[:,0])

    t = sec_since_boot()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Long mpc reset, solution_status: %s" % (
                          self.solution_status))
      self.reset()


if __name__ == "__main__":
  ocp = gen_long_mpc_solver()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE, build=False)
