#!/usr/bin/env python3
import time
import numpy as np
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.hpipm import solve_qp

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
MPC_SOURCES = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)

X_DIM = 3
PARAM_DIM = 6
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 3.
X_EGO_COST = 0.
V_EGO_COST = 0.
A_EGO_COST = 0.
J_EGO_COST = 5.
A_CHANGE_COST = 200.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
QP_FAILURE = 4  # acados status for a failed QP solve

# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
MIN_X_LEAD_FACTOR = 0.5

def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.0
  elif personality==log.LongitudinalPersonality.standard:
    return 1.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.5
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality in (log.LongitudinalPersonality.relaxed, log.LongitudinalPersonality.standard):
    return 1.45
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.25
  else:
    raise NotImplementedError("Longitudinal personality not supported")

def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

# dynamics model: triple integrator with jerk as input, discretized exactly on the shooting nodes.
# The states are condensed out of the QP: x_k = TU[k] @ u + TX[k] @ x_0
DT = np.diff(T_IDXS)
TU = np.zeros((N+1, X_DIM, N))
TX = np.zeros((N+1, X_DIM, X_DIM))
TX[0] = np.eye(X_DIM)
for k in range(N):
  A = np.array([[1., DT[k], DT[k]**2 / 2], [0., 1., DT[k]], [0., 0., 1.]])
  TU[k+1] = A @ TU[k]
  TU[k+1, :, k] = [DT[k]**3 / 6, DT[k]**2 / 2, DT[k]]
  TX[k+1] = A @ TX[k]

# Constraints on speed, acceleration and desired distance to the obstacle, which is
# treated as a slack constraint so it behaves like an asymmetrical cost:
#   0 <= [v_ego, a_ego - a_min, a_max - a_ego, danger distance] <= CONSTR_MAX
# The rows of their jacobian that don't depend on the iterate:
CONSTR_MAX = 1e4
C_CONST = np.stack([TU[:N, 1], TU[:N, 2], -TU[:N, 2], np.zeros((N, N))], axis=1)


def gen_long_qp(x_sol, u_sol, x0, slacks, p, W, Zl):
  """
  The Gauss-Newton QP acados builds for one SQP-RTI step from the current iterate (no shift),
  in deltas from the iterate, for solve_qp. Stage costs and slack penalties are scaled by the
  shooting interval.
  """
  a_min, a_max, x_obstacle, a_prev, t_follow, lead_danger_factor = p.T
  x_ego, v_ego = x_sol[:, 0], x_sol[:, 1]
  u = u_sol[:, 0]

  w_obstacle = W[:, 0] * np.append(DT, 1.)
  w_ego = W[:, 1:4] * np.append(DT, 1.)[:, None]
  w_a_change = W[:N, 4] * DT
  w_jerk = W[:N, 5] * DT
  Z = np.zeros((2, CONSTR_DIM*N))
  Z[0] = (DT[:, None] * Zl).ravel()

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep:
  #   y = ((x_obstacle - x_ego) - desired_dist_comfort) / (v_ego + 10.)
  # The danger zone constraint is the same with lead_danger_factor * desired_dist_comfort.
  # Both are linearized in x_ego and v_ego.
  den = v_ego + 10.
  dd_dv = v_ego / COMFORT_BRAKE + t_follow
  desired_dist_comfort = get_safe_obstacle_distance(v_ego, t_follow)
  y = (x_obstacle - x_ego - desired_dist_comfort) / den
  h = (x_obstacle - x_ego - lead_danger_factor * desired_dist_comfort) / den
  dy_dx = -1. / den
  dy_dv = -(dd_dv + y) / den
  dh_dv = -(lead_danger_factor * dd_dv + h) / den

  # states from rolling out the current inputs from x0, offset c from the iterate
  x_roll = TU @ u + TX @ x0
  c = x_roll - x_sol
  y_roll = y + dy_dx * c[:, 0] + dy_dv * c[:, 1]
  h_roll = h + dy_dx * c[:, 0] + dh_dv * c[:, 1]

  # costs: obstacle distance, a_ego - a_prev, j_ego, x_ego, v_ego and a_ego
  J = dy_dx[:N, None] * TU[:N, 0] + dy_dv[:N, None] * TU[:N, 1]
  H = (TU[:N, 2].T * w_a_change) @ TU[:N, 2] + np.diag(w_jerk) + (J.T * w_obstacle[:N]) @ J
  g = J.T @ (w_obstacle[:N] * y_roll[:N]) + TU[:N, 2].T @ (w_a_change * (x_roll[:N, 2] - a_prev[:N])) + w_jerk * u
  for i in range(X_DIM):  # x_ego, v_ego and a_ego
    H += (TU[:N, i].T * w_ego[:N, i]) @ TU[:N, i]
    g += TU[:N, i].T @ (w_ego[:N, i] * x_roll[:N, i])

  # the terminal state stays a QP variable, its a_ego - a_prev weight is always zero
  J_N = np.array([dy_dx[N], dy_dv[N], 0.])
  Q = w_obstacle[N] * np.outer(J_N, J_N) + np.diag(w_ego[N])
  q = w_obstacle[N] * y[N] * J_N + w_ego[N] * x_sol[N]

  # constraints
  C = C_CONST.copy()
  C[:, 3] = dy_dx[:N, None] * TU[:N, 0] + dh_dv[:N, None] * TU[:N, 1]
  h_lin = np.stack([x_roll[:N, 1], x_roll[:N, 2] - a_min[:N], a_max[:N] - x_roll[:N, 2], h_roll[:N]], axis=1).ravel()
  d = np.concatenate([np.stack([-h_lin, h_lin - CONSTR_MAX]) - slacks, -slacks])
  return (H, g, Q, q, TU[N], c[N], C.reshape(-1, N), Z, Z * slacks, d), c


class LongitudinalMpc:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.reset()
    self.source = LongitudinalPlanSource.cruise

  def reset(self):
    # the SQP iterate is x_sol, u_sol and the constraint slacks
    self.slacks = np.zeros((2, CONSTR_DIM*N))

    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.j_solution = np.zeros(N)
    self.a_prev = np.array(self.a_solution)
    self.params = np.zeros((N+1, PARAM_DIM))

    self.last_cloudlog_t = 0
    self.crash_cnt = 0.0
    self.solution_status = 0
    # timers
    self.solve_time = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    self.W = np.tile(cost_weights, (N+1, 1))
    # TODO don't hardcode A_CHANGE_COST idx
    # reduce the cost on (a-a_prev) later in the horizon.
    self.W[:, 4] *= np.interp(T_IDXS, [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])

    # Set L2 slack cost on lower bound constraints
    self.Zl = np.array(constraint_cost_weights)

  def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard):
    jerk_factor = get_jerk_factor(personality)
    a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
    cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost, jerk_factor * J_EGO_COST]
    constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    self.set_cost_weights(cost_weights, constraint_cost_weights)

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.:  # probably only helps if v < v_prev
      self.x_sol = np.tile(self.x0, (N+1, 1))

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.present:
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
    min_x_lead = MIN_X_LEAD_FACTOR * (v_ego + v_lead) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv

  def update(self, radarstate, personality=log.LongitudinalPersonality.standard):
    t_follow = get_T_FOLLOW(personality)

    lead_xv_0 = self.process_lead(radarstate.leadOne)
    lead_xv_1 = self.process_lead(radarstate.leadTwo)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])

    x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle])
    self.source = MPC_SOURCES[np.argmin(x_obstacles[0])]

    self.params[:,0] = ACCEL_MIN
    self.params[:,1] = ACCEL_MAX
    self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.a_prev)
    self.params[:,4] = t_follow
    self.params[:,5] = LEAD_DANGER_FACTOR

    self.run()
    if (np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            radarstate.leadOne.modelProb > 0.9):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

  def run(self):
    t = time.perf_counter()
    qp, c = gen_long_qp(self.x_sol, self.u_sol, self.x0, self.slacks, self.params, self.W, self.Zl)
    du, dx_N, ds, failed = solve_qp(*qp)
    self.solution_status = QP_FAILURE if failed else 0

    if not failed:
      dx = TU @ du + c
      dx[N] = dx_N
      self.x_sol = self.x_sol + dx
      self.u_sol = self.u_sol + du[:, None]
      self.slacks = self.slacks + ds
    self.solve_time = time.perf_counter() - t

    self.v_solution = self.x_sol[:,1]
    self.a_solution = self.x_sol[:,2]
    self.j_solution = self.u_sol[:,0]

    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    t = time.monotonic()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()
