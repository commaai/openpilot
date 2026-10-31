import time
import numpy as np
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
MPC_SOURCES = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)

X_EGO_OBSTACLE_COST = 3.
J_EGO_COST = 5.
A_CHANGE_COST = 200.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6

# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS = np.array([index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)])
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
MIN_X_LEAD_FACTOR = 0.5

# QP solver: acados SQP-RTI step solved with HPIPM (BALANCE mode + acados overrides)
QP_ITER_MAX = 10
QP_TOL = 1e-3
QP_MU0 = 1.
QP_ALPHA_MIN = 1e-8
QP_REG_PRIM = 1e-15  # HPIPM regularization, matters for the zero-cost upper slacks
QP_TAU_MIN = 1e-16
QP_LAM_MIN = 1e-16
QP_INIT_THR = 0.1
QP_FAILURE = 4


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
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.75
  elif personality==log.LongitudinalPersonality.standard:
    return 1.45
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.25
  else:
    raise NotImplementedError("Longitudinal personality not supported")

def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE


# triple integrator, exact discretization: x_k = TU[k] @ u + TX[k] @ x_0
DT = np.diff(T_IDXS)
TU = np.zeros((N+1, 3, N))
TX = np.zeros((N+1, 3, 3))
TX[0] = np.eye(3)
for k in range(N):
  A = np.array([[1., DT[k], DT[k]**2 / 2], [0., 1., DT[k]], [0., 0., 1.]])
  TU[k+1] = A @ TU[k]
  TU[k+1, :, k] = [DT[k]**3 / 6, DT[k]**2 / 2, DT[k]]
  TX[k+1] = A @ TX[k]

# acados scales stage costs, slack penalties included, by the shooting interval
W_OBSTACLE = X_EGO_OBSTACLE_COST * np.append(DT, 1.)
Z = np.zeros((2, 4*N))
Z[0] = (DT[:, None] * [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]).ravel()

# soft constraints 0 <= h(x) <= H_MAX on [v, a - a_min, a_max - a, danger distance], each with
# slacks [sl, su]. su has zero cost, so HPIPM's centering inflates it every solve.
# Inequalities, one row per type: t = SGN * (C @ u) + SEL @ [sl, su] - d >= 0
H_MAX = 1e4
SGN = np.array([[1.], [-1.], [0.], [0.]])
SEL = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
C_CONST = np.stack([TU[:N, 1], TU[:N, 2], -TU[:N, 2], np.zeros((N, N))], axis=1)


def solve_qp(H, g, Q, q, B, b, C, z, d):
  """
  Replica of HPIPM's IPM as run by acados on the condensed QP (delta formulation, cold start,
  conditional Mehrotra predictor-corrector):
    min 1/2 u'Hu + g'u + 1/2 x'Qx + q'x + 1/2 s'Zs + z's  s.t.  x = Bu + b,  t(u, s) >= 0
  """
  nc = d.size
  Hr = H + QP_REG_PRIM * np.eye(N)
  Qr = Q + QP_REG_PRIM * np.eye(3)
  Zr = Z + QP_REG_PRIM
  BtQB = B.T @ Qr @ B

  u, x, pi = np.zeros(N), np.zeros(3), np.zeros(3)
  lam_t = np.empty((2,) + d.shape)
  lam, t = lam_t
  t[2:] = np.maximum(-d[2:], QP_INIT_THR)
  s = d[2:] + t[2:]
  t[:2] = np.maximum(s - d[:2], QP_INIT_THR)
  lam[:] = QP_MU0 / t
  dlam_t = np.empty_like(lam_t)
  dlam, dt = dlam_t

  def step(rm):
    gam = t_inv * (rm - lam_rd)
    rs = rg_s + gam[:2] + gam[2:]
    du = H_inv @ (rhs - C.T @ (gam[0] - gam[1] - (F * rs).sum(axis=0)))
    Cdu = C @ du
    ds = -(rs + Gs * Cdu) * D_inv
    np.subtract(SGN * Cdu + SEL @ ds, rd, out=dt)
    np.multiply(-t_inv, rm + lam * dt, out=dlam)
    return -1. / min((dlam_t / lam_t).min(), -1.), du, ds

  def mu_step(alpha):
    lt = lam_t + alpha * dlam_t
    return lt[0].ravel() @ lt[1].ravel() / nc

  alpha = 1.
  for it in range(QP_ITER_MAX + 1):
    rg_u = H @ u + g - C.T @ (lam[0] - lam[1]) + B.T @ pi
    rg_s = Z * s + z - lam[:2] - lam[2:]
    rg_x = Q @ x + q - pi
    rb = B @ u + b - x
    rd = d + t - SGN * (C @ u) - SEL @ s
    rm = lam * t
    if not (max(np.abs(rg_u).max(), np.abs(rg_s).max(), np.abs(rg_x).max()) > QP_TOL or np.abs(rb).max() > QP_TOL or
            np.abs(rd).max() > QP_TOL or abs(rm.max() - QP_TAU_MIN) > QP_TOL):
      break
    if it == QP_ITER_MAX or alpha <= QP_ALPHA_MIN:
      break

    # KKT system with slacks and terminal state eliminated
    mu = rm.sum() / nc
    t_inv = 1. / t
    Gam = lam * t_inv
    lam_rd = lam * rd
    D_inv = 1. / (Zr + Gam[:2] + Gam[2:])
    Gs = SGN[:2] * Gam[:2]
    F = Gs * D_inv
    w = (Gam[:2] * (Zr + Gam[2:]) * D_inv).sum(axis=0)
    H_inv = np.linalg.inv(Hr + (C.T * w) @ C + BtQB)
    rhs = -(rg_u + B.T @ (Qr @ rb + rg_x))

    alpha, _, _ = step(rm - QP_TAU_MIN)
    mu_aff = mu_step(alpha)
    sigma_mu = max((mu_aff / mu)**3 * mu, QP_TAU_MIN)
    alpha, du, ds = step(rm + dt * dlam - sigma_mu)
    if mu_step(alpha) > 2. * mu_aff:
      alpha, du, ds = step(rm - sigma_mu)

    a = alpha * ((1. - alpha) * 0.99 + alpha * 0.9999999) if alpha < 1. else alpha
    dx = B @ du + rb
    u = u + a * du
    s = s + a * ds
    x = x + a * dx
    pi = pi + a * (Qr @ dx + rg_x)
    lam_t += a * dlam_t
    np.maximum(lam_t, QP_LAM_MIN, out=lam_t)

  failed = np.isnan(rm).any() or (it < QP_ITER_MAX and alpha <= QP_ALPHA_MIN)
  return u, x, s, it, failed


class LongitudinalMpc:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.reset()
    self.source = LongitudinalPlanSource.cruise

  def reset(self):
    # SQP iterate
    self.x_sol = np.zeros((N+1, 3))
    self.u_sol = np.zeros(N)
    self.slacks = np.zeros((2, 4*N))

    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.j_solution = np.zeros(N)
    self.a_prev = np.array(self.a_solution)
    self.x0 = np.zeros(3)
    self.last_cloudlog_t = 0
    self.crash_cnt = 0.0
    self.solution_status = 0
    self.qp_iter = 0
    self.solve_time = 0.0
    self.set_weights()

  def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard):
    jerk_factor = get_jerk_factor(personality)
    a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
    self.w_a_change = jerk_factor * a_change_cost * np.interp(T_IDXS[:N], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0]) * DT
    self.w_jerk = jerk_factor * J_EGO_COST * DT
    self.H_const = (TU[:N, 2].T * self.w_a_change) @ TU[:N, 2] + np.diag(self.w_jerk)

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
    return x_lead_traj, v_lead_traj

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
    return self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)

  def update(self, radarstate, personality=log.LongitudinalPersonality.standard):
    t_follow = get_T_FOLLOW(personality)

    x_lead_0, v_lead_0 = self.process_lead(radarstate.leadOne)
    x_lead_1, v_lead_1 = self.process_lead(radarstate.leadTwo)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = x_lead_0 + get_stopped_equivalence_factor(v_lead_0)
    lead_1_obstacle = x_lead_1 + get_stopped_equivalence_factor(v_lead_1)
    self.source = MPC_SOURCES[int(lead_1_obstacle[0] < lead_0_obstacle[0])]

    t0 = time.perf_counter()
    qp, c = self.build_qp(np.minimum(lead_0_obstacle, lead_1_obstacle), t_follow)
    du, dx_N, ds, self.qp_iter, failed = solve_qp(*qp)
    self.solution_status = QP_FAILURE if failed else 0
    if not failed:
      dx = TU @ du + c
      dx[N] = dx_N
      self.x_sol = self.x_sol + dx
      self.u_sol = self.u_sol + du
      self.slacks = self.slacks + ds
    self.solve_time = time.perf_counter() - t0

    self.v_solution = self.x_sol[:, 1]
    self.a_solution = self.x_sol[:, 2]
    self.j_solution = self.u_sol
    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    if failed:
      t = time.monotonic()
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()

    if (np.any(x_lead_0[FCW_IDXS] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            radarstate.leadOne.modelProb > 0.9):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

  def build_qp(self, x_obstacle, t_follow):
    # Gauss-Newton QP for the step from the current iterate (SQP-RTI, no shift), x_1..x_{N-1} condensed out
    x, v, a = self.x_sol.T
    den = v + 10.
    dd_dv = v / COMFORT_BRAKE + t_follow
    dist = get_safe_obstacle_distance(v, t_follow)
    y = (x_obstacle - x - dist) / den
    h = (x_obstacle - x - LEAD_DANGER_FACTOR * dist) / den
    dy_dx = -1. / den
    dy_dv = -(dd_dv + y) / den
    dh_dv = -(LEAD_DANGER_FACTOR * dd_dv + h) / den

    # states from rolling out the current inputs from x0, offset c from the iterate
    x_roll = TU @ self.u_sol + TX @ self.x0
    c = x_roll - self.x_sol
    y_roll = y + dy_dx * c[:, 0] + dy_dv * c[:, 1]
    h_roll = h + dy_dx * c[:, 0] + dh_dv * c[:, 1]

    J = dy_dx[:N, None] * TU[:N, 0] + dy_dv[:N, None] * TU[:N, 1]
    H = self.H_const + (J.T * W_OBSTACLE[:N]) @ J
    g = J.T @ (W_OBSTACLE[:N] * y_roll[:N]) + TU[:N, 2].T @ (self.w_a_change * (x_roll[:N, 2] - self.a_prev[:N])) + self.w_jerk * self.u_sol

    # the terminal state stays a QP variable, its cost only has the obstacle term
    J_N = np.array([dy_dx[N], dy_dv[N], 0.])
    Q = W_OBSTACLE[N] * np.outer(J_N, J_N)
    q = W_OBSTACLE[N] * y[N] * J_N

    C = C_CONST.copy()
    C[:, 3] = dy_dx[:N, None] * TU[:N, 0] + dh_dv[:N, None] * TU[:N, 1]
    h_lin = np.stack([x_roll[:N, 1], x_roll[:N, 2] - ACCEL_MIN, ACCEL_MAX - x_roll[:N, 2], h_roll[:N]], axis=1).ravel()
    d = np.concatenate([np.stack([-h_lin, h_lin - H_MAX]) - self.slacks, -self.slacks])
    return (H, g, Q, q, TU[N], c[N], C.reshape(-1, N), Z * self.slacks, d), c
