import numpy as np

# slow down before a curve in the model's plan, so lateral accel stays below this, even if the model does not slow down
A_LAT_MAX = 3.0        # m/s²
CURVE_T_MIN = 0.5      # s, the curve right in front of the car is too late to slow down for
CURVE_T_MAX = 6.0      # s, how far into the plan to look
MIN_DIST = 3.0         # m, do not divide by tiny distances
TIGHTER_RATIO = 1.2    # a curve point this much tighter than now is a curve ahead
CURVE_ACCEL_MIN = -2.0  # m/s², hardest braking for a curve ahead
IN_CURVE_ACCEL_MIN = -0.5  # m/s², hardest braking in a curve that does not get tighter
CURVE_JERK = 2.0       # m/s³, how fast the braking for a curve can build up
NO_LIMIT = 10.0        # m/s², no curve ahead


def get_curve_accel(t, x, v, yaw_rate, v_ego):
  """Constant accel that brings v_ego down to the curve speed limit at each plan point, the lowest of them.

  Only a curve that gets tighter than the curve now gets the full braking. In a curve that does not get tighter
  (we are in it, or we come out of it) the braking is gentle.
  """
  t, x, v, yaw_rate = (np.asarray(a, dtype=float) for a in (t, x, v, yaw_rate))
  curv_all = np.abs(yaw_rate) / np.maximum(v, 1.0)
  # one plan point is not enough: the curve must also be at the next or previous point
  curv_all = np.minimum(curv_all, np.maximum(np.r_[curv_all[1:], 0.0], np.r_[0.0, curv_all[:-1]]))
  m = (t >= CURVE_T_MIN) & (t <= CURVE_T_MAX)
  curv = curv_all[m]
  v_max = np.sqrt(A_LAT_MAX / np.maximum(curv, 1e-4))
  need = (v_max ** 2 - v_ego ** 2) / (2 * np.maximum(x[m], MIN_DIST))
  tighter = curv > TIGHTER_RATIO * curv_all[0]
  need = np.where(tighter, np.maximum(need, CURVE_ACCEL_MIN), np.maximum(need, IN_CURVE_ACCEL_MIN))
  need = need[v_max < v_ego]
  if len(need) == 0:
    return NO_LIMIT
  return float(need.min())


class CurveSpeed:
  def __init__(self, dt):
    self.dt = dt
    self.a_curve = NO_LIMIT

  def reset(self):
    self.a_curve = NO_LIMIT

  def update(self, model, v_ego):
    p = model.position
    if len(p.t) == 0 or len(model.velocity.x) != len(p.t) or len(model.orientationRate.z) != len(p.t):
      self.a_curve = NO_LIMIT
      return self.a_curve
    a_new = get_curve_accel(p.t, p.x, model.velocity.x, model.orientationRate.z, v_ego)
    # braking for a curve builds up with limited jerk, release is at once
    self.a_curve = max(a_new, min(self.a_curve, 0.0) - CURVE_JERK * self.dt)
    return self.a_curve
