import numpy as np

# slow down before a curve in the model's plan, so lateral accel stays below this, even if the model does not slow down
A_LAT_MAX = 2.5        # m/s², for comfort (the ISO limit is 3)
CURVE_T_MIN = 0.5      # s, the curve right in front of the car is too late to slow down for
CURVE_T_MAX = 6.0      # s, how far into the plan to look
MIN_DIST = 3.0         # m, do not divide by tiny distances
TIGHTER_RATIO = 1.2    # a curve point this much tighter than now is a curve ahead
CURVE_ACCEL_MIN = -2.0  # m/s², hardest braking for a curve ahead
IN_CURVE_ACCEL_MIN = -0.5  # m/s², hardest braking in a curve that does not get tighter
CURVE_JERK = 2.0       # m/s³, how fast the braking for a curve can build up
RELEASE_JERK = 0.5     # m/s³, how fast the limit goes back up after a curve
NO_LIMIT = 10.0        # m/s², no curve ahead

# The model often sees how sharp a curve is only late (at night). A second, soft limit starts a gentle slow down
# before the curve gets near A_LAT_MAX, so the car is already slower when the model sees the real curve.
A_LAT_SOFT = 1.8       # m/s²
SOFT_ACCEL_MIN = -0.6  # m/s², most braking for the soft limit


def curve_points(t, v, yaw_rate):
  """Curvature at each plan point, and the mask of the points to slow down for."""
  t, v, yaw_rate = (np.asarray(a, dtype=float) for a in (t, v, yaw_rate))
  curv = np.abs(yaw_rate) / np.maximum(v, 1.0)
  # one plan point is not enough: the curve must also be at the next or previous point
  curv = np.minimum(curv, np.maximum(np.r_[curv[1:], 0.0], np.r_[0.0, curv[:-1]]))
  return curv, (t >= CURVE_T_MIN) & (t <= CURVE_T_MAX)


def get_curve_accel(t, x, v, yaw_rate, v_ego, a_lat_max=A_LAT_MAX):
  """Highest constant accel that keeps v_ego at or below the curve speed limit at each plan point.

  It is positive below the curve speed, so the accel goes down smoothly near the limit (no on/off).
  Only a curve that gets tighter than the curve now gets the full braking. In a curve that does not get tighter
  (we are in it, or we come out of it) the braking is gentle.
  """
  curv_all, m = curve_points(t, v, yaw_rate)
  x = np.asarray(x, dtype=float)
  curv = curv_all[m]
  if len(curv) == 0:
    return NO_LIMIT
  v_max = np.sqrt(a_lat_max / np.maximum(curv, 1e-4))
  need = (v_max ** 2 - v_ego ** 2) / (2 * np.maximum(x[m], MIN_DIST))
  tighter = curv > TIGHTER_RATIO * curv_all[0]
  need = np.where(tighter, np.maximum(need, CURVE_ACCEL_MIN), np.maximum(need, IN_CURVE_ACCEL_MIN))
  return float(min(need.min(), NO_LIMIT))


class CurveSpeed:
  def __init__(self, dt):
    self.dt = dt
    self.reset()

  def reset(self):
    self.a_curve = NO_LIMIT

  def update(self, model, v_ego):
    p = model.position
    if len(p.t) == 0 or len(model.velocity.x) != len(p.t) or len(model.orientationRate.z) != len(p.t):
      self.reset()
      return self.a_curve

    plan = (p.t, p.x, model.velocity.x, model.orientationRate.z, v_ego)
    a_hard = get_curve_accel(*plan, A_LAT_MAX)
    a_soft = max(get_curve_accel(*plan, A_LAT_SOFT), SOFT_ACCEL_MIN)
    a_new = min(a_hard, a_soft)

    if a_new < self.a_curve:
      # braking for a curve builds up with limited jerk
      self.a_curve = max(a_new, min(self.a_curve, 0.0) - CURVE_JERK * self.dt)
    else:
      # after a curve, stop braking quickly, but let the accel come back slowly, so the car does not speed up at once
      release = CURVE_JERK if self.a_curve < 0.0 else RELEASE_JERK
      self.a_curve = min(a_new, self.a_curve + release * self.dt)
    return self.a_curve
