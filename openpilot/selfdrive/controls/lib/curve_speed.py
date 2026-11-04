import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY

# Help e2e long slow down for a sharp curve in its own plan. At each plan point, compare the plan speed with the
# curve speed for A_LAT_MAX. Only the gap is corrected, on top of the model's own accel: when the model slows down
# by itself, the gap and the correction go away.
A_LAT_MAX = 2.8              # m/s², after the bank. Below the 3 m/s² limit of clip_curvature
CURVE_T_MIN = 0.5            # s, the curve right in front of the car is too late to slow down for
CURVE_T_MAX = 6.0            # s, how far into the plan to look
BANK_MAX = np.radians(4.0)   # limit the bank credit, so a bad roll estimate cannot turn off the curve limit
A_LAT_MIN = 0.5              # m/s², lowest lateral accel limit after the bank
COAST_GAP = 0.3              # m/s², for a smaller gap, only stop the acceleration
CURVE_ACCEL_MIN = -2.0       # m/s², hardest braking for a curve
CURVE_JERK = 2.0             # m/s³, how fast the correction starts and ends
NO_LIMIT = 10.0              # m/s², no curve ahead

# The model often sees how sharp a curve is only late (e.g. at night), but its yaw rate std is higher before such a
# curve. Take each plan point as this many stds sharper, so the car slows earlier when the model is not sure.
YAW_RATE_STD_K = 1.0


def get_curve_gap(t, v, yaw_rate, yaw_rate_std, roll, a_lat_max=A_LAT_MAX):
  """Extra decel (m/s², 0 or more) that brings the plan speed down to the curve speed at every plan point."""
  t, v, yaw_rate, yaw_rate_std = (np.asarray(a, dtype=float) for a in (t, v, yaw_rate, yaw_rate_std))
  curv = (np.abs(yaw_rate) + YAW_RATE_STD_K * yaw_rate_std) / np.maximum(v, 1.0)
  # one plan point is not enough: the curve must also be at the next or previous point
  curv = np.minimum(curv, np.maximum(np.r_[curv[1:], 0.0], np.r_[0.0, curv[:-1]]))

  m = (t >= CURVE_T_MIN) & (t <= CURVE_T_MAX)
  if m.sum() == 0:
    return 0.0
  # bank into the curve lowers the lateral accel that the steering must make (same sign as clip_curvature)
  bank_into = np.clip(roll * np.sign(yaw_rate[m]), -BANK_MAX, BANK_MAX)
  a_lat = np.maximum(a_lat_max + ACCELERATION_DUE_TO_GRAVITY * np.sin(bank_into), A_LAT_MIN)
  v_max = np.sqrt(a_lat / np.maximum(curv[m], 1e-4))
  # a constant extra decel c for t seconds lowers the plan speed at t by c * t
  gap = (v[m] - v_max) / t[m]
  return float(max(gap.max(), 0.0))


def get_curve_accel(a_e2e, gap):
  """Accel limit for a gap: no limit, stop the acceleration (small gap), or the model's accel minus the gap."""
  if gap <= 0.0:
    return NO_LIMIT
  if gap < COAST_GAP:
    return min(a_e2e, 0.0)
  return max(a_e2e - gap, CURVE_ACCEL_MIN)


class CurveSpeed:
  def __init__(self, dt):
    self.dt = dt
    self.reset()

  def reset(self):
    self.a_curve = NO_LIMIT

  def update(self, model, roll, a_e2e):
    position = model.position
    n = len(position.t)
    if n == 0 or len(model.velocity.x) != n or len(model.orientationRate.z) != n:
      self.reset()
      return self.a_curve

    yaw_rate_std = model.orientationRate.zStd if len(model.orientationRate.zStd) == n else np.zeros(n)
    gap = get_curve_gap(position.t, model.velocity.x, model.orientationRate.z, yaw_rate_std, roll)
    a_target = get_curve_accel(a_e2e, gap)

    # the correction starts from the model's own accel and changes with limited jerk
    a_prev = a_e2e if self.a_curve >= NO_LIMIT else self.a_curve
    step = CURVE_JERK * self.dt
    if a_target < NO_LIMIT:
      self.a_curve = float(np.clip(a_target, a_prev - step, a_prev + step))
    elif a_prev + step < a_e2e:
      self.a_curve = a_prev + step
    else:
      self.a_curve = NO_LIMIT
    return self.a_curve
