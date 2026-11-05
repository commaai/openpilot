import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter

# Help e2e long slow down for a sharp curve in its own plan. At each plan point, compare the plan speed with the
# curve speed for A_LAT_MAX. Only the gap is corrected, on top of the model's own accel: when the model slows down
# by itself, the gap and the correction go away.
A_LAT_MAX = 2.8              # m/s², after the bank. Below the 3 m/s² limit of clip_curvature
CURVE_T_MIN = 0.5            # s, the curve right in front of the car is too late to slow down for
CURVE_T_MAX = 6.0            # s, how far into the plan to look
BANK_MAX = np.radians(4.0)   # limit the bank credit, so a bad roll estimate cannot turn off the curve limit
A_LAT_MIN = 0.5              # m/s², lowest lateral accel limit after the bank
COAST_GAP = 0.3              # m/s², up to this gap, the acceleration goes down smoothly to 0
CURVE_ACCEL_MIN = -2.0       # m/s², hardest braking for a curve
CURVE_JERK = 2.0             # m/s³, how fast the correction starts and ends
NO_LIMIT = 10.0              # m/s², no curve ahead
# the gap from the plan is noisy: filter it, and use two thresholds so the limit does not go on and off
GAP_ON = 0.05                # m/s²
GAP_OFF = 0.01               # m/s²

# After the curve limit slowed the car, the model is often slow to accelerate again. Give its accel a small boost
# when the curve ends, that fades out over EXIT_TIME.
EXIT_TIME = 5.0              # s after the curve limit ends
EXIT_RAMP = 1.0              # s, the boost goes up to EXIT_GAIN in this time, so there is no step
EXIT_GAIN = 1.3
EXIT_MIN_ACCEL = 0.1         # m/s², the model must ask for at least this much accel
EXIT_LAT_ACCEL = 1.5         # m/s², only when the plan lateral accel in the next 3 s is below this

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
  """Accel limit for a gap: the model's accel minus the gap, and its acceleration goes down to 0 up to COAST_GAP.
  It is continuous in the gap: at a gap of 0 it is the model's accel."""
  coast = max(a_e2e, 0.0) * max(1.0 - gap / COAST_GAP, 0.0) + min(a_e2e, 0.0)
  return max(min(a_e2e - gap, coast), CURVE_ACCEL_MIN)


class CurveSpeed:
  def __init__(self, dt):
    self.dt = dt
    # only for a gap that gets smaller, a bigger gap is used at once
    self.gap = FirstOrderFilter(0.0, 0.5, dt)
    self.reset()

  def reset(self):
    self.a_curve = NO_LIMIT
    self.time_since_limit = np.inf
    self.lat_accel_ahead = 0.0
    self.gap.x = 0.0
    self.active = False

  def update(self, model, roll, a_e2e):
    position = model.position
    n = len(position.t)
    if n == 0 or len(model.velocity.x) != n or len(model.orientationRate.z) != n:
      self.reset()
      return self.a_curve

    t, v, yaw_rate = (np.asarray(a, dtype=float) for a in (position.t, model.velocity.x, model.orientationRate.z))
    near = (t >= CURVE_T_MIN) & (t <= 3.0)
    self.lat_accel_ahead = float(np.max(np.abs(v[near] * yaw_rate[near]))) if near.sum() > 0 else 0.0

    yaw_rate_std = model.orientationRate.zStd if len(model.orientationRate.zStd) == n else np.zeros(n)
    gap = get_curve_gap(position.t, model.velocity.x, model.orientationRate.z, yaw_rate_std, roll)
    # a bigger gap is used at once (no delay before braking), a smaller gap comes down slowly
    if gap > self.gap.x:
      self.gap.x = gap
    else:
      self.gap.update(gap)
    if self.active:
      self.active = self.gap.x > GAP_OFF
    else:
      self.active = self.gap.x > GAP_ON
    a_target = get_curve_accel(a_e2e, self.gap.x) if self.active else NO_LIMIT

    # the correction starts from the model's own accel and changes with limited jerk
    a_prev = a_e2e if self.a_curve >= NO_LIMIT else self.a_curve
    step = CURVE_JERK * self.dt
    if a_target < NO_LIMIT:
      self.a_curve = float(np.clip(a_target, a_prev - step, a_prev + step))
    elif a_prev + step < a_e2e:
      self.a_curve = a_prev + step
    else:
      self.a_curve = NO_LIMIT
    self.time_since_limit = 0.0 if self.a_curve < NO_LIMIT else self.time_since_limit + self.dt
    return self.a_curve

  def exit_accel(self, a_e2e):
    """The model's accel with the curve exit boost. Only when it accelerates, after the curve limit, as the curve ends."""
    if self.a_curve < NO_LIMIT or self.time_since_limit > EXIT_TIME or a_e2e <= EXIT_MIN_ACCEL:
      return a_e2e
    if self.lat_accel_ahead > EXIT_LAT_ACCEL:
      return a_e2e
    return a_e2e * float(np.interp(self.time_since_limit, [0.0, EXIT_RAMP, EXIT_TIME], [1.0, EXIT_GAIN, 1.0]))
