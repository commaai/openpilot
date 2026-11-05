import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter

MAX_LAT_ACCEL = 2.8      # m/s², lateral accel the curve speed is for, after the road bank
MAX_BANK = 0.07          # rad (about 4°), most bank to give credit for, so a bad roll estimate cannot turn this off
MIN_CURVE_ACCEL = -2.0   # m/s², hardest braking for a curve
MAX_CURVE_JERK = 2.0     # m/s³, how fast the extra braking can change


class CurveSpeed:
  """Helps e2e long slow down for a sharp curve in its own plan: only the part of the plan that is too fast is corrected."""
  def __init__(self, dt):
    self.dt = dt
    self.gap = FirstOrderFilter(0.0, 0.2, dt)
    self.correction = 0.0

  def reset(self):
    self.gap.x = 0.0
    self.correction = 0.0

  def update(self, model, roll, a_e2e):
    plan_t, plan_v, yaw_rate = (np.array(x) for x in (model.position.t, model.velocity.x, model.orientationRate.z))
    gap = 0.0
    if len(plan_t) == len(plan_v) == len(yaw_rate):
      # plan points far enough ahead to slow down for, and close enough to trust
      ahead = (plan_t > 0.5) & (plan_t < 6.0)
      # bank into the curve lowers the lateral accel the car must make (same sign as clip_curvature)
      bank = np.clip(roll * np.sign(yaw_rate[ahead]), -MAX_BANK, MAX_BANK)
      curvature = np.abs(yaw_rate[ahead]) / np.maximum(plan_v[ahead], 1.0)
      v_max = np.sqrt((MAX_LAT_ACCEL + ACCELERATION_DUE_TO_GRAVITY * bank) / np.maximum(curvature, 1e-4))
      # constant extra decel that brings the plan speed down to the curve speed at each plan point
      gap = max(np.max((plan_v[ahead] - v_max) / plan_t[ahead], initial=0.0), 0.0)
    max_step = MAX_CURVE_JERK * self.dt
    self.correction = float(np.clip(self.gap.update(gap), self.correction - max_step, self.correction + max_step))
    return min(a_e2e, max(a_e2e - self.correction, MIN_CURVE_ACCEL))
