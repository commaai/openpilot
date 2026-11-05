import numpy as np

from opendbc.car.lateral import AVERAGE_ROAD_ROLL
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter

MAX_LAT_ACCEL = 2.8  # m/s^2
MIN_CURVE_ACCEL = -2.0  # m/s^2
MAX_CURVE_JERK = 2.0  # m/s^3
COAST_CORRECTION = 0.3  # m/s^2


class CurveSpeed:
  # slows down only as much as the model's plan is too fast for a curve
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
      ahead = (plan_t > 0.5) & (plan_t < 6.0)
      # banked curves allow more speed, up to the bank angle safety allows for
      bank = np.minimum(roll * np.sign(yaw_rate[ahead]), AVERAGE_ROAD_ROLL)
      curvature = np.abs(yaw_rate[ahead]) / np.maximum(plan_v[ahead], 1.0)
      v_max = np.sqrt((MAX_LAT_ACCEL + ACCELERATION_DUE_TO_GRAVITY * bank) / np.maximum(curvature, 1e-4))
      # extra decel needed to hit the curve speed
      gap = max(np.max((plan_v[ahead] - v_max) / plan_t[ahead], initial=0.0), 0.0)
    max_step = MAX_CURVE_JERK * self.dt
    self.correction = float(np.clip(self.gap.update(gap), self.correction - max_step, self.correction + max_step))
    # the model often doesn't slow down later like it plans, so stop accelerating first
    coast = max(a_e2e, 0.0) * np.interp(self.correction, [0.0, COAST_CORRECTION], [1.0, 0.0]) + min(a_e2e, 0.0)
    return min(coast, max(a_e2e - self.correction, MIN_CURVE_ACCEL))
