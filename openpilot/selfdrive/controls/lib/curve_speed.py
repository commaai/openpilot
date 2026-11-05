import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter


class CurveSpeed:
  """Helps e2e long slow down for a sharp curve in its own plan: only the part of the plan that is too fast is corrected."""
  def __init__(self, dt):
    self.dt = dt
    self.gap = FirstOrderFilter(0.0, 0.2, dt)
    self.reset()

  def reset(self):
    self.gap.x = 0.0
    self.correction = 0.0

  def update(self, model, roll, a_e2e):
    t, v, yaw_rate = (np.array(x) for x in (model.position.t, model.velocity.x, model.orientationRate.z))
    gap = 0.0
    if len(t) == len(v) == len(yaw_rate):
      m = (t > 0.5) & (t < 6.0)
      # bank into the curve lowers the lateral accel the car must make (same sign as clip_curvature)
      a_lat = 2.8 + ACCELERATION_DUE_TO_GRAVITY * np.clip(roll * np.sign(yaw_rate[m]), -0.07, 0.07)
      v_max = np.sqrt(a_lat / np.maximum(np.abs(yaw_rate[m]) / np.maximum(v[m], 1.0), 1e-4))
      # constant extra decel that brings the plan speed down to the curve speed at each plan point
      gap = max(np.max((v[m] - v_max) / t[m], initial=0.0), 0.0)
    # the extra braking changes by at most 2 m/s³
    self.correction = float(np.clip(self.gap.update(gap), self.correction - 2.0 * self.dt, self.correction + 2.0 * self.dt))
    return min(a_e2e, max(a_e2e - self.correction, -2.0))
