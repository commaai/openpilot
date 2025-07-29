import numpy as np
from abc import abstractmethod, ABC

from openpilot.common.realtime import DT_CTRL


class LatControl(ABC):
  def __init__(self, CP, CI):
    self.sat_count_rate = 1.0 * DT_CTRL
    self.sat_limit = CP.steerLimitTimer
    self.sat_count = 0.
    self.sat_check_min_speed = 10.

    # we define the steer torque scale as [-1.0...1.0]
    self.steer_max = 1.0

  @abstractmethod
  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, calibrated_pose, curvature_limited):
    pass

  def reset(self):
    self.sat_count = 0.

  def _check_saturation(self, saturated, CS, steer_limited_by_safety, curvature_limited):
    # Saturated only if control output is not being limited by car torque/angle rate limits
    if (saturated or curvature_limited) and CS.vEgo > self.sat_check_min_speed and not steer_limited_by_safety and not CS.steeringPressed:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate
    self.sat_count = np.clip(self.sat_count, 0.0, self.sat_limit)
    return self.sat_count > (self.sat_limit - 1e-3)
