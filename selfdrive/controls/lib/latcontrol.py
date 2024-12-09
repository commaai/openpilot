from abc import abstractmethod, ABC

from openpilot.common.numpy_fast import clip
from openpilot.common.realtime import DT_CTRL

MIN_LATERAL_CONTROL_SPEED = 0.3  # m/s


class LatControl(ABC):
  def __init__(self, CP, CI):
    self.sat_count_rate = 1.0 * DT_CTRL
    self.sat_limit = CP.steerLimitTimer
    self.sat_count = 0.
    self.sat_check_min_speed = 10.

    # we define the steer torque scale as [-1.0...1.0]
    self.steer_max = 1.0

  @abstractmethod
  def update(self, active, CS, VM, params, steer_limited, desired_curvature, calibrated_pose):
    pass

  def reset(self):
    self.sat_count = 0.

  def _check_saturation(self, saturated, CS, steer_limited):
    if saturated and CS.vEgo > self.sat_check_min_speed and not steer_limited and not CS.steeringPressed:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate
    self.sat_count = clip(self.sat_count, 0.0, self.sat_limit)
    return self.sat_count > (self.sat_limit - 1e-3)
