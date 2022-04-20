from abc import abstractmethod, ABC

from common.realtime import DT_CTRL
from common.numpy_fast import clip

MIN_STEER_SPEED = 0.3


class LatControl(ABC):
  def __init__(self, CP, CI):
    self.sat_count_rate = 1.0 * DT_CTRL
    self.sat_limit = CP.steerLimitTimer
    self.sat_count = 0.

    # we define the steer torque scale as [-1.0...1.0]
    self.steer_max = 1.0

  @abstractmethod
  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate, llk):
    pass

  def reset(self):
    self.sat_count = 0.

  def _check_saturation(self, saturated, CS):
    if saturated and CS.vEgo > 10. and not CS.steeringRateLimited and not CS.steeringPressed:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate
    self.sat_count = clip(self.sat_count, 0.0, self.sat_limit)
    return self.sat_count > (self.sat_limit - 1e-3)
