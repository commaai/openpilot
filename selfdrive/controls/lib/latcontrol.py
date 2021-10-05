from common.numpy_fast import clip
from common.realtime import DT_CTRL

MIN_STEER_SPEED = 0.3


class LatControl:
  def __init__(self, CP):
    self.sat_count_rate = 1.0 * DT_CTRL
    self.sat_limit = CP.steerLimitTimer

  def reset(self):
    self.sat_count = 0.

  def _check_saturation(self, control, limit, CS):
    saturated = abs(control) == limit

    if saturated and CS.vEgo > 10. and not CS.steeringRateLimited and not CS.steeringPressed:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit
