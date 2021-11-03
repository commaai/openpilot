from common.realtime import DT_CTRL

MIN_STEER_SPEED = 0.3


class LatControl:
  def __init__(self, CP, CI):
    self.sat_count_rate = 1.0 * DT_CTRL
    self.sat_limit = CP.steerLimitTimer

  def reset(self):
    self.sat_count = 0.

  def _check_saturation(self, saturated, CS):
    if saturated and CS.vEgo > 10. and not CS.steeringRateLimited and not CS.steeringPressed:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate
    return self.sat_count > self.sat_limit
