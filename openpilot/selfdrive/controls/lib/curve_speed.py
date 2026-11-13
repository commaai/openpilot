import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY

MAX_LAT_ACCEL = 2.8
MAX_ACCEL_OFFSET = 0.5
MAX_OFFSET_JERK = 0.5
LOOKAHEAD_TIME = 6.0


class CurveSpeed:
  def __init__(self, dt):
    self.dt = dt
    self.accel_offset = 0.0

  def reset(self):
    self.accel_offset = 0.0

  def update(self, model, roll):
    target = 0.0
    for t, v, yaw_rate, predicted_roll in zip(model.position.t, model.velocity.x, model.orientationRate.z, model.orientation.x, strict=False):
      lat_accel = v * yaw_rate - ACCELERATION_DUE_TO_GRAVITY * (roll + predicted_roll)
      if t < LOOKAHEAD_TIME and abs(lat_accel) >= MAX_LAT_ACCEL:
        target = -MAX_ACCEL_OFFSET
        break
    max_step = MAX_OFFSET_JERK * self.dt
    self.accel_offset = float(np.clip(target, self.accel_offset - max_step, self.accel_offset + max_step))
    return self.accel_offset
