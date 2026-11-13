from openpilot.common.filter_simple import FirstOrderFilter

MAX_LAT_ACCEL = 2.8
CURVE_ACCEL_OFFSET = -0.5


class CurveSpeed:
  def __init__(self, dt):
    self.accel_offset = FirstOrderFilter(0.0, 0.5, dt)

  def reset(self):
    self.accel_offset.x = 0.0

  def update(self, model):
    target = 0.0
    for t, v, yaw_rate in zip(model.position.t, model.velocity.x, model.orientationRate.z, strict=False):
      if 0.5 < t < 6.0 and abs(v * yaw_rate) > MAX_LAT_ACCEL:
        target = CURVE_ACCEL_OFFSET
        break
    return self.accel_offset.update(target)
