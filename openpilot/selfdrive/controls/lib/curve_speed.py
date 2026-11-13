from opendbc.car.interfaces import ACCEL_MAX

MAX_LAT_ACCEL = 2.8
MIN_CURVE_ACCEL = -0.5


def get_curve_accel(model):
  for t, v, yaw_rate in zip(model.position.t, model.velocity.x, model.orientationRate.z, strict=False):
    if 0.5 < t < 6.0 and abs(v * yaw_rate) > MAX_LAT_ACCEL:
      return MIN_CURVE_ACCEL
  return ACCEL_MAX
