import numpy as np

from opendbc.car.structs import car
from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.locationd.torqued import (
  FACTOR_SANITY_MIN,
  FACTOR_SANITY_MAX,
  MIN_BUCKET_POINTS,
  STEER_BUCKET_BOUNDS,
  TorqueEstimator,
)


class TestTorqued(OpenpilotTestCase):
  def test_cal_percent(self):
    est = TorqueEstimator(car.CarParams())
    msg = est.get_msg()
    assert msg.lateralTorqueParameters.calPerc == 0

    for (low, high), min_pts in zip(est.filtered_points.buckets.keys(),
                                    est.filtered_points.buckets_min_points.values(), strict=True):
      for _ in range(int(min_pts)):
        est.filtered_points.add_point((low + high) / 2.0, 0.0)

    # enough bucket points, but not enough total points
    msg = est.get_msg()
    assert msg.lateralTorqueParameters.calPerc == (len(est.filtered_points) / est.min_points_total * 100 + 100) / 2

    # add enough points to bucket with most capacity
    key = list(est.filtered_points.buckets)[0]
    for _ in range(est.min_points_total - len(est.filtered_points)):
      est.filtered_points.add_point((key[0] + key[1]) / 2.0, 0.0)

    msg = est.get_msg()
    assert msg.lateralTorqueParameters.calPerc == 100

  def test_high_lat_accel_factor_outer_buckets(self):
    offline_factor = 1.95575
    true_factor = 3.0
    v_ego = 25.0

    CP = car.CarParams.new_message()
    CP.brand = 'toyota'
    CP.lateralTuning.init('torque')
    CP.lateralTuning.torque.latAccelFactor = offline_factor
    CP.lateralTuning.torque.friction = 0.12

    est = TorqueEstimator(CP)
    car_control = messaging.new_message('carControl').carControl
    car_output = messaging.new_message('carOutput').carOutput
    car_state = messaging.new_message('carState').carState
    device_motion = messaging.new_message('deviceMotion').deviceMotion
    car_control.latActive = True
    car_state.vEgo = v_ego
    car_state.steeringPressed = False
    device_motion.inputsOK = True
    device_motion.sensorsOK = True
    device_motion.posenetOK = True
    device_motion.orientationNED = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'valid': True}

    # Build steer sequence covering all buckets (1.5x min points -> >4000 total),
    # including outer buckets [-0.5, -0.3) and [0.3, 0.5) where |lat_accel| = 1.26 > 1.0.
    steer_seq = [0.0] * est.hist_len
    for (low, high), min_pts in zip(STEER_BUCKET_BOUNDS, MIN_BUCKET_POINTS, strict=True):
      steer_val = 0.42 if low >= 0.3 else (-0.42 if high <= -0.3 else (low + high) / 2.0)
      steer_seq.extend([steer_val] * int(1.5 * min_pts))

    for idx, steer in enumerate(steer_seq):
      t = idx * DT_MDL
      lat_accel = true_factor * steer
      car_output.actuatorsOutput.torque = float(-steer)
      device_motion.angularVelocityDevice = {'x': 0.0, 'y': 0.0, 'z': float(lat_accel / v_ego), 'valid': True}
      device_motion.timestamp = int(t * 1e9)
      for which, m in (('carControl', car_control), ('carOutput', car_output), ('carState', car_state), ('deviceMotion', device_motion)):
        est.handle_log(t, which, m)

    assert len(est.filtered_points.buckets[(-0.5, -0.3)]) >= 100
    assert len(est.filtered_points.buckets[(0.3, 0.5)]) >= 100

    msg = est.get_msg()
    ltp = msg.lateralTorqueParameters
    assert ltp.valid
    assert abs(ltp.latAccelFactorRaw - true_factor) < 0.1
    assert np.isclose(est.max_lataccel_factor, (1.0 + FACTOR_SANITY_MAX) * offline_factor)

    # Converge filtered estimate past the old 1.3x offline cap toward true_factor
    for _ in range(3000):
      msg = est.get_msg()
    assert msg.lateralTorqueParameters.latAccelFactorFiltered > 1.3 * offline_factor
    assert msg.lateralTorqueParameters.latAccelFactorFiltered <= est.max_lataccel_factor + 1e-6

  def test_factor_sanity_lower_floor_preserved(self):
    offline_factor = 2.0
    CP = car.CarParams.new_message()
    CP.brand = 'toyota'
    CP.lateralTuning.init('torque')
    CP.lateralTuning.torque.latAccelFactor = offline_factor
    CP.lateralTuning.torque.friction = 0.12

    est = TorqueEstimator(CP)
    for (low, high), min_pts in zip(STEER_BUCKET_BOUNDS, MIN_BUCKET_POINTS, strict=True):
      for steer in np.linspace(low + 0.01, high - 0.01, int(1.5 * min_pts)):
        est.filtered_points.add_point(float(steer), float(0.5 * steer))

    msg = est.get_msg()
    assert msg.lateralTorqueParameters.valid
    assert msg.lateralTorqueParameters.latAccelFactorRaw < (1.0 - FACTOR_SANITY_MIN) * offline_factor

    for _ in range(3000):
      msg = est.get_msg()
    min_allowed = (1.0 - FACTOR_SANITY_MIN) * offline_factor
    assert msg.lateralTorqueParameters.latAccelFactorFiltered >= min_allowed - 1e-3
