from opendbc.car.structs import car
from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.locationd.torqued import LONG_ACC_THRESHOLD, TorqueEstimator


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

  def test_long_accel_filter(self):
    est = TorqueEstimator(car.CarParams(), track_all_points=True)

    def step(t, a_ego):
      carControl = messaging.new_message('carControl').carControl
      carOutput = messaging.new_message('carOutput').carOutput
      carState = messaging.new_message('carState').carState
      deviceMotion = messaging.new_message('deviceMotion').deviceMotion

      carControl.latActive = True
      carOutput.actuatorsOutput.torque = -0.05
      carState.vEgo = 20.0
      carState.aEgo = a_ego
      carState.steeringPressed = False

      deviceMotion.orientationNED = {'x': 0.0, 'valid': True}
      deviceMotion.angularVelocityDevice = {'z': 0.2 / 20.0, 'valid': True}
      deviceMotion.inputsOK, deviceMotion.sensorsOK, deviceMotion.posenetOK = True, True, True
      deviceMotion.timestamp = int(t * 1e9)

      for which, msg in (('carControl', carControl), ('carOutput', carOutput), ('carState', carState), ('deviceMotion', deviceMotion)):
        est.handle_log(t, which, msg)

    # Warmup buffer to hist_len so deviceMotion begins processing points
    t = 0.0
    for _ in range(est.hist_len - 1):
      step(t, a_ego=0.0)
      t += DT_MDL

    # 1. Low longitudinal accel: point is admitted to filtered_points
    prev_filtered = len(est.filtered_points)
    prev_all = len(est.all_torque_points)
    step(t, a_ego=0.2)
    t += DT_MDL
    assert len(est.filtered_points) == prev_filtered + 1
    assert len(est.all_torque_points) == prev_all + 1

    # 2. High positive longitudinal accel: filtered out of filtered_points, but tracked in all_torque_points
    prev_filtered = len(est.filtered_points)
    prev_all = len(est.all_torque_points)
    step(t, a_ego=LONG_ACC_THRESHOLD + 0.5)
    t += DT_MDL
    assert len(est.filtered_points) == prev_filtered
    assert len(est.all_torque_points) == prev_all + 1

    # 3. High negative longitudinal accel (braking): filtered out of filtered_points
    prev_filtered = len(est.filtered_points)
    prev_all = len(est.all_torque_points)
    step(t, a_ego=-LONG_ACC_THRESHOLD - 0.5)
    t += DT_MDL
    assert len(est.filtered_points) == prev_filtered
    assert len(est.all_torque_points) == prev_all + 1

    # 4. Boundary cases: exactly at threshold (+/- LONG_ACC_THRESHOLD) are admitted
    prev_filtered = len(est.filtered_points)
    step(t, a_ego=float(LONG_ACC_THRESHOLD))
    t += DT_MDL
    assert len(est.filtered_points) == prev_filtered + 1

    prev_filtered = len(est.filtered_points)
    step(t, a_ego=float(-LONG_ACC_THRESHOLD))
    t += DT_MDL
    assert len(est.filtered_points) == prev_filtered + 1

