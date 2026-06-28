import random
import numpy as np

from openpilot.cereal import messaging
from openpilot.selfdrive.locationd.paramsd import VehicleParamsLearner, retrieve_initial_vehicle_params, migrate_cached_vehicle_params_if_needed
from openpilot.selfdrive.locationd.models.car_kf import CarKalman, States
from openpilot.selfdrive.locationd.test.test_locationd_scenarios import TEST_ROUTE
from openpilot.selfdrive.test.process_replay.migration import migrate, migrate_carParams
from openpilot.common.params import Params
from openpilot.tools.lib.logreader import LogReader
from opendbc.car.structs import car


def get_random_live_parameters(CP):
  msg = messaging.new_message("liveParameters")
  msg.liveParameters.steerRatio = (random.random() + 0.5) * CP.steerRatio
  msg.liveParameters.stiffnessFactor = random.random()
  msg.liveParameters.angleOffsetAverageDeg = random.random()
  msg.liveParameters.debugFilterState.std = [random.random() for _ in range(CarKalman.P_initial.shape[0])]
  return msg


def _make_car_state(speed: float, gear: car.CarState.GearShifter, steering_angle: float = 0.0):
  msg = messaging.new_message('carState')
  msg.carState.vEgo = speed
  msg.carState.gearShifter = gear
  msg.carState.steeringAngleDeg = steering_angle
  return msg.carState


class TestVehicleParamsLearner:
  def _make_learner(self):
    CP = car.CarParams()
    CP.mass = 1700
    CP.rotationalInertia = 2500
    CP.centerToFront = 1.2
    CP.wheelbase = 2.7
    CP.tireStiffnessFront = 1.0
    CP.tireStiffnessRear = 1.0
    CP.steerRatio = 13.0
    return VehicleParamsLearner(CP, steer_ratio=13.0, stiffness_factor=1.0, angle_offset=0.0)

  def test_not_active_in_reverse(self):
    learner = self._make_learner()
    # Speed above MIN_ACTIVE_SPEED but in reverse — must not activate
    cs = _make_car_state(speed=3.0, gear=car.CarState.GearShifter.reverse)
    learner.handle_log(0.0, 'carState', cs)
    assert not learner.active, "paramsd must not activate while in reverse gear"

  def test_active_in_drive(self):
    learner = self._make_learner()
    cs = _make_car_state(speed=3.0, gear=car.CarState.GearShifter.drive)
    learner.handle_log(0.0, 'carState', cs)
    assert learner.active, "paramsd should activate when driving forward"

  def test_stiffness_unchanged_during_reverse(self):
    learner = self._make_learner()
    initial_stiffness = float(learner.kf.x[States.STIFFNESS].item())

    # Simulate reverse at walking speed
    for i in range(50):
      cs = _make_car_state(speed=2.0, gear=car.CarState.GearShifter.reverse)
      learner.handle_log(i * 0.05, 'carState', cs)

    final_stiffness = float(learner.kf.x[States.STIFFNESS].item())
    np.testing.assert_allclose(final_stiffness, initial_stiffness,
                               err_msg="Stiffness should not drift while vehicle is in reverse")


class TestParamsd:
  def test_read_saved_params(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = get_random_live_parameters(CP)
    params.put("LiveParametersV2", msg.to_bytes(), block=True)
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes(), block=True)

    migrate_cached_vehicle_params_if_needed(params) # this is not tested here but should not mess anything up or throw an error
    sr, sf, offset, p_init = retrieve_initial_vehicle_params(params, CP, replay=True, debug=True)
    np.testing.assert_allclose(sr, msg.liveParameters.steerRatio)
    np.testing.assert_allclose(sf, msg.liveParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.liveParameters.angleOffsetAverageDeg)
    np.testing.assert_equal(p_init.shape, CarKalman.P_initial.shape)
    np.testing.assert_allclose(np.diagonal(p_init), msg.liveParameters.debugFilterState.std)

  # TODO Remove this test after the support for old format is removed
  def test_read_saved_params_old_format(self):
    params = Params()

    lr = migrate(LogReader(TEST_ROUTE), [migrate_carParams])
    CP = next(m for m in lr if m.which() == "carParams").carParams

    msg = get_random_live_parameters(CP)
    params.put("LiveParameters", msg.liveParameters.to_dict(), block=True)
    params.put("CarParamsPrevRoute", CP.as_builder().to_bytes(), block=True)
    params.remove("LiveParametersV2")

    migrate_cached_vehicle_params_if_needed(params)
    sr, sf, offset, _ = retrieve_initial_vehicle_params(params, CP, replay=True, debug=True)
    np.testing.assert_allclose(sr, msg.liveParameters.steerRatio)
    np.testing.assert_allclose(sf, msg.liveParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.liveParameters.angleOffsetAverageDeg)
    assert params.get("LiveParametersV2") is not None

  def test_read_saved_params_corrupted_old_format(self):
    params = Params()
    params.put("LiveParameters", {}, block=True)
    params.remove("LiveParametersV2")

    migrate_cached_vehicle_params_if_needed(params)
    assert params.get("LiveParameters") is None
    assert params.get("LiveParametersV2") is None
