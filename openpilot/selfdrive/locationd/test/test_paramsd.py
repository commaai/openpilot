import random
import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import messaging
from openpilot.selfdrive.locationd.paramsd import retrieve_initial_vehicle_params
from openpilot.selfdrive.locationd.models.car_kf import CarKalman
from openpilot.common.params import Params
from opendbc.car.structs import car


def get_random_vehicle_parameters(CP):
  msg = messaging.new_message("vehicleParameters")
  msg.vehicleParameters.steerRatio = (random.random() + 0.5) * CP.steerRatio
  msg.vehicleParameters.stiffnessFactor = random.random()
  msg.vehicleParameters.angleOffsetAverageDeg = random.random()
  msg.vehicleParameters.debugFilterState.std = [random.random() for _ in range(CarKalman.P_initial.shape[0])]
  return msg


class TestParamsd(OpenpilotTestCase):
  def test_read_saved_params(self):
    params = Params()

    CP = car.CarParams(carFingerprint="HONDA_CIVIC", steerRatio=15.38)

    msg = get_random_vehicle_parameters(CP)
    params.put("LiveParametersV2", msg.to_bytes(), block=True)
    params.put("CarParamsPrevRoute", CP.to_bytes(), block=True)

    sr, sf, offset, p_init = retrieve_initial_vehicle_params(params, CP, replay=True, debug=True)
    np.testing.assert_allclose(sr, msg.vehicleParameters.steerRatio)
    np.testing.assert_allclose(sf, msg.vehicleParameters.stiffnessFactor)
    np.testing.assert_allclose(offset, msg.vehicleParameters.angleOffsetAverageDeg)
    np.testing.assert_equal(p_init.shape, CarKalman.P_initial.shape)
    np.testing.assert_allclose(np.diagonal(p_init), msg.vehicleParameters.debugFilterState.std)
