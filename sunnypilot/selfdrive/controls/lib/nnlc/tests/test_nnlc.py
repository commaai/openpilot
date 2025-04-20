import numpy as np
from parameterized import parameterized

from cereal import car, log, messaging
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.params import Params
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.locationd.helpers import Pose
from openpilot.common.mock.generators import generate_livePose
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.selfdrive.modeld.constants import ModelConstants


def generate_modelV2():
  model = messaging.new_message('modelV2')
  position = log.XYZTData.new_message()
  speed = 30
  position.x = [float(x) for x in (speed + 0.5) * np.array(ModelConstants.T_IDXS)]
  model.modelV2.position = position
  orientation = log.XYZTData.new_message()
  curvature = 0.05
  orientation.x = [float(curvature) for _ in ModelConstants.T_IDXS]
  orientation.y = [0.0 for _ in ModelConstants.T_IDXS]
  model.modelV2.orientation = orientation
  velocity = log.XYZTData.new_message()
  velocity.x = [float(x) for x in (speed + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
  velocity.x[0] = float(speed)  # always start at current speed
  model.modelV2.velocity = velocity
  acceleration = log.XYZTData.new_message()
  acceleration.x = [float(x) for x in np.zeros_like(ModelConstants.T_IDXS)]
  acceleration.y = [float(y) for y in np.zeros_like(ModelConstants.T_IDXS)]
  model.modelV2.acceleration = acceleration

  return model


class TestNeuralNetworkLateralControl:

  @parameterized.expand([HONDA.HONDA_CIVIC, TOYOTA.TOYOTA_RAV4, HYUNDAI.HYUNDAI_SANTA_CRUZ_1ST_GEN])
  def test_saturation(self, car_name):
    params = Params()
    params.put_bool("NeuralNetworkLateralControl", True)

    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
    CI = CarInterface(CP, CP_SP)

    sunnypilot_interfaces.setup_interfaces(CP, CP_SP, params)

    CP_SP = convert_to_capnp(CP_SP)
    VM = VehicleModel(CP)

    controller = LatControlTorque(CP.as_reader(), CP_SP.as_reader(), CI)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    CS.steeringPressed = False

    params = log.LiveParametersData.new_message()

    lp = generate_livePose()
    pose = Pose.from_live_pose(lp.livePose)

    mdl = generate_modelV2()
    sm = {'modelV2': mdl.modelV2}
    model_v2 = sm['modelV2']
    controller.extension.model_v2 = model_v2

    # Saturate for curvature limited and controller limited
    for _ in range(1000):
      controller.extension.update_model_v2(model_v2)
      _, _, lac_log = controller.update(True, CS, VM, params, False, 0, pose, True)
    assert lac_log.saturated

    for _ in range(1000):
      controller.extension.update_model_v2(model_v2)
      _, _, lac_log = controller.update(True, CS, VM, params, False, 0, pose, False)
    assert not lac_log.saturated

    for _ in range(1000):
      controller.extension.update_model_v2(model_v2)
      _, _, lac_log = controller.update(True, CS, VM, params, False, 1, pose, False)
    assert lac_log.saturated
