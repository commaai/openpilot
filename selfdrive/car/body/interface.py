import math
from openpilot.selfdrive.car import get_safety_config, structs
from openpilot.selfdrive.car.interfaces import CarInterfaceBase
from openpilot.selfdrive.car.body.values import SPEED_FROM_RPM


class CarInterface(CarInterfaceBase):
  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, experimental_long, docs) -> structs.CarParams:
    ret.notCar = True
    ret.carName = "body"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.body)]

    ret.minSteerSpeed = -math.inf
    ret.maxLateralAccel = math.inf  # TODO: set to a reasonable value
    ret.steerLimitTimer = 1.0
    ret.steerActuatorDelay = 0.

    ret.wheelSpeedFactor = SPEED_FROM_RPM

    ret.radarUnavailable = True
    ret.openpilotLongitudinalControl = True
    ret.steerControlType = structs.CarParams.SteerControlType.angle

    return ret
