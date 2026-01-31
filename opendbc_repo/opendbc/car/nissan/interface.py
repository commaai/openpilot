from opendbc.car import get_safety_config, structs
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.nissan.carcontroller import CarController
from opendbc.car.nissan.carstate import CarState
from opendbc.car.nissan.values import CAR, NissanSafetyFlags

class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController

  DRIVABLE_GEARS = (structs.CarState.GearShifter.brake,)

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "nissan"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.nissan)]
    ret.autoResumeSng = False
    ret.steerAtStandstill = True

    ret.steerLimitTimer = 1.0

    ret.steerActuatorDelay = 0.1

    ret.steerControlType = structs.CarParams.SteerControlType.angle
    ret.radarUnavailable = True

    if candidate == CAR.NISSAN_ALTIMA:
      # Altima has EPS on C-CAN unlike the others that have it on V-CAN
      ret.safetyConfigs[0].safetyParam |= NissanSafetyFlags.ALT_EPS_BUS.value

    return ret
