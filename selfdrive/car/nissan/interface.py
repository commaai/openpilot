from panda import Panda
from openpilot.selfdrive.car import get_safety_config, structs
from openpilot.selfdrive.car.interfaces import CarInterfaceBase
from openpilot.selfdrive.car.nissan.values import CAR


class CarInterface(CarInterfaceBase):

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, experimental_long, docs) -> structs.CarParams:
    ret.carName = "nissan"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.nissan)]
    ret.autoResumeSng = False

    ret.steerLimitTimer = 1.0

    ret.steerActuatorDelay = 0.1

    ret.steerControlType = structs.CarParams.SteerControlType.angle
    ret.radarUnavailable = True

    if candidate == CAR.NISSAN_ALTIMA:
      # Altima has EPS on C-CAN unlike the others that have it on V-CAN
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_NISSAN_ALT_EPS_BUS

    return ret
