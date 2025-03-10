from opendbc.car import get_safety_config, structs
from opendbc.car.interfaces import CarInterfaceBase


class CarInterface(CarInterfaceBase):

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, experimental_long, docs) -> structs.CarParams:
    ret.brand = "rivian"

    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.rivian)]

    ret.steerActuatorDelay = 0.25
    ret.steerLimitTimer = 0.4
    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    ret.steerControlType = structs.CarParams.SteerControlType.torque
    ret.radarUnavailable = True

    # TODO: pending finding/handling missing set speed and fixing up radar parser
    ret.experimentalLongitudinalAvailable = False
    if experimental_long:
      ret.openpilotLongitudinalControl = True
      #ret.safetyConfigs[0].safetyParam |= Panda.FLAG_RIVIAN_LONG_CONTROL

    ret.longitudinalActuatorDelay = 0.35
    ret.vEgoStopping = 0.25
    ret.stopAccel = 0

    return ret
