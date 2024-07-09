from cereal import car
from openpilot.selfdrive.car import get_safety_config
from openpilot.selfdrive.car.interfaces import CarInterfaceBase
from openpilot.selfdrive.car.fca_giorgio.values import CAR


class CarInterface(CarInterfaceBase):
  @staticmethod
  def _get_params(ret, candidate: CAR, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "fca_giorgio"
    ret.radarUnavailable = True

    # Set global parameters

    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.fcaGiorgio)]

    # Global lateral tuning defaults, can be overridden per-vehicle

    ret.steerLimitTimer = 1.0
    ret.steerActuatorDelay = 0.1
    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    # Global longitudinal tuning defaults, can be overridden per-vehicle

    ret.pcmCruise = not ret.openpilotLongitudinalControl

    return ret

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)

    events = self.create_common_events(ret, pcm_enable=not self.CS.CP.openpilotLongitudinalControl)

    ret.events = events.to_msg()
    return ret

