from cereal import car
from openpilot.selfdrive.car import get_safety_config
from openpilot.selfdrive.car.interfaces import CarInterfaceBase
from openpilot.selfdrive.car.gwm.values import CAR

EventName = car.CarEvent.EventName

class CarInterface(CarInterfaceBase):
  @staticmethod
  def _get_params(ret, candidate: CAR, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "gwm"
    ret.radarUnavailable = True

    # Set global parameters

    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.allOutput,1)]

    # Global lateral tuning defaults, can be overriden per-vehicle

    ret.steerLimitTimer = 0.4
    ret.steerActuatorDelay = 0.2
    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    # Global longitudinal tuning defaults, can be overridden per-vehicle

    ret.pcmCruise = not ret.openpilotLongitudinalControl

    return ret

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_cam)

    events = self.create_common_events(ret, pcm_enable=not self.CS.CP.openpilotLongitudinalControl)

    # BEGIN TODO clean-after-port
    if ret.brakePressed:
      events.add(EventName.reverseGear)
    # END TODO clean-after-port

    ret.events = events.to_msg()
    return ret