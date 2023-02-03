#!/usr/bin/env python3
from cereal import car
from selfdrive.car import STD_CARGO_KG, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.car.nissan.values import CAR


class CarInterface(CarInterfaceBase):

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long):
    ret.carName = "nissan"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.nissan)]
    ret.autoResumeSng = False

    ret.steerLimitTimer = 1.0

    ret.steerActuatorDelay = 0.1
    ret.steerRatio = 17

    ret.steerControlType = car.CarParams.SteerControlType.angle
    ret.radarUnavailable = True

    if candidate in (CAR.ROGUE, CAR.XTRAIL):
      ret.mass = 1610 + STD_CARGO_KG
      ret.wheelbase = 2.705
      ret.centerToFront = ret.wheelbase * 0.44
    elif candidate in (CAR.LEAF, CAR.LEAF_IC):
      ret.mass = 1610 + STD_CARGO_KG
      ret.wheelbase = 2.705
      ret.centerToFront = ret.wheelbase * 0.44
    elif candidate == CAR.ALTIMA:
      # Altima has EPS on C-CAN unlike the others that have it on V-CAN
      ret.safetyConfigs[0].safetyParam = 1 # EPS is on alternate bus
      ret.mass = 1492 + STD_CARGO_KG
      ret.wheelbase = 2.824
      ret.centerToFront = ret.wheelbase * 0.44

    return ret

  # returns a car.CarState
  def _update(self, c):
    ret = self.CS.update(self.cp, self.cp_adas, self.cp_cam)

    buttonEvents = []
    be = car.CarState.ButtonEvent.new_message()
    be.type = car.CarState.ButtonEvent.Type.accelCruise
    buttonEvents.append(be)

    events = self.create_common_events(ret)

    if self.CS.lkas_enabled:
      events.add(car.CarEvent.EventName.invalidLkasSetting)

    ret.events = events.to_msg()

    return ret

  def apply(self, c):
    return self.CC.update(c, self.CS)
