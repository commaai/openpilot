#!/usr/bin/env python3
from cereal import car
from selfdrive.car.nissan.values import CAR
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

class CarInterface(CarInterfaceBase):

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None, disable_radar=False):

    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "nissan"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.nissan)]

    ret.steerLimitTimer = 1.0
    ret.steerRateCost = 0.5

    ret.steerActuatorDelay = 0.1

    if candidate in (CAR.ROGUE, CAR.XTRAIL):
      ret.mass = 1610 + STD_CARGO_KG
      ret.wheelbase = 2.705
      ret.centerToFront = ret.wheelbase * 0.44
      ret.steerRatio = 17
    elif candidate in (CAR.LEAF, CAR.LEAF_IC):
      ret.mass = 1610 + STD_CARGO_KG
      ret.wheelbase = 2.705
      ret.centerToFront = ret.wheelbase * 0.44
      ret.steerRatio = 17
    elif candidate == CAR.ALTIMA:
      # Altima has EPS on C-CAN unlike the others that have it on V-CAN
      ret.safetyConfigs[0].safetyParam = 1 # EPS is on alternate bus
      ret.mass = 1492 + STD_CARGO_KG
      ret.wheelbase = 2.824
      ret.centerToFront = ret.wheelbase * 0.44
      ret.steerRatio = 17

    ret.steerControlType = car.CarParams.SteerControlType.angle
    ret.radarOffCan = True

    # TODO: get actual value, for now starting with reasonable value for
    # civic and scaling by mass and wheelbase
    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    # TODO: start from empirically derived lateral slip stiffness for the civic and scale by
    # mass and CG position, so all cars will have approximately similar dyn behaviors
    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)

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
    hud_control = c.hudControl
    ret = self.CC.update(c, self.CS, self.frame, c.actuators,
                         c.cruiseControl.cancel, hud_control.visualAlert,
                         hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                         hud_control.leftLaneDepart, hud_control.rightLaneDepart)
    self.frame += 1
    return ret
