#!/usr/bin/env python3
from cereal import car
from selfdrive.car.body.values import CAR
from selfdrive.car import STD_CARGO_KG, scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)
    self.cp_adas = self.CS.get_adas_can_parser(CP)

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None):

    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "body"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.allOutput)]

    ret.steerLimitTimer = 1.0
    ret.steerRateCost = 0.5

    ret.steerActuatorDelay = 0.1

    ret.mass = 1610 + STD_CARGO_KG
    ret.wheelbase = 2.705
    ret.centerToFront = ret.wheelbase * 0.44
    ret.steerRatio = 17

    ret.steerControlType = car.CarParams.SteerControlType.angle
    ret.radarOffCan = True

    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    self.cp.update_strings(can_strings)
    self.cp_cam.update_strings(can_strings)
    self.cp_adas.update_strings(can_strings)

    ret = self.CS.update(self.cp, self.cp_adas, self.cp_cam)

    ret.canValid = self.cp.can_valid and self.cp_adas.can_valid and self.cp_cam.can_valid

    buttonEvents = []
    be = car.CarState.ButtonEvent.new_message()
    be.type = car.CarState.ButtonEvent.Type.accelCruise
    buttonEvents.append(be)

    events = self.create_common_events(ret)

    if self.CS.lkas_enabled:
      events.add(car.CarEvent.EventName.invalidLkasSetting)

    ret.events = events.to_msg()

    self.CS.out = ret.as_reader()
    return self.CS.out

  def apply(self, c):
    hud_control = c.hudControl
    ret = self.CC.update(c, self.CS, self.frame, c.actuators,
                         c.cruiseControl.cancel, hud_control.visualAlert,
                         hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                         hud_control.leftLaneDepart, hud_control.rightLaneDepart)
    self.frame += 1
    return ret
