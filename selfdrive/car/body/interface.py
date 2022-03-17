#!/usr/bin/env python3
from cereal import car
from selfdrive.car import scale_rot_inertia, scale_tire_stiffness, gen_empty_fingerprint, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase

class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), car_fw=None, disable_radar=False):

    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)
    ret.carName = "body"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.allOutput)]

    ret.steerLimitTimer = 1.0
    ret.steerRateCost = 0.5

    ret.steerActuatorDelay = 0.1

    ret.mass = 9
    ret.wheelbase = 0.406
    ret.wheelSpeedFactor = 0.008587
    ret.centerToFront = ret.wheelbase * 0.44
    ret.steerRatio = 1

    ret.steerControlType = car.CarParams.SteerControlType.angle
    ret.radarOffCan = False

    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)

    return ret

  # returns a car.CarState
  def update(self, c, can_strings):
    self.cp.update_strings(can_strings)

    ret = self.CS.update(self.cp)

    ret.canValid = self.cp.can_valid

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
