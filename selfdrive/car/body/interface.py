#!/usr/bin/env python3
import math
from cereal import car
from common.realtime import DT_CTRL
from selfdrive.car import scale_rot_inertia, scale_tire_stiffness, get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase
from selfdrive.car.body.values import SPEED_FROM_RPM

class CarInterface(CarInterfaceBase):
  @staticmethod
  def get_params(candidate, fingerprint=None, car_fw=None, disable_radar=False):

    ret = CarInterfaceBase.get_std_params(candidate, fingerprint)

    ret.notCar = True
    ret.carName = "body"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.body)]

    ret.minSteerSpeed = -math.inf
    ret.maxLateralAccel = math.inf  # TODO: set to a reasonable value
    ret.steerRatio = 0.5
    ret.steerRateCost = 0.5
    ret.steerLimitTimer = 1.0
    ret.steerActuatorDelay = 0.

    ret.mass = 9
    ret.wheelbase = 0.406
    ret.wheelSpeedFactor = SPEED_FROM_RPM
    ret.centerToFront = ret.wheelbase * 0.44

    ret.radarOffCan = True
    ret.openpilotLongitudinalControl = True
    ret.steerControlType = car.CarParams.SteerControlType.angle

    ret.rotationalInertia = scale_rot_inertia(ret.mass, ret.wheelbase)

    ret.tireStiffnessFront, ret.tireStiffnessRear = scale_tire_stiffness(ret.mass, ret.wheelbase, ret.centerToFront)

    return ret

  def _update(self, c):
    ret = self.CS.update(self.cp)

    # wait for everything to init first
    if self.frame > int(5. / DT_CTRL):
      # body always wants to enable
      ret.init('events', 1)
      ret.events[0].name = car.CarEvent.EventName.pcmEnable
      ret.events[0].enable = True
    self.frame += 1

    return ret

  def apply(self, c):
    return self.CC.update(c, self.CS)
