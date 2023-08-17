#!/usr/bin/env python3
from cereal import car
from system.swaglog import cloudlog
import cereal.messaging as messaging
from selfdrive.car import get_safety_config
from selfdrive.car.interfaces import CarInterfaceBase


# mocked car interface to work with chffrplus
class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    cloudlog.debug("Using Mock Car Interface")

    self.sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal'])

    self.speed = 0.
    self.prev_speed = 0.

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "mock"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.noOutput)]
    ret.mass = 1700.
    ret.wheelbase = 2.70
    ret.centerToFront = ret.wheelbase * 0.5
    ret.steerRatio = 13.  # reasonable

    return ret

  # returns a car.CarState
  def _update(self, c):
    self.sm.update(0)
    gps_sock = 'gpsLocationExternal' if self.sm.rcv_frame['gpsLocationExternal'] > 1 else 'gpsLocation'
    if self.sm.updated[gps_sock]:
      self.prev_speed = self.speed
      self.speed = self.sm[gps_sock].speed

    # create message
    ret = car.CarState.new_message()

    # speeds
    ret.vEgo = self.speed
    ret.vEgoRaw = self.speed

    ret.aEgo = self.speed - self.prev_speed
    ret.brakePressed = ret.aEgo < -0.5

    ret.standstill = self.speed < 0.01
    ret.wheelSpeeds.fl = self.speed
    ret.wheelSpeeds.fr = self.speed
    ret.wheelSpeeds.rl = self.speed
    ret.wheelSpeeds.rr = self.speed

    return ret

  def apply(self, c, now_nanos):
    # in mock no carcontrols
    actuators = car.CarControl.Actuators.new_message()
    return actuators, []
