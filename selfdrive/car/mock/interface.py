#!/usr/bin/env python3
from cereal import car
import cereal.messaging as messaging
from openpilot.selfdrive.car.interfaces import CarInterfaceBase

# mocked car interface for dashcam mode
class CarInterface(CarInterfaceBase):
  def __init__(self, CP, CarController, CarState):
    super().__init__(CP, CarController, CarState)

    self.speed = 0.
    self.sm = messaging.SubMaster(['gpsLocation', 'gpsLocationExternal'])

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "mock"
    ret.mass = 1700.
    ret.wheelbase = 2.70
    ret.centerToFront = ret.wheelbase * 0.5
    ret.steerRatio = 13.
    return ret

  def _update(self, c):
    self.sm.update(0)
    gps_sock = 'gpsLocationExternal' if self.sm.recv_frame['gpsLocationExternal'] > 1 else 'gpsLocation'

    ret = car.CarState.new_message()
    ret.vEgo = self.sm[gps_sock].speed
    ret.vEgoRaw = self.sm[gps_sock].speed

    return ret
