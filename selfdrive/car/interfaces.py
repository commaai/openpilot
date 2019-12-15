import os
import time
from cereal import car
from selfdrive.car import gen_empty_fingerprint

# generic car and radar interfaces

class CarInterfaceBase():
  def __init__(self, CP, CarController):
    pass

  @staticmethod
  def calc_accel_override(a_ego, a_target, v_ego, v_target):
    return 1.

  @staticmethod
  def compute_gb(accel, speed):
    raise NotImplementedError

  @staticmethod
  def get_params(candidate, fingerprint=gen_empty_fingerprint(), vin="", has_relay=False):
    raise NotImplementedError

  # returns a car.CarState, pass in car.CarControl
  def update(self, c, can_strings):
    raise NotImplementedError

  # return sendcan, pass in a car.CarControl
  def apply(self, c):
    raise NotImplementedError

class RadarInterfaceBase():
  def __init__(self, CP):
    self.pts = {}
    self.delay = 0

  def update(self, can_strings):
    ret = car.RadarData.new_message()

    if 'NO_RADAR_SLEEP' not in os.environ:
      time.sleep(0.05)  # radard runs on RI updates

    return ret
