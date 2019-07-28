#!/usr/bin/env python
from cereal import car
import time


class RadarInterface(object):
  def __init__(self, CP):
    # radar
    self.pts = {}
    self.delay = 0.1

  def update(self, can_strings):

    ret = car.RadarData.new_message()
    time.sleep(0.05)  # radard runs on RI updates

    return ret
