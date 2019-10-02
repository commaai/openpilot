#!/usr/bin/env python
import os
import time
from cereal import car

class RadarInterface(object):
  def __init__(self, CP):
    # radar
    self.pts = {}
    self.delay = 0.1

  def update(self, can_strings):
    ret = car.RadarData.new_message()

    if 'NO_RADAR_SLEEP' not in os.environ:
      time.sleep(0.05)  # radard runs on RI updates

    return ret
