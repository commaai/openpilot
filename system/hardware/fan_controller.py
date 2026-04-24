#!/usr/bin/env python3
import numpy as np

from openpilot.common.pid import PIDController
from openpilot.system.hardware import HARDWARE

# bang-bang thresholds fit from fleet
T_HIGH = 80.0
T_LOW = 65.0


class FanController:
  def __init__(self, rate: int) -> None:
    self.last_ignition = False
    self.high = False
    self.controller = PIDController(k_p=0, k_i=4e-3, rate=rate)

  def update(self, cur_temp: float, ignition: bool) -> int:
    if HARDWARE.get_device_type() == 'mici' and ignition:
      if cur_temp >= T_HIGH:
        self.high = True
      elif cur_temp <= T_LOW or not self.last_ignition:
        self.high = False
      self.last_ignition = ignition
      return 100 if self.high else 50

    self.controller.pos_limit = 100 if ignition else 30
    self.controller.neg_limit = 30 if ignition else 0

    if ignition != self.last_ignition:
      self.controller.reset()
    self.last_ignition = ignition

    return int(self.controller.update(
                 error=(cur_temp - 75),  # temperature setpoint in C
                 feedforward=np.interp(cur_temp, [60.0, 100.0], [0, 100])
              ))
