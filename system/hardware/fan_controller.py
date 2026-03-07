#!/usr/bin/env python3
import math

import numpy as np

from openpilot.common.pid import PIDController


class FanController:
  def __init__(self, rate: int) -> None:
    self.last_ignition = False
    self.controller = PIDController(k_p=0, k_i=4e-3, rate=rate)

  def update(self, cur_temp: float, ignition: bool) -> int:
    self.controller.pos_limit = 100 if ignition else 30
    self.controller.neg_limit = 30 if ignition else 0

    if ignition != self.last_ignition:
      self.controller.reset()
    self.last_ignition = ignition

    # guard against NaN/Inf from failed sensor reads
    if not math.isfinite(cur_temp):
      cur_temp = 0.

    return int(self.controller.update(
                 error=(cur_temp - 75),  # temperature setpoint in C
                 feedforward=np.interp(cur_temp, [60.0, 100.0], [0, 100])
              ))
