#!/usr/bin/env python3
import numpy as np

from openpilot.common.pid import PIDController
from openpilot.common.hardware import HARDWARE

# raise fan setpoint on tici/tizi to reduce noise
# after raising LMH threshold in AGNOS 18.1 to prevent CPU throttling
OFFSET = 0 if HARDWARE.get_device_type() == "mici" else 5


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

    return int(self.controller.update(
                 error=(cur_temp - (75 + OFFSET)),  # temperature setpoint in C
                 feedforward=np.interp(cur_temp, [60.0 + OFFSET, 100.0 + OFFSET], [0, 100])
              ))


VOICE_IDLE_FAN_CAP = 60
VOICE_MAX_TEMP = 70.  # above this the thermal need wins over a quiet mic


def voice_fan_limit(desired: int, cur_temp: float, listening: bool, querying: bool) -> int:
  if not listening or cur_temp > VOICE_MAX_TEMP:
    return desired
  return 0 if querying else min(desired, VOICE_IDLE_FAN_CAP)
