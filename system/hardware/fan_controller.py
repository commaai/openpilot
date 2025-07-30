#!/usr/bin/env python3
import numpy as np
from abc import ABC, abstractmethod

from openpilot.common.realtime import DT_HW
from openpilot.common.swaglog import cloudlog
from openpilot.common.pid import PIDController

class BaseFanController(ABC):
  @abstractmethod
  def update(self, cur_temp: float, ignition: bool) -> int:
    pass


class TiciFanController(BaseFanController):
  def __init__(self) -> None:
    super().__init__()
    cloudlog.info("Setting up TICI fan handler")

    self.last_ignition = False
    self.controller = PIDController(k_p=0, k_i=4e-3, k_f=1, rate=(1 / DT_HW))
    self.limit_maxxed = False

  def update(self, cur_temp: float, ignition: bool) -> int:
    self.controller.pos_limit = 100 if ignition else 30
    self.controller.neg_limit = 30 if ignition else 0

    if ignition != self.last_ignition:
      self.controller.reset()
    

    error = cur_temp - 75
    fan_pwr_out = int(self.controller.update(
                      error=error,
                      feedforward=np.interp(cur_temp, [60.0, 100.0], [0, 100]),
                      freeze_integrator=self.limit_maxxed,
                    ))
    self.limit_maxxed = abs(fan_pwr_out - self.controller.control) < 1e-3
    self.limit_maxxed = self.limit_maxxed or abs(self.controller.neg_limit - fan_pwr_out) < 1e-3

    self.last_ignition = ignition
    return fan_pwr_out

