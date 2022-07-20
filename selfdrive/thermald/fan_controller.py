#!/usr/bin/env python3
from abc import ABC, abstractmethod

from common.realtime import DT_TRML
from common.numpy_fast import interp
from system.swaglog import cloudlog
from selfdrive.controls.lib.pid import PIDController

class BaseFanController(ABC):
  @abstractmethod
  def update(self, max_cpu_temp: float, ignition: bool) -> int:
    pass


class TiciFanController(BaseFanController):
  def __init__(self) -> None:
    super().__init__()
    cloudlog.info("Setting up TICI fan handler")

    self.last_ignition = False
    self.controller = PIDController(k_p=0, k_i=4e-3, k_f=1, neg_limit=-80, pos_limit=0, rate=(1 / DT_TRML))

  def update(self, max_cpu_temp: float, ignition: bool) -> int:
    self.controller.neg_limit = -(80 if ignition else 30)
    self.controller.pos_limit = -(30 if ignition else 0)

    if ignition != self.last_ignition:
      self.controller.reset()

    error = 70 - max_cpu_temp
    fan_pwr_out = -int(self.controller.update(
                      error=error,
                      feedforward=interp(max_cpu_temp, [60.0, 100.0], [0, -80])
                    ))

    self.last_ignition = ignition
    return fan_pwr_out

