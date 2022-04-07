#!/usr/bin/env python3

import os
from smbus2 import SMBus
from abc import ABC, abstractmethod
from common.realtime import DT_TRML
from common.numpy_fast import interp
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.pid import PIDController

class BaseFanController(ABC):
  @abstractmethod
  def update(self, max_cpu_temp: float, ignition: bool) -> int:
    pass


class EonFanController(BaseFanController):
  # Temp thresholds to control fan speed - high hysteresis
  TEMP_THRS_H = [50., 65., 80., 10000]
  # Temp thresholds to control fan speed - low hysteresis
  TEMP_THRS_L = [42.5, 57.5, 72.5, 10000]
  # Fan speed options
  FAN_SPEEDS = [0, 16384, 32768, 65535]

  def __init__(self) -> None:
    super().__init__()
    cloudlog.info("Setting up EON fan handler")

    self.fan_speed = -1
    self.setup_eon_fan()

  def setup_eon_fan(self) -> None:
    os.system("echo 2 > /sys/module/dwc3_msm/parameters/otg_switch")

  def set_eon_fan(self, speed: int) -> None:
    if self.fan_speed != speed:
      # FIXME: this is such an ugly hack to get the right index
      val = speed // 16384

      bus = SMBus(7, force=True)
      try:
        i = [0x1, 0x3 | 0, 0x3 | 0x08, 0x3 | 0x10][val]
        bus.write_i2c_block_data(0x3d, 0, [i])
      except OSError:
        # tusb320
        if val == 0:
          bus.write_i2c_block_data(0x67, 0xa, [0])
        else:
          bus.write_i2c_block_data(0x67, 0xa, [0x20])
          bus.write_i2c_block_data(0x67, 0x8, [(val - 1) << 6])
      bus.close()
      self.fan_speed = speed

  def update(self, max_cpu_temp: float, ignition: bool) -> int:
    new_speed_h = next(speed for speed, temp_h in zip(self.FAN_SPEEDS, self.TEMP_THRS_H) if temp_h > max_cpu_temp)
    new_speed_l = next(speed for speed, temp_l in zip(self.FAN_SPEEDS, self.TEMP_THRS_L) if temp_l > max_cpu_temp)

    if new_speed_h > self.fan_speed:
      self.set_eon_fan(new_speed_h)
    elif new_speed_l < self.fan_speed:
      self.set_eon_fan(new_speed_l)

    return self.fan_speed


class UnoFanController(BaseFanController):
  def __init__(self) -> None:
    super().__init__()
    cloudlog.info("Setting up UNO fan handler")

  def update(self, max_cpu_temp: float, ignition: bool) -> int:
    new_speed = int(interp(max_cpu_temp, [40.0, 80.0], [0, 80]))

    if not ignition:
      new_speed = min(30, new_speed)

    return new_speed


class TiciFanController(BaseFanController):
  def __init__(self) -> None:
    super().__init__()
    cloudlog.info("Setting up TICI fan handler")

    self.last_ignition = False
    self.controller = PIDController(k_p=0, k_i=2e-3, k_f=1, neg_limit=-80, pos_limit=0, rate=(1 / DT_TRML))

  def update(self, max_cpu_temp: float, ignition: bool) -> int:
    self.controller.neg_limit = -(80 if ignition else 30)
    self.controller.pos_limit = -(30 if ignition else 0)

    if ignition != self.last_ignition:
      self.controller.reset()

    error = 75 - max_cpu_temp
    fan_pwr_out = -int(self.controller.update(
                      error=error,
                      feedforward=interp(max_cpu_temp, [60.0, 100.0], [0, -80])
                    ))

    self.last_ignition = ignition
    return fan_pwr_out

