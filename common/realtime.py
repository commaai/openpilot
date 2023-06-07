"""Utilities for reading real time clocks and keeping soft real time constraints."""
import gc
import os
import time
from collections import deque
from typing import Optional, List, Union

from setproctitle import getproctitle  # pylint: disable=no-name-in-module

from common.clock import sec_since_boot  # pylint: disable=no-name-in-module, import-error
from system.hardware import PC


# time step for each process
DT_CTRL = 0.01  # controlsd
DT_MDL = 0.05  # model
DT_TRML = 0.5  # thermald and manager
DT_DMON = 0.05  # driver monitoring


class Priority:
  # CORE 2
  # - modeld = 55
  # - camerad = 54
  CTRL_LOW = 51 # plannerd & radard

  # CORE 3
  # - boardd = 55
  CTRL_HIGH = 53


def set_realtime_priority(level: int) -> None:
  if not PC:
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(level))  # pylint: disable=no-member


def set_core_affinity(cores: List[int]) -> None:
  if not PC:
    os.sched_setaffinity(0, cores)  # pylint: disable=no-member


def config_realtime_process(cores: Union[int, List[int]], priority: int) -> None:
  gc.disable()
  set_realtime_priority(priority)
  c = cores if isinstance(cores, list) else [cores, ]
  set_core_affinity(c)


class Ratekeeper:
  def __init__(self, rate: float, print_delay_threshold: Optional[float] = 0.0) -> None:
    """Rate in Hz for ratekeeping. print_delay_threshold must be nonnegative."""
    self._interval = 1. / rate
    self._next_frame_time = sec_since_boot() + self._interval
    self._print_delay_threshold = print_delay_threshold
    self._frame = 0
    self._remaining = 0.0
    self._process_name = getproctitle()
    self._dts = deque([self._interval], maxlen=100)
    self._last_monitor_time = sec_since_boot()

  @property
  def frame(self) -> int:
    return self._frame

  @property
  def remaining(self) -> float:
    return self._remaining

  @property
  def lagging(self) -> bool:
    avg_dt = sum(self._dts) / len(self._dts)
    expected_dt = self._interval * (1 / 0.9)
    return avg_dt > expected_dt

  # Maintain loop rate by calling this at the end of each loop
  def keep_time(self) -> bool:
    lagged = self.monitor_time()
    if self._remaining > 0:
      time.sleep(self._remaining)
    return lagged

  # this only monitor the cumulative lag, but does not enforce a rate
  def monitor_time(self) -> bool:
    prev = self._last_monitor_time
    self._last_monitor_time = sec_since_boot()
    self._dts.append(self._last_monitor_time - prev)

    lagged = False
    remaining = self._next_frame_time - sec_since_boot()
    self._next_frame_time += self._interval
    if self._print_delay_threshold is not None and remaining < -self._print_delay_threshold:
      print(f"{self._process_name} lagging by {-remaining * 1000:.2f} ms")
      lagged = True
    self._frame += 1
    self._remaining = remaining
    return lagged
