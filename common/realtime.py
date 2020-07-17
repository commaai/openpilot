"""Utilities for reading real time clocks and keeping soft real time constraints."""
import os
import time
import platform
import subprocess
import multiprocessing
from cffi import FFI

from common.android import ANDROID
from common.common_pyx import sec_since_boot  # pylint: disable=no-name-in-module, import-error


# time step for each process
DT_CTRL = 0.01  # controlsd
DT_MDL = 0.05  # model
DT_DMON = 0.1  # driver monitoring
DT_TRML = 0.5  # thermald and manager


ffi = FFI()
ffi.cdef("long syscall(long number, ...);")
libc = ffi.dlopen(None)

def _get_tid():
  if platform.machine() == "x86_64":
    NR_gettid = 186
  elif platform.machine() == "aarch64":
    NR_gettid = 178
  else:
    raise NotImplementedError

  return libc.syscall(NR_gettid)


def set_realtime_priority(level):
  if os.getuid() != 0:
    print("not setting priority, not root")
    return

  return subprocess.call(['chrt', '-f', '-p', str(level), str(_get_tid())])

def set_core_affinity(core):
  if os.getuid() != 0:
    print("not setting affinity, not root")
    return

  if ANDROID:
    return subprocess.call(['taskset', '-p', str(core), str(_get_tid())])
  else:
    return -1


class Ratekeeper():
  def __init__(self, rate, print_delay_threshold=0.):
    """Rate in Hz for ratekeeping. print_delay_threshold must be nonnegative."""
    self._interval = 1. / rate
    self._next_frame_time = sec_since_boot() + self._interval
    self._print_delay_threshold = print_delay_threshold
    self._frame = 0
    self._remaining = 0
    self._process_name = multiprocessing.current_process().name

  @property
  def frame(self):
    return self._frame

  @property
  def remaining(self):
    return self._remaining

  # Maintain loop rate by calling this at the end of each loop
  def keep_time(self):
    lagged = self.monitor_time()
    if self._remaining > 0:
      time.sleep(self._remaining)
    return lagged

  # this only monitor the cumulative lag, but does not enforce a rate
  def monitor_time(self):
    lagged = False
    remaining = self._next_frame_time - sec_since_boot()
    self._next_frame_time += self._interval
    if self._print_delay_threshold is not None and remaining < -self._print_delay_threshold:
      print("%s lagging by %.2f ms" % (self._process_name, -remaining * 1000))
      lagged = True
    self._frame += 1
    self._remaining = remaining
    return lagged
