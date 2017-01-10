"""Utilities for reading real time clocks and keeping soft real time constraints."""
import time
import ctypes
import platform
import subprocess
import multiprocessing
import os

CLOCK_MONOTONIC_RAW = 4 # see <linux/time.h>
CLOCK_BOOTTIME = 7

class timespec(ctypes.Structure):
  _fields_ = [
    ('tv_sec', ctypes.c_long),
    ('tv_nsec', ctypes.c_long),
  ]


try:
  libc = ctypes.CDLL('libc.so', use_errno=True)
except OSError:
  try:
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
  except OSError:
    libc = None

if libc is not None:
  libc.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(timespec)]

def clock_gettime(clk_id):
  if platform.system().lower() == "darwin":
    # TODO: fix this
    return time.time()
  else:
    t = timespec()
    if libc.clock_gettime(clk_id, ctypes.pointer(t)) != 0:
      errno_ = ctypes.get_errno()
      raise OSError(errno_, os.strerror(errno_))
    return t.tv_sec + t.tv_nsec * 1e-9

def monotonic_time():
  return clock_gettime(CLOCK_MONOTONIC_RAW)

def sec_since_boot():
  return clock_gettime(CLOCK_BOOTTIME)


def set_realtime_priority(level):
  if os.getuid() != 0:
    print("not setting priority, not root")
    return
  if platform.machine() == "x86_64":
    NR_gettid = 186
  elif platform.machine() == "aarch64":
    NR_gettid = 178
  else:
    raise NotImplementedError

  tid = libc.syscall(NR_gettid)
  return subprocess.call(['chrt', '-f', '-p', str(level), str(tid)])


class Ratekeeper(object):
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
    if remaining < -self._print_delay_threshold:
      print(self._process_name, "lagging by", round(-remaining * 1000, 2), "ms")
      lagged = True
    self._frame += 1
    self._remaining = remaining
    return lagged
