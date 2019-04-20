"""Utilities for reading real time clocks and keeping soft real time constraints."""
import os
import time
import platform
import threading
import subprocess
import multiprocessing

from cffi import FFI
ffi = FFI()
ffi.cdef("""

typedef int clockid_t;
struct timespec {
    long tv_sec;   /* Seconds.  */
    long tv_nsec;  /* Nanoseconds.  */
};
int clock_gettime (clockid_t clk_id, struct timespec *tp);

long syscall(long number, ...);

"""
)
libc = ffi.dlopen(None)


# see <linux/time.h>
CLOCK_MONOTONIC_RAW = 4
CLOCK_BOOTTIME = 7

if platform.system() != 'Darwin' and hasattr(libc, 'clock_gettime'):
  c_clock_gettime = libc.clock_gettime

  tlocal = threading.local()
  def clock_gettime(clk_id):
    if not hasattr(tlocal, 'ts'):
      tlocal.ts = ffi.new('struct timespec *')

    ts = tlocal.ts

    r = c_clock_gettime(clk_id, ts)
    if r != 0:
      raise OSError("clock_gettime")
    return ts.tv_sec + ts.tv_nsec * 1e-9
else:
  # hack. only for OS X < 10.12
  def clock_gettime(clk_id):
    return time.time()

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
      print("%s lagging by %.2f ms" % (self._process_name, -remaining * 1000))
      lagged = True
    self._frame += 1
    self._remaining = remaining
    return lagged

