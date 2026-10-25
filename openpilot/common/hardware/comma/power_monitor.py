#!/usr/bin/env python3
import sys
import time
import datetime
import numpy as np
from collections import deque

from openpilot.common.realtime import Ratekeeper
from openpilot.common.filter_simple import FirstOrderFilter


def read_power():
  with open("/sys/bus/i2c/devices/0-0040/hwmon/hwmon1/power1_input") as f:
    return int(f.read()) / 1e6

def sample_power(seconds=5) -> list[float]:
  rate = 123
  rk = Ratekeeper(rate, print_delay_threshold=None)

  pwrs = []
  for _ in range(rate*seconds):
    pwrs.append(read_power())
    rk.keep_time()
  return pwrs

def get_power(seconds=5):
  pwrs = sample_power(seconds)
  return np.mean(pwrs)

def wait_for_power(min_pwr, max_pwr, min_secs_in_range, timeout):
  start_time = time.monotonic()
  pwrs = deque([min_pwr - 1.]*min_secs_in_range, maxlen=min_secs_in_range)
  while (time.monotonic() - start_time < timeout):
    pwrs.append(get_power(1))
    if all(min_pwr <= p <= max_pwr for p in pwrs):
      break
  return np.mean(pwrs)


if __name__ == "__main__":
  duration = None
  if len(sys.argv) > 1:
    duration = int(sys.argv[1])

  rate = 23
  rk = Ratekeeper(rate, print_delay_threshold=None)
  fltr = FirstOrderFilter(0, 5, 1. / rate, initialized=False)

  measurements = []
  start_time = time.monotonic()

  try:
    while duration is None or time.monotonic() - start_time < duration:
      fltr.update(read_power())
      if rk.frame % rate == 0:
        measurements.append(fltr.x)
        t = datetime.timedelta(seconds=time.monotonic() - start_time)
        avg = sum(measurements) / len(measurements)
        print(f"Now: {fltr.x:.2f} W, Avg: {avg:.2f} W over {t}")
      rk.keep_time()
  except KeyboardInterrupt:
    pass

  t = datetime.timedelta(seconds=time.monotonic() - start_time)
  avg = sum(measurements) / len(measurements)
  print(f"\nAverage power: {avg:.2f}W over {t}")
