#!/usr/bin/env python3
import sys
import time
import datetime

from openpilot.common.realtime import Ratekeeper
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.tools.zookeeper import Zookeeper

if __name__ == "__main__":
  z = Zookeeper()
  z.set_device_power(True)
  z.set_device_ignition(False)

  duration = None
  if len(sys.argv) > 1:
    duration = int(sys.argv[1])

  rate = 123
  rk = Ratekeeper(rate, print_delay_threshold=None)
  fltr = FirstOrderFilter(0, 5, 1. / rate, initialized=False)

  measurements = []
  start_time = time.monotonic()

  try:
    while duration is None or time.monotonic() - start_time < duration:
      fltr.update(z.read_power())
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
