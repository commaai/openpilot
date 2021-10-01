#!/usr/bin/env python

import sys
import time
from tools.zookeeper import Zookeeper

# Usage: check_consumption.py <averaging_time_sec> <max_average_power_W>
# Exit code: 0 -> passed
#	           1 -> failed

if __name__ == "__main__":
  z = Zookeeper()

  duration = None
  if len(sys.argv) > 1:
    duration = int(sys.argv[1])

  try:
    start_time = time.monotonic()
    measurements = []
    while duration is None or time.monotonic() - start_time < duration:
      p = z.read_power()
      print(round(p, 3), "W")
      measurements.append(p)
      time.sleep(0.25)
  except KeyboardInterrupt:
    pass
  finally:
    average_power = sum(measurements)/len(measurements)
    print(f"Average power: {round(average_power, 4)}W")
