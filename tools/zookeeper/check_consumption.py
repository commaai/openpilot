#!/usr/bin/env python

import sys
import time
from tools.zookeeper import Zookeeper

# Usage: check_consumption.py <averaging_time_sec> <max_average_power_W>
# Exit code: 0 -> passed
#	           1 -> failed

z = Zookeeper()

averaging_time_s = int(sys.argv[1])
max_average_power = float(sys.argv[2])

start_time = time.time()
measurements = []
while time.time() - start_time < averaging_time_s:
  measurements.append(z.read_power())
  time.sleep(0.1)

average_power = sum(measurements)/len(measurements)
print(f"Average power: {round(average_power, 4)}W")

if average_power > max_average_power:
  exit(1)




