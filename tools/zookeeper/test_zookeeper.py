#!/usr/bin/env python3

import time
from tools.zookeeper import Zookeeper

z = Zookeeper()
z.set_device_power(True)

i = 0
ign = False
while 1:
  voltage = round(z.read_voltage(), 2)
  current = round(z.read_current(), 3)
  power = round(z.read_power(), 2)
  z.set_device_ignition(ign)
  print(f"Voltage: {voltage}V, Current: {current}A, Power: {power}W, Ignition: {ign}")

  if i > 200:
    ign = not ign
    i = 0
  
  i += 1
  time.sleep(0.1)
