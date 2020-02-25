#!/usr/bin/env python
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda

power = 0
if __name__ == "__main__":
  p = Panda()
  while True:
    p.set_fan_power(power)
    time.sleep(5)
    print("Power: ", power, "RPM: ", str(p.get_fan_rpm()))
    power += 10
    power %=100
