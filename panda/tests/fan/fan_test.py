#!/usr/bin/env python
import time

from panda import Panda

if __name__ == "__main__":
  p = Panda()
  power = 0
  while True:
    p.set_fan_power(power)
    time.sleep(5)
    print("Power: ", power, "RPM:", str(p.get_fan_rpm()), "Expected:", int(6500 * power / 100))
    power += 10
    power %= 110
