#!/usr/bin/env python3
import time

from panda import Panda

power = 0
if __name__ == "__main__":
  p = Panda()
  while True:
    p.set_ir_power(power)
    print("Power: ", str(power))
    time.sleep(1)
    power += 10
    power %= 100
