#!/usr/bin/env python3
import numpy as np
from common.realtime import Ratekeeper

if __name__ == '__main__':
  RATE = 123
  print("measuring for 5 seconds at %dhz 3 times" % RATE)
  rk = Ratekeeper(RATE, print_delay_threshold=None)

  for _ in range(3):
    pwrs = []
    for _ in range(RATE*5):
      with open("/sys/bus/i2c/devices/0-0040/hwmon/hwmon1/power1_input") as f:
        pwrs.append(int(f.read()) / 1000.)
      rk.keep_time()
    print("mean %.2f std %.2f" % (np.mean(pwrs), np.std(pwrs)))

