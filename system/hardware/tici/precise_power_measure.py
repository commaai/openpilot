#!/usr/bin/env python3
import numpy as np
from openpilot.system.hardware.tici.power_monitor import sample_power

if __name__ == '__main__':
  print("measuring for 5 seconds")
  for _ in range(3):
    pwrs = sample_power()
    print(f"mean {np.mean(pwrs):.2f} std {np.std(pwrs):.2f}")
