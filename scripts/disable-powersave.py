#!/usr/bin/env python3
from openpilot.system.hardware import HARDWARE

if __name__ == "__main__":
  HARDWARE.set_power_save(False)
