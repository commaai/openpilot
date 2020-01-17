#!/usr/bin/env python3
import time
from smbus2 import SMBus

def setup_leon_fan():
  bus = SMBus(7, force=True)

  # https://www.nxp.com/docs/en/data-sheet/PTN5150.pdf
  j = 0
  for i in [0x1, 0x3 | 0, 0x3 | 0x08, 0x3 | 0x10]:
    print("FAN SPEED", j)
    ret = bus.read_i2c_block_data(0x3d, 0, 4)
    print(ret)
    ret = bus.write_i2c_block_data(0x3d, 0, [i])
    time.sleep(1)
    ret = bus.read_i2c_block_data(0x3d, 0, 4)
    print(ret)
    j += 1

  bus.close()

setup_leon_fan()

