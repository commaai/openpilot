#!/usr/bin/env python3
# pylint: skip-file

import time
from smbus2 import SMBus

def setup_leon_fan():
  bus = SMBus(7, force=True)

  # http://www.ti.com/lit/ds/symlink/tusb320.pdf
  for i in [0,1,2,3]:
    print("FAN SPEED", i)
    if i == 0:
      bus.write_i2c_block_data(0x67, 0xa, [0])
    else:
      bus.write_i2c_block_data(0x67, 0xa, [0x20])
      bus.write_i2c_block_data(0x67, 0x8, [(i-1)<<6])
    time.sleep(1)

  bus.close()

setup_leon_fan()
