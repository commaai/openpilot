#!/usr/bin/env python3
import os
from smbus2 import SMBus

def setup_fan():
  os.system("echo 2 > /sys/module/dwc3_msm/parameters/otg_switch")

  bus = SMBus(7, force=True)
  try:
    bus.write_byte_data(0x21, 0x10, 0xf)   # mask all interrupts
    bus.write_byte_data(0x21, 0x03, 0x1)   # set drive current and global interrupt disable
    bus.write_byte_data(0x21, 0x02, 0x2)   # needed?
    bus.write_byte_data(0x21, 0x04, 0x4)   # manual override source
    print("OP detected")
    ret = False
  except IOError:
    print("LEON detected")
    ret = True
  bus.close()
  return ret

def get_fan_type():
  if not setup_fan():
    return

  bus = SMBus(7, force=True)
  try:
    # alternate type
    bus.write_i2c_block_data(0x3d, 0, [0x1])
    print("Alternate type detected")
    return
  except IOError:
    # tusb320 type
    print("tusb320 type detected")
  bus.close()


def main(gctx=None):
  get_fan_type()

if __name__ == "__main__":
  main()
