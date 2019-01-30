#!/usr/bin/env python2.7
import sys
from smbus2 import SMBus

def set_eon_fan(val):
    bus = SMBus(7, force=True)
    bus.write_byte_data(0x21, 0x04, 0x2)
    bus.write_byte_data(0x21, 0x03, (val*2)+1)
    bus.write_byte_data(0x21, 0x04, 0x4)
    bus.close()

set_eon_fan(int(sys.argv[1]))