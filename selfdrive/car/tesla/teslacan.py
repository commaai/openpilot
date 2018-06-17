import struct

import common.numpy_fast as np
from selfdrive.config import Conversions as CV


def create_steering_control(enabled, apply_steer, idx):
  """Creates a CAN message for the Tesla DBC DAS_steeringControl."""
  if enabled == False:
    steering_type = 0
  else:
    steering_type = 1
  type_counter = steering_type << 6
  type_counter += idx
  checksum = ((apply_steer & 0xFF) + ((apply_steer >> 8) & 0xFF) + type_counter + 0x8C) & 0xFF  
  msg = struct.pack("!hBB", apply_steer, type_counter, checksum)
  
  return [0x488, 0, msg, 2]


def create_epb_enable_signal(idx):
  """Creates a CAN message to simulate EPB enable message"""
  checksum = (0x1 + idx + 0x16) & 0xFF
  msg = struct.pack("!BBB", 1, idx, checksum)

  return [0x214, 0, msg, 2]
  