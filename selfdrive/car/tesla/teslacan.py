import struct

import common.numpy_fast as np
from selfdrive.config import Conversions as CV
from common.fingerprints import TESLA as CAR

gtw_epas_control_msg = [struct.pack("!BBB", 0x0C, 0xC0, 0xCE),
                        struct.pack("!BBB", 0x0C, 0xC1, 0xCF),
                        struct.pack("!BBB", 0x0C, 0xC2, 0xD0),
                        struct.pack("!BBB", 0x0C, 0xC3, 0xD1),
                        struct.pack("!BBB", 0x0C, 0xC4, 0xD2),
                        struct.pack("!BBB", 0x0C, 0xC5, 0xD3),
                        struct.pack("!BBB", 0x0C, 0xC6, 0xD4),
                        struct.pack("!BBB", 0x0C, 0xC7, 0xD5),
                        struct.pack("!BBB", 0x0C, 0xC8, 0xD6),
                        struct.pack("!BBB", 0x0C, 0xC9, 0xD7),
                        struct.pack("!BBB", 0x0C, 0xCA, 0xD8),
                        struct.pack("!BBB", 0x0C, 0xCB, 0xD9),
                        struct.pack("!BBB", 0x0C, 0xCC, 0xDA),
                        struct.pack("!BBB", 0x0C, 0xCD, 0xDB),
                        struct.pack("!BBB", 0x0C, 0xCE, 0xDC)]
                        

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


def create_gtw_enable_signal(idx):
  """Creates a CAN message to simulate EPB enable message"""

  return [0x101, 0, gtw_epas_control_msg[idx], 2]