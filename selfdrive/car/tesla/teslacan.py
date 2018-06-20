import struct
from ctypes import create_string_buffer


def add_tesla_checksum(msg_id,msg):
 """Calculates the checksum for the data part of the Tesla message"""
 checksum = ((msg_id) & 0xFF) + ((msg_id >> 8) & 0xFF)
 for i in range(0,len(msg),1):
  checksum = (checksum + ord(msg[i])) & 0xFF
 return checksum


def create_steering_control(enabled, apply_steer, idx):
 """Creates a CAN message for the Tesla DBC DAS_steeringControl."""
 msg_id = 0x488
 msg_len = 4
 msg = create_string_buffer(msg_len)
 if enabled == False:
  steering_type = 0
 else:
  steering_type = 1
 type_counter = steering_type << 6
 type_counter += idx
 struct.pack_into('!hB', msg, 0,  apply_steer, type_counter)
 struct.pack_into('B', msg, msg_len-1, add_tesla_checksum(msg_id,msg))
 return [msg_id, 0, msg.raw, 2]


def create_epb_enable_signal(idx):
  """Creates a CAN message to simulate EPB enable message"""
  msg_id = 0x214
  msg_len = 3
  msg = create_string_buffer(msg_len)
  struct.pack_into('BB', msg, 0,  1, idx)
  struct.pack_into('B', msg, msg_len-1, add_tesla_checksum(msg_id,msg))
  return [msg_id, 0, msg.raw, 2]
  
