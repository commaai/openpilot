import struct

import common.numpy_fast as np
from selfdrive.config import Conversions as CV


# *** Honda specific ***
def can_cksum(mm):
  s = 0
  for c in mm:
    c = ord(c)
    s += (c>>4)
    s += c & 0xF
  s = 8-s
  s %= 0x10
  return s

def fix(msg, addr):
  msg2 = msg[0:-1] + chr(ord(msg[-1]) | can_cksum(struct.pack("I", addr)+msg))
  return msg2

def make_can_msg(addr, dat, idx, alt):
  if idx is not None:
    dat += chr(idx << 4)
    dat = fix(dat, addr)
  return [addr, 0, dat, alt]

def create_brake_command(apply_brake, pcm_override, pcm_cancel_cmd, chime, idx):
  """Creates a CAN message for the Honda DBC BRAKE_COMMAND."""
  return []

def create_gas_command(gas_amount, idx):
  """Creates a CAN message for the Honda DBC GAS_COMMAND."""
  return []

def create_steering_control(apply_steer, idx):
  """Creates a CAN message for the Honda DBC STEERING_CONTROL."""
#   msg = struct.pack("!h", apply_steer) + ("\x80\x00" if apply_steer != 0 else "\x00\x00")
#   return make_can_msg(0xe4, msg, idx, 0)

def create_ui_commands(pcm_speed, hud, civic, idx):
  """Creates an iterable of CAN messages for the UIs."""
  commands = []
#   if civic:  # 2 more msgs
#     msg_0x35e = chr(0) * 7
#     commands.append(make_can_msg(0x35e, msg_0x35e, idx, 0))
  return commands

def create_radar_commands(v_ego, civic, idx):
  """Creates an iterable of CAN messages for the radar system."""
  commands = []
  return commands
