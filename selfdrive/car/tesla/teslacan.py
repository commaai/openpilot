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
  pump_on = apply_brake > 0
  brakelights = apply_brake > 0
  brake_rq = apply_brake > 0

  pcm_fault_cmd = False
  amount = struct.pack("!H", (apply_brake << 6) + pump_on)
  msg = amount + struct.pack("BBB", (pcm_override << 4) |
                             (pcm_fault_cmd << 2) |
                             (pcm_cancel_cmd << 1) | brake_rq, 0x80,
                             brakelights << 7) + chr(chime) + "\x00"
  return make_can_msg(0x1fa, msg, idx, 0)

def create_gas_command(gas_amount, idx):
  """Creates a CAN message for the Tesla DBC acceleration command."""
  msg = struct.pack("!H", gas_amount)
  return make_can_msg(0x200, msg, idx, 0)

def create_sound_packet(command):
  msg = struct.pack("B", command)
  return [0x720, 0, msg, 0]

def create_steering_control(apply_steer, idx, controls_allowed):
  """Creates a CAN message for the Tesla EPAS STEERING_CONTROL."""
  """BO_ 1160 DAS_steeringControl: 4 NEO
       SG_ DAS_steeringControlType : 23|2@0+ (1,0) [0|0] ""  EPAS
       SG_ DAS_steeringControlChecksum : 31|8@0+ (1,0) [0|0] ""  EPAS
       SG_ DAS_steeringControlCounter : 19|4@0+ (1,0) [0|0] ""  EPAS
       SG_ DAS_steeringAngleRequest : 6|15@0+ (0.1,-1638.35) [-1638.35|1638.35] "deg"  EPAS
       SG_ DAS_steeringHapticRequest : 7|1@0+ (1,0) [0|0] ""  EPAS"""
  if controls_allowed == False:
    steering_type = 0
  else:
    steering_type = 1
  type_counter = steering_type << 6
  type_counter += idx
  checksum = ((apply_steer & 0xFF) + ((apply_steer >> 8) & 0xFF) + type_counter + 0x8C) & 0xFF  
  msg = struct.pack("!hBB", apply_steer, type_counter, checksum)
  #TODO: change 0x489 to 0x488 for production
  return [0x488, 0, msg, 1]

def create_ui_commands(pcm_speed, hud, idx):
  """Creates an iterable of CAN messages for the UIs."""
  commands = []
  pcm_speed_real = np.clip(int(round(pcm_speed / 0.002759506)), 0,
                           64000)  # conversion factor from dbc file
  msg_0x30c = struct.pack("!HBBBBB", pcm_speed_real, hud.pcm_accel,
                          hud.v_cruise, hud.X2, hud.car, hud.X4)
  commands.append(make_can_msg(0x30c, msg_0x30c, idx, 0))

  msg_0x33d = chr(hud.X5) + chr(hud.lanes) + chr(hud.beep) + chr(hud.X8)
  commands.append(make_can_msg(0x33d, msg_0x33d, idx, 0))
  #if civic:  # 2 more msgs
  #  msg_0x35e = chr(0) * 7
  #  commands.append(make_can_msg(0x35e, msg_0x35e, idx, 0))
  #if civic or accord:
  #  msg_0x39f = (
  #    chr(0) * 2 + chr(hud.acc_alert) + chr(0) + chr(0xff) + chr(0x7f) + chr(0)
  #  )
  #  commands.append(make_can_msg(0x39f, msg_0x39f, idx, 0))
  return commands

def create_radar_commands(v_ego, idx):
  """Creates an iterable of CAN messages for the radar system."""
  commands = []
  v_ego_kph = np.clip(int(round(v_ego * CV.MS_TO_KPH)), 0, 255)
  speed = struct.pack('!B', v_ego_kph)
  msg_0x300 = ("\xf9" + speed + "\x8a\xd0" +\
              ("\x20" if idx == 0 or idx == 3 else "\x00") +\
                "\x00\x00")
  #if civic:
  #  msg_0x301 = "\x02\x38\x44\x32\x4f\x00\x00"
    # add 8 on idx.
  #  commands.append(make_can_msg(0x300, msg_0x300, idx + 8, 1))
  #elif accord:
    # 0300( 768)(    69) f9008ad0100000ef
    # 0301( 769)(    69) 0ed8522256000029
  #  msg_0x301 = "\x0e\xd8\x52\x22\x56\x00\x00"
    # add 0xc on idx? WTF is this?
  #  commands.append(make_can_msg(0x300, msg_0x300, idx + 0xc, 1))
  #else:
  #  msg_0x301 = "\x0f\x18\x51\x02\x5a\x00\x00"
  #  commands.append(make_can_msg(0x300, msg_0x300, idx, 1))
  commands.append(make_can_msg(0x301, msg_0x301, idx, 1))
  return commands