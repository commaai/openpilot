import struct

import common.numpy_fast as np
from selfdrive.config import Conversions as CV
from common.fingerprints import TESLA as CAR

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


#def create_brake_command(packer, apply_brake, pcm_override, pcm_cancel_cmd, chime, fcw, idx):
#  """Creates a CAN message for the Honda DBC BRAKE_COMMAND."""
#  pump_on = apply_brake > 0
#  brakelights = apply_brake > 0
#  brake_rq = apply_brake > 0
#  pcm_fault_cmd = False
#
#  values = {
#    "COMPUTER_BRAKE": apply_brake,
#    "COMPUTER_BRAKE_REQUEST": pump_on,
#    "CRUISE_OVERRIDE": pcm_override,
#    "CRUISE_FAULT_CMD": pcm_fault_cmd,
#    "CRUISE_CANCEL_CMD": pcm_cancel_cmd,
#    "COMPUTER_BRAKE_REQUEST_2": brake_rq,
#    "SET_ME_0X80": 0x80,
#    "BRAKE_LIGHTS": brakelights,
#    "CHIME": chime,
#    "FCW": fcw << 1,  # TODO: Why are there two bits for fcw? According to dbc file the first bit should also work
#  }
#  return packer.make_can_msg("BRAKE_COMMAND", 0, values, idx)


#def create_gas_command(packer, gas_amount, idx):
#  """Creates a CAN message for the Honda DBC GAS_COMMAND."""
#  enable = gas_amount > 0.001
#
#  values = {"ENABLE": enable}
#
#  if enable:
#    values["GAS_COMMAND"] = gas_amount * 255.
#    values["GAS_COMMAND2"] = gas_amount * 255.
#
#  return packer.make_can_msg("GAS_COMMAND", 0, values, idx)


def create_steering_control(packer, enabled, apply_steer, car_fingerprint, idx):
  """Creates a CAN message for the Tesla DBC DAS_steeringControl."""
  
  if enabled == False:
    steering_type = 0
  else:
    steering_type = 1
  type_counter = steering_type << 6
  type_counter += idx
  checksum = ((apply_steer & 0xFF) + ((apply_steer >> 8) & 0xFF) + type_counter + 0x8C) & 0xFF  
  msg = struct.pack("!hBB", apply_steer, type_counter, checksum)
  
  values = {
    "DAS_steeringHapticRequest": 0,
    "DAS_steeringAngleRequest": apply_steer,
    "DAS_steeringControlCounter": idx,
    "DAS_steeringControlChecksum": checksum,
    "DAS_steeringControlType": steering_type,
  }
  
  #return packer.make_can_msg("DAS_steeringControl", 0, values, idx)
  return [0x488, 0, msg, 1]

#def create_steering_control(packer, apply_steer, idx, controls_allowed):
#  """Creates a CAN message for the Tesla EPAS STEERING_CONTROL."""
#  """BO_ 1160 DAS_steeringControl: 4 NEO
#      SG_ DAS_steeringControlType : 23|2@0+ (1,0) [0|0] "" EPAS
#      SG_ DAS_steeringControlChecksum : 31|8@0+ (1,0) [0|0] "" EPAS
#      SG_ DAS_steeringControlCounter : 19|4@0+ (1,0) [0|0] "" EPAS
#      SG_ DAS_steeringAngleRequest : 6|15@0+ (0.1,-1638.35) [-1638.35|1638.35] "deg" EPAS
#      SG_ DAS_steeringHapticRequest : 7|1@0+ (1,0) [0|0] "" EPAS
#  """
#  if controls_allowed == False:
#    steering_type = 0
#  else:
#    steering_type = 1
#  type_counter = steering_type << 6
#  type_counter += idx
#  checksum = ((apply_steer & 0xFF) + ((apply_steer >> 8) & 0xFF) + type_counter + 0x8C) & 0xFF  
#  msg = struct.pack("!hBB", apply_steer, type_counter, checksum)
#  #TODO: change 0x489 to 0x488 for production
#  return [0x488, 0, msg, 1]


#def create_ui_commands(packer, pcm_speed, hud, car_fingerprint, idx):
#  """Creates an iterable of CAN messages for the UIs."""
#  commands = []
#
#  acc_hud_values = {
#    'PCM_SPEED': pcm_speed * CV.MS_TO_KPH,
#    'PCM_GAS': hud.pcm_accel,
#    'CRUISE_SPEED': hud.v_cruise,
#    'ENABLE_MINI_CAR': hud.mini_car,
#    'HUD_LEAD': hud.car,
#    'SET_ME_X03': 0x03,
#    'SET_ME_X03_2': 0x03,
#    'SET_ME_X01': 0x01,
#  }
#  commands.append(packer.make_can_msg("ACC_HUD", 0, acc_hud_values, idx))
#
#  lkas_hud_values = {
#    'SET_ME_X41': 0x41,
#    'SET_ME_X48': 0x48,
#    'STEERING_REQUIRED': hud.steer_required,
#    'SOLID_LANES': hud.lanes,
#    'BEEP': hud.beep,
#  }
#  commands.append(packer.make_can_msg('LKAS_HUD', 0, lkas_hud_values, idx))
#
#  if car_fingerprint in (CAR.CIVIC, CAR.ODYSSEY):
#    commands.append(packer.make_can_msg('HIGHBEAM_CONTROL', 0, {'HIGHBEAMS_ON': False}, idx))
#
#    radar_hud_values = {
#      'ACC_ALERTS': hud.acc_alert,
#      'LEAD_SPEED': 0x1fe,  # What are these magic values
#      'LEAD_STATE': 0x7,
#      'LEAD_DISTANCE': 0x1e,
#    }
#    commands.append(packer.make_can_msg('RADAR_HUD', 0, radar_hud_values, idx))
#  return commands
#
#
#def create_radar_commands(v_ego, car_fingerprint, idx):
#  """Creates an iterable of CAN messages for the radar system."""
#  commands = []
#  v_ego_kph = np.clip(int(round(v_ego * CV.MS_TO_KPH)), 0, 255)
#  speed = struct.pack('!B', v_ego_kph)
#
#  msg_0x300 = ("\xf9" + speed + "\x8a\xd0" +
#               ("\x20" if idx == 0 or idx == 3 else "\x00") +
#               "\x00\x00")
#
#  if car_fingerprint == CAR.CIVIC:
#    msg_0x301 = "\x02\x38\x44\x32\x4f\x00\x00"
#    commands.append(make_can_msg(0x300, msg_0x300, idx + 8, 1))  # add 8 on idx.
#  else:
#    if car_fingerprint == CAR.CRV:
#      msg_0x301 = "\x00\x00\x50\x02\x51\x00\x00"
#    elif car_fingerprint == CAR.ACURA_RDX:
#      msg_0x301 = "\x0f\x57\x4f\x02\x5a\x00\x00"
#    elif car_fingerprint == CAR.ODYSSEY:
#      msg_0x301 = "\x00\x00\x56\x02\x55\x00\x00"
#    elif car_fingerprint == CAR.ACURA_ILX:
#      msg_0x301 = "\x0f\x18\x51\x02\x5a\x00\x00"
#    elif car_fingerprint == CAR.PILOT:
#      msg_0x301 = "\x00\x00\x56\x02\x58\x00\x00"
#    elif car_fingerprint == CAR.RIDGELINE:
#      msg_0x301 = "\x00\x00\x56\x02\x57\x00\x00"
#    commands.append(make_can_msg(0x300, msg_0x300, idx, 1))
#
#  commands.append(make_can_msg(0x301, msg_0x301, idx, 1))
#  return commands
