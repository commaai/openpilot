import struct
from selfdrive.config import Conversions as CV
from ctypes import create_string_buffer
from selfdrive.car.honda.values import CAR, HONDA_BOSCH

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


def create_brake_command(packer, apply_brake, pump_on, pcm_override, pcm_cancel_cmd, chime, fcw, idx):
  # TODO: do we loose pressure if we keep pump off for long?
  brakelights = apply_brake > 0
  brake_rq = apply_brake > 0
  pcm_fault_cmd = False

  values = {
    "COMPUTER_BRAKE": apply_brake,
    "BRAKE_PUMP_REQUEST": pump_on,
    "CRUISE_OVERRIDE": pcm_override,
    "CRUISE_FAULT_CMD": pcm_fault_cmd,
    "CRUISE_CANCEL_CMD": pcm_cancel_cmd,
    "COMPUTER_BRAKE_REQUEST": brake_rq,
    "SET_ME_0X80": 0x80,
    "BRAKE_LIGHTS": brakelights,
    "CHIME": chime,
    # TODO: Why are there two bits for fcw? According to dbc file the first bit should also work
    "FCW": fcw << 1,
  }
  return packer.make_can_msg("BRAKE_COMMAND", 0, values, idx)


def create_gas_command(packer, gas_amount, idx):
  enable = gas_amount > 0.001

  values = {"ENABLE": enable}

  if enable:
    values["GAS_COMMAND"] = gas_amount * 255.
    values["GAS_COMMAND2"] = gas_amount * 255.

  return packer.make_can_msg("GAS_COMMAND", 0, values, idx)

def create_acc_commands(packer, enabled, accel, fingerprint, idx):
  commands = []

  # 0 = off
  # 5 = on
  control_on = 5 if enabled else 0
  # 0  = gas
  # 17 = no gas
  # 31 = ?!?!
  state_flag = 0 if enabled and accel > 0 else 17
  # 0 to +2000? = range
  # 720 = no gas
  # (scale from a max of 800 to 2000)
  gas_command = int(accel) if enabled and accel > 0 else 720
  # 1 = brake
  # 0 = no brake
  braking_flag = 1 if enabled and accel < 0 else 0
  # -1599 to +800? = range
  # 0 = no accel
  gas_brake = int(accel) if enabled else 0

  acc_control_values = {
    "GAS_COMMAND": gas_command,
    "STATE_FLAG": state_flag,
    "BRAKING_1": braking_flag,
    "BRAKING_2": braking_flag,
    # setting CONTROL_ON causes car to set POWERTRAIN_DATA->ACC_STATUS = 1
    "CONTROL_ON": control_on,
    "GAS_BRAKE": gas_brake,
    "SET_TO_1": 0x01,
  }
  commands.append(packer.make_can_msg("ACC_CONTROL", 0, acc_control_values, idx))

  acc_control_on_values = {
    "SET_TO_3": 0x03,
    "CONTROL_ON": enabled,
    "SET_TO_FF": 0xff,
    "SET_TO_75": 0x75,
    "SET_TO_30": 0x30,
  }
  commands.append(packer.make_can_msg("ACC_CONTROL_ON", 0, acc_control_on_values, idx))

  return commands

def create_steering_control(packer, apply_steer, lkas_active, car_fingerprint, radar_off_can, idx):
  values = {
    "STEER_TORQUE": apply_steer if lkas_active else 0,
    "STEER_TORQUE_REQUEST": lkas_active,
  }
  # Set bus 2 for accord and new crv.
  bus = 2 if car_fingerprint in HONDA_BOSCH and radar_off_can else 2
  return packer.make_can_msg("STEERING_CONTROL", bus, values, idx)


def create_ui_commands(packer, pcm_speed, hud, car_fingerprint, radar_off_can, openpilot_longitudinal_control, idx):
  commands = []

  if car_fingerprint in HONDA_BOSCH:
    acc_hud_values = {
      'CRUISE_SPEED': hud.v_cruise,
      'ENABLE_MINI_CAR': hud.mini_car,
      #'SET_TO_1': 0x01,
      'HUD_LEAD': hud.car,
      'HUD_DISTANCE': 0x02,
      'ACC_ON': hud.car != 0,
      'SET_TO_X3': 0x03,
    }
  else:
    acc_hud_values = {
      'PCM_SPEED': pcm_speed * CV.MS_TO_KPH,
      'PCM_GAS': hud.pcm_accel,
      'CRUISE_SPEED': hud.v_cruise,
      'ENABLE_MINI_CAR': hud.mini_car,
      'HUD_LEAD': hud.car,
      'SET_ME_X03': 0x03,
      'SET_ME_X03_2': 0x03,
      'SET_ME_X01': 0x01,
    }

  if openpilot_longitudinal_control:
    commands.append(packer.make_can_msg("ACC_HUD", 0, acc_hud_values, idx))

  lkas_hud_values = {
    'SET_ME_X41': 0x41,
    'SET_ME_X48': 0x48,
    'STEERING_REQUIRED': hud.steer_required,
    'SOLID_LANES': hud.lanes,
    'BEEP': hud.beep,
  }
  # Bosch sends commands to bus 2.
  bus = 2 if car_fingerprint in HONDA_BOSCH and radar_off_can else 0
  commands.append(packer.make_can_msg('LKAS_HUD', bus, lkas_hud_values, idx))

  if car_fingerprint in (CAR.CIVIC, CAR.ODYSSEY):

    radar_hud_values = {
      'ACC_ALERTS': hud.acc_alert,
      'LEAD_SPEED': 0x1fe,  # What are these magic values
      'LEAD_STATE': 0x7,
      'LEAD_DISTANCE': 0x1e,
    }
  elif car_fingerprint in HONDA_BOSCH:
    radar_hud_values = {
      #'SET_TO_1' : 0x01,
    }

  if openpilot_longitudinal_control:
    commands.append(packer.make_can_msg('RADAR_HUD', 0, radar_hud_values, idx))

  return commands

from common.numpy_fast import clip
def create_radar_commands(v_ego, idx):
  commands = []
  v_ego_kph = clip(int(round(v_ego * CV.MS_TO_KPH)), 0, 255)
  speed = struct.pack('!B', v_ego_kph)

  msg_0x300 = ("\xf9" + speed + "\x8a\xd0" +
              ("\x20" if idx == 0 or idx == 3 else "\x00") +
              "\x00\x00")
  commands.append(make_can_msg(0x300, msg_0x300, idx, 1))

  # car_fingerprint == CAR.PILOT:
  msg_0x301 = "\x00\x00\x56\x02\x58\x00\x00"
  commands.append(make_can_msg(0x301, msg_0x301, idx, 1))

  return commands

def spam_buttons_command(packer, button_val, idx):
  values = {
    'CRUISE_BUTTONS': button_val,
    'CRUISE_SETTING': 0,
  }
  return packer.make_can_msg("SCM_BUTTONS", 0, values, idx)

def create_radar_VIN_msg(id,radarVIN,radarCAN,radarTriggerMessage,useRadar):
  msg_id = 0x560
  msg_len = 8
  msg = create_string_buffer(msg_len)
  if id == 0:
    struct.pack_into('BBBBBBBB', msg, 0, id,radarCAN,useRadar,((radarTriggerMessage >> 8) & 0xFF),(radarTriggerMessage & 0xFF),ord(radarVIN[0]),ord(radarVIN[1]),ord(radarVIN[2]))
  if id == 1:
    struct.pack_into('BBBBBBBB', msg, 0, id,ord(radarVIN[3]),ord(radarVIN[4]),ord(radarVIN[5]),ord(radarVIN[6]),ord(radarVIN[7]),ord(radarVIN[8]),ord(radarVIN[9]))
  if id == 2:
    struct.pack_into('BBBBBBBB', msg, 0, id,ord(radarVIN[10]),ord(radarVIN[11]),ord(radarVIN[12]),ord(radarVIN[13]),ord(radarVIN[14]),ord(radarVIN[15]),ord(radarVIN[16]))
  return [msg_id, 0, msg.raw, 0]
