from selfdrive.car.honda.values import HONDA_BOSCH, CAR, CarControllerParams
from selfdrive.config import Conversions as CV

# CAN bus layout with relay
# 0 = ACC-CAN - radar side
# 1 = F-CAN B - powertrain
# 2 = ACC-CAN - camera side
# 3 = F-CAN A - OBDII port

def get_pt_bus(car_fingerprint):
  return 1 if car_fingerprint in HONDA_BOSCH else 0


def get_lkas_cmd_bus(car_fingerprint, radar_disabled=False):
  if radar_disabled:
    # when radar is disabled, steering commands are sent directly to powertrain bus
    return get_pt_bus(car_fingerprint)
  # normally steering commands are sent to radar, which forwards them to powertrain bus
  return 0

def create_brake_command(packer, apply_brake, pump_on, pcm_override, pcm_cancel_cmd, fcw, idx, car_fingerprint, stock_brake):
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
    "SET_ME_1": 1,
    "BRAKE_LIGHTS": brakelights,
    "CHIME": stock_brake["CHIME"] if fcw else 0,  # send the chime for stock fcw
    "FCW": fcw << 1,  # TODO: Why are there two bits for fcw?
    "AEB_REQ_1": 0,
    "AEB_REQ_2": 0,
    "AEB_STATUS": 0,
  }
  bus = get_pt_bus(car_fingerprint)
  return packer.make_can_msg("BRAKE_COMMAND", bus, values, idx)


def create_acc_commands(packer, enabled, active, accel, gas, idx, stopping, starting, car_fingerprint):
  commands = []
  bus = get_pt_bus(car_fingerprint)
  min_gas_accel = CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]

  control_on = 5 if enabled else 0
  gas_command = gas if active and accel > min_gas_accel else -30000
  accel_command = accel if active else 0
  braking = 1 if active and accel < min_gas_accel else 0
  standstill = 1 if active and stopping else 0
  standstill_release = 1 if active and starting else 0

  acc_control_values = {
    # setting CONTROL_ON causes car to set POWERTRAIN_DATA->ACC_STATUS = 1
    "CONTROL_ON": control_on,
    "GAS_COMMAND": gas_command,  # used for gas
    "ACCEL_COMMAND": accel_command,  # used for brakes
    "BRAKE_LIGHTS": braking,
    "BRAKE_REQUEST": braking,
    "STANDSTILL": standstill,
    "STANDSTILL_RELEASE": standstill_release,
  }
  commands.append(packer.make_can_msg("ACC_CONTROL", bus, acc_control_values, idx))

  acc_control_on_values = {
    "SET_TO_3": 0x03,
    "CONTROL_ON": enabled,
    "SET_TO_FF": 0xff,
    "SET_TO_75": 0x75,
    "SET_TO_30": 0x30,
  }
  commands.append(packer.make_can_msg("ACC_CONTROL_ON", bus, acc_control_on_values, idx))

  return commands

def create_steering_control(packer, apply_steer, lkas_active, car_fingerprint, idx, radar_disabled):
  values = {
    "STEER_TORQUE": apply_steer if lkas_active else 0,
    "STEER_TORQUE_REQUEST": lkas_active,
  }
  bus = get_lkas_cmd_bus(car_fingerprint, radar_disabled)
  return packer.make_can_msg("STEERING_CONTROL", bus, values, idx)


def create_bosch_supplemental_1(packer, car_fingerprint, idx):
  # non-active params
  values = {
    "SET_ME_X04": 0x04,
    "SET_ME_X80": 0x80,
    "SET_ME_X10": 0x10,
  }
  bus = get_lkas_cmd_bus(car_fingerprint)
  return packer.make_can_msg("BOSCH_SUPPLEMENTAL_1", bus, values, idx)


def create_ui_commands(packer, pcm_speed, hud, car_fingerprint, is_metric, idx, openpilot_longitudinal_control, stock_hud):
  commands = []
  bus_pt = get_pt_bus(car_fingerprint)
  radar_disabled = car_fingerprint in HONDA_BOSCH and openpilot_longitudinal_control
  bus_lkas = get_lkas_cmd_bus(car_fingerprint, radar_disabled)

  if openpilot_longitudinal_control:
    if car_fingerprint in HONDA_BOSCH:
      acc_hud_values = {
        'CRUISE_SPEED': hud.v_cruise,
        'ENABLE_MINI_CAR': 1,
        'SET_TO_1': 1,
        'HUD_LEAD': hud.car,
        'HUD_DISTANCE': 3,
        'ACC_ON': hud.car != 0,
        'SET_TO_X1': 1,
        'IMPERIAL_UNIT': int(not is_metric),
      }
    else:
      acc_hud_values = {
        'PCM_SPEED': pcm_speed * CV.MS_TO_KPH,
        'PCM_GAS': hud.pcm_accel,
        'CRUISE_SPEED': hud.v_cruise,
        'ENABLE_MINI_CAR': 1,
        'HUD_LEAD': hud.car,
        'HUD_DISTANCE': 3,    # max distance setting on display
        'IMPERIAL_UNIT': int(not is_metric),
        'SET_ME_X01_2': 1,
        'SET_ME_X01': 1,
        "FCM_OFF": stock_hud["FCM_OFF"],
        "FCM_OFF_2": stock_hud["FCM_OFF_2"],
        "FCM_PROBLEM": stock_hud["FCM_PROBLEM"],
        "ICONS": stock_hud["ICONS"],
      }
    commands.append(packer.make_can_msg("ACC_HUD", bus_pt, acc_hud_values, idx))

  lkas_hud_values = {
    'SET_ME_X41': 0x41,
    'SET_ME_X48': 0x48,
    'STEERING_REQUIRED': hud.steer_required,
    'SOLID_LANES': hud.lanes,
    'BEEP': 0,
  }
  commands.append(packer.make_can_msg('LKAS_HUD', bus_lkas, lkas_hud_values, idx))

  if radar_disabled and car_fingerprint in HONDA_BOSCH:
    radar_hud_values = {
      'CMBS_OFF': 0x01,
      'SET_TO_1': 0x01,
    }
    commands.append(packer.make_can_msg('RADAR_HUD', bus_pt, radar_hud_values, idx))

    if car_fingerprint == CAR.CIVIC_BOSCH:
      commands.append(packer.make_can_msg("LEGACY_BRAKE_COMMAND", bus_pt, {}, idx))

  return commands


def spam_buttons_command(packer, button_val, idx, car_fingerprint):
  values = {
    'CRUISE_BUTTONS': button_val,
    'CRUISE_SETTING': 0,
  }
  bus = get_pt_bus(car_fingerprint)
  return packer.make_can_msg("SCM_BUTTONS", bus, values, idx)
