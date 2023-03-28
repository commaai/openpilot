from common.conversions import Conversions as CV
from selfdrive.car.honda.values import HondaFlags, HONDA_BOSCH, HONDA_BOSCH_RADARLESS, CAR, CarControllerParams

# CAN bus layout with relay
# 0 = ACC-CAN - radar side
# 1 = F-CAN B - powertrain
# 2 = ACC-CAN - camera side
# 3 = F-CAN A - OBDII port


def get_pt_bus(car_fingerprint):
  return 1 if car_fingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS) else 0


def get_lkas_cmd_bus(car_fingerprint, radar_disabled=False):
  no_radar = car_fingerprint in HONDA_BOSCH_RADARLESS
  if radar_disabled or no_radar:
    # when radar is disabled, steering commands are sent directly to powertrain bus
    return get_pt_bus(car_fingerprint)
  # normally steering commands are sent to radar, which forwards them to powertrain bus
  return 0


def get_cruise_speed_conversion(car_fingerprint: str, is_metric: bool) -> float:
  # on certain cars, CRUISE_SPEED changes to imperial with car's unit setting
  return CV.MPH_TO_MS if car_fingerprint in HONDA_BOSCH_RADARLESS and not is_metric else CV.KPH_TO_MS


def create_brake_command(packer, apply_brake, pump_on, pcm_override, pcm_cancel_cmd, fcw, car_fingerprint, stock_brake):
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
  return packer.make_can_msg("BRAKE_COMMAND", bus, values)


def create_acc_commands(packer, enabled, active, accel, gas, stopping, car_fingerprint):
  commands = []
  bus = get_pt_bus(car_fingerprint)
  min_gas_accel = CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]

  control_on = 5 if enabled else 0
  gas_command = gas if active and accel > min_gas_accel else -30000
  accel_command = accel if active else 0
  braking = 1 if active and accel < min_gas_accel else 0
  standstill = 1 if active and stopping else 0
  standstill_release = 1 if active and not stopping else 0

  # common ACC_CONTROL values
  acc_control_values = {
    'ACCEL_COMMAND': accel_command,
    'STANDSTILL': standstill,
  }

  if car_fingerprint in HONDA_BOSCH_RADARLESS:
    acc_control_values.update({
      "CONTROL_ON": enabled,
      "IDLESTOP_ALLOW": 0,  # disallows idle stop
    })
  else:
    acc_control_values.update({
      # setting CONTROL_ON causes car to set POWERTRAIN_DATA->ACC_STATUS = 1
      "CONTROL_ON": control_on,
      "GAS_COMMAND": gas_command,  # used for gas
      "BRAKE_LIGHTS": braking,
      "BRAKE_REQUEST": braking,
      "STANDSTILL_RELEASE": standstill_release,
    })
    acc_control_on_values = {
      "SET_TO_3": 0x03,
      "CONTROL_ON": enabled,
      "SET_TO_FF": 0xff,
      "SET_TO_75": 0x75,
      "SET_TO_30": 0x30,
    }
    commands.append(packer.make_can_msg("ACC_CONTROL_ON", bus, acc_control_on_values))

  commands.append(packer.make_can_msg("ACC_CONTROL", bus, acc_control_values))
  return commands


def create_steering_control(packer, apply_steer, lkas_active, car_fingerprint, radar_disabled):
  values = {
    "STEER_TORQUE": apply_steer if lkas_active else 0,
    "STEER_TORQUE_REQUEST": lkas_active,
  }
  bus = get_lkas_cmd_bus(car_fingerprint, radar_disabled)
  return packer.make_can_msg("STEERING_CONTROL", bus, values)


def create_bosch_supplemental_1(packer, car_fingerprint):
  # non-active params
  values = {
    "SET_ME_X04": 0x04,
    "SET_ME_X80": 0x80,
    "SET_ME_X10": 0x10,
  }
  bus = get_lkas_cmd_bus(car_fingerprint)
  return packer.make_can_msg("BOSCH_SUPPLEMENTAL_1", bus, values)


def create_ui_commands(packer, CP, enabled, pcm_speed, hud, is_metric, acc_hud, lkas_hud):
  commands = []
  bus_pt = get_pt_bus(CP.carFingerprint)
  radar_disabled = CP.carFingerprint in (HONDA_BOSCH - HONDA_BOSCH_RADARLESS) and CP.openpilotLongitudinalControl
  bus_lkas = get_lkas_cmd_bus(CP.carFingerprint, radar_disabled)

  if CP.openpilotLongitudinalControl:
    acc_hud_values = {
      'CRUISE_SPEED': hud.v_cruise,
      'ENABLE_MINI_CAR': 1 if enabled else 0,
      'HUD_DISTANCE': 0,  # max distance setting on display
      'IMPERIAL_UNIT': int(not is_metric),
      'HUD_LEAD': 2 if enabled and hud.lead_visible else 1 if enabled else 0,
      'SET_ME_X01_2': 1,
    }

    if CP.carFingerprint in HONDA_BOSCH:
      acc_hud_values['ACC_ON'] = int(enabled)
      acc_hud_values['FCM_OFF'] = 1
      acc_hud_values['FCM_OFF_2'] = 1
    else:
      acc_hud_values['PCM_SPEED'] = pcm_speed * CV.MS_TO_KPH
      acc_hud_values['PCM_GAS'] = hud.pcm_accel
      acc_hud_values['SET_ME_X01'] = 1
      acc_hud_values['FCM_OFF'] = acc_hud['FCM_OFF']
      acc_hud_values['FCM_OFF_2'] = acc_hud['FCM_OFF_2']
      acc_hud_values['FCM_PROBLEM'] = acc_hud['FCM_PROBLEM']
      acc_hud_values['ICONS'] = acc_hud['ICONS']
    commands.append(packer.make_can_msg("ACC_HUD", bus_pt, acc_hud_values))

  lkas_hud_values = {
    'SET_ME_X41': 0x41,
    'STEERING_REQUIRED': hud.steer_required,
    'SOLID_LANES': hud.lanes_visible,
    'BEEP': 0,
  }

  if CP.carFingerprint in HONDA_BOSCH_RADARLESS:
    lkas_hud_values['LANE_LINES'] = 3
    lkas_hud_values['DASHED_LANES'] = hud.lanes_visible
    # car likely needs to see LKAS_PROBLEM fall within a specific time frame, so forward from camera
    lkas_hud_values['LKAS_PROBLEM'] = lkas_hud['LKAS_PROBLEM']

  if not (CP.flags & HondaFlags.BOSCH_EXT_HUD):
    lkas_hud_values['SET_ME_X48'] = 0x48

  if CP.flags & HondaFlags.BOSCH_EXT_HUD and not CP.openpilotLongitudinalControl:
    commands.append(packer.make_can_msg('LKAS_HUD_A', bus_lkas, lkas_hud_values))
    commands.append(packer.make_can_msg('LKAS_HUD_B', bus_lkas, lkas_hud_values))
  else:
    commands.append(packer.make_can_msg('LKAS_HUD', bus_lkas, lkas_hud_values))

  if radar_disabled:
    radar_hud_values = {
      'CMBS_OFF': 0x01,
      'SET_TO_1': 0x01,
    }
    commands.append(packer.make_can_msg('RADAR_HUD', bus_pt, radar_hud_values))

    if CP.carFingerprint == CAR.CIVIC_BOSCH:
      commands.append(packer.make_can_msg("LEGACY_BRAKE_COMMAND", bus_pt, {}))

  return commands


def spam_buttons_command(packer, button_val, car_fingerprint):
  values = {
    'CRUISE_BUTTONS': button_val,
    'CRUISE_SETTING': 0,
  }
  # send buttons to camera on radarless cars
  bus = 2 if car_fingerprint in HONDA_BOSCH_RADARLESS else get_pt_bus(car_fingerprint)
  return packer.make_can_msg("SCM_BUTTONS", bus, values)
