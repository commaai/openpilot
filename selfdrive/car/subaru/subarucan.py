from cereal import car
from common.conversions import Conversions as CV
from selfdrive.car.subaru.values import Buttons

VisualAlert = car.CarControl.HUDControl.VisualAlert

def create_steering_control(packer, apply_steer, bus):
  values = {
    "LKAS_Output": apply_steer,
    "LKAS_Request": 1 if apply_steer != 0 else 0,
    "SET_1": 1
  }
  return packer.make_can_msg("ES_LKAS", bus, values)


def create_steering_status(packer):
  return packer.make_can_msg("ES_LKAS_State", 0, {})

def create_es_distance(packer, frame, es_distance_msg, pcm_cancel_cmd, long_active, brake_cmd, brake_value, cruise_throttle, cruise_button, low_speed, bus):
  if es_distance_msg is None:
    values = {
      "Signal1": 1,
      "Cruise_Fault": 0,
      "Cruise_Throttle": 0,
      "Signal2": 2,
      "Car_Follow": 0,
      "Low_Speed": 0,
      "Cruise_Soft_Disable": 0,
      "Signal7": 0,
      "Cruise_Brake_Active": 0,
      "Distance_Swap": 0,
      "Cruise_EPB": 0,
      "Signal4": 1,
      "Close_Distance": 5,
      "Signal5": 0,
      "Cruise_Cancel": 0,
      "Cruise_Set": 0,
      "Cruise_Resume": 0,
      "Signal6": 0
    }

  else:
    values = {s: es_distance_msg[s] for s in [
      "CHECKSUM",
      "COUNTER",
      "Signal1",
      "Cruise_Fault",
      "Cruise_Throttle",
      "Signal2",
      "Car_Follow",
      "Low_Speed",
      "Cruise_Soft_Disable",
      "Signal7",
      "Cruise_Brake_Active",
      "Distance_Swap",
      "Cruise_EPB",
      "Signal4",
      "Close_Distance",
      "Signal5",
      "Cruise_Cancel",
      "Cruise_Set",
      "Cruise_Resume",
      "Signal6",
    ]}
    values["COUNTER"] = (values["COUNTER"] + 1) % 0x10

  if pcm_cancel_cmd:
    values["Cruise_Cancel"] = 1

  if long_active:
    values["Cruise_Throttle"] = cruise_throttle
    values["Cruise_Fault"] = 0
    values["Car_Follow"] = 0

  if brake_cmd:
    values["Cruise_Throttle"] = 808 if brake_value >= 35 else 1818
    values["Cruise_Brake_Active"] = 1
  
  values["Low_Speed"] = low_speed
  
  # Do not disable openpilot on Eyesight Soft Disable
  values["Cruise_Soft_Disable"] = 0

  return packer.make_can_msg("ES_Distance", bus, values)


def create_es_lkas_state(packer, es_lkas_state_msg, enabled, visual_alert, left_line, right_line, left_lane_depart, right_lane_depart):
  values = copy.copy(es_lkas_state_msg)

  # Filter the stock LKAS "Keep hands on wheel" alert
  #if values["LKAS_Alert_Msg"] == 1:
  values["LKAS_Alert_Msg"] = 0

  # Filter the stock LKAS sending an audible alert when it turns off LKAS
  #if values["LKAS_Alert"] == 27:
  values["LKAS_Alert"] = 0

  # Show Keep hands on wheel alert for openpilot steerRequired alert
  if hud_control.visualAlert == VisualAlert.steerRequired:
    values["LKAS_Alert_Msg"] = 1

  # Ensure we don't overwrite potentially more important alerts from stock (e.g. FCW)
  if hud_control.visualAlert == VisualAlert.ldw and values["LKAS_Alert"] == 0:
    if hud_control.leftLaneDepart:
      values["LKAS_Alert"] = 12  # Left lane departure dash alert
    elif hud_control.rightLaneDepart:
      values["LKAS_Alert"] = 11  # Right lane departure dash alert

  values["LKAS_ACTIVE"] = 1  # Show LKAS lane lines

  values["LKAS_Dash_State"] = 2 if enabled else (1 if cruise_on else 0) # Green enabled indicator

  values["LKAS_Left_Line_Visible"] = int(hud_control.leftLaneVisible)
  values["LKAS_Right_Line_Visible"] = int(hud_control.rightLaneVisible)

  return packer.make_can_msg("ES_LKAS_State", 0, values)


def create_es_dashstatus(packer, dashstatus_msg):
  values = copy.copy(dashstatus_msg)

  values["LKAS_State_Msg"] = 0

  values["Cruise_State_Msg"] = 0
  values["Display_Own_Car"] = 1
  values["Cruise_Activated"] = enabled
  values["Cruise_Disengaged"] = not enabled
  values["Conventional_Cruise"] = False
  values["Cruise_Distance"] = 10
  values["Cruise_Status_Msg"] = 0
  values["Cruise_Soft_Disable"] = 0
  values["Cruise_On"] = cruise_on
  values["Cruise_Set_Speed"] = hud_control.setSpeed * CV.MS_TO_MPH if hud_control.speedVisible else 0 # TODO: handle kph on dash
  
  return packer.make_can_msg("ES_DashStatus", 0, values)


def create_infotainmentstatus(packer, infotainmentstatus_msg, visual_alert):
  # Filter stock LKAS disabled and Keep hands on steering wheel OFF alerts
  if infotainmentstatus_msg["LKAS_State_Infotainment"] in (3, 4):
    infotainmentstatus_msg["LKAS_State_Infotainment"] = 0

  # Show Keep hands on wheel alert for openpilot steerRequired alert
  if visual_alert == VisualAlert.steerRequired:
    infotainmentstatus_msg["LKAS_State_Infotainment"] = 3

  # Show Obstacle Detected for fcw
  if visual_alert == VisualAlert.fcw:
    infotainmentstatus_msg["LKAS_State_Infotainment"] = 2

  return packer.make_can_msg("INFOTAINMENT_STATUS", 0, infotainmentstatus_msg)


# *** Subaru Pre-global ***

def subaru_preglobal_checksum(packer, values, addr):
  dat = packer.make_can_msg(addr, 0, values)[2]
  return (sum(dat[:7])) % 256


def create_preglobal_steering_control(packer, apply_steer, bus):
  values = {
    "LKAS_Command": apply_steer,
    "LKAS_Active": 1 if apply_steer != 0 else 0
  }
  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_LKAS")

  return packer.make_can_msg("ES_LKAS", bus, values)


def create_preglobal_es_distance(packer, cruise_button, es_distance_msg):
  values = copy.copy(es_distance_msg)
  values["Cruise_Button"] = cruise_button
  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_Distance")

  return packer.make_can_msg("ES_Distance", bus, values)