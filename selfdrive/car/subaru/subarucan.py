import copy
from cereal import car

VisualAlert = car.CarControl.HUDControl.VisualAlert

def create_steering_control(packer, apply_steer):
  values = {
    "LKAS_Output": apply_steer,
    "LKAS_Request": 1 if apply_steer != 0 else 0,
    "SET_1": 1
  }
  return packer.make_can_msg("ES_LKAS", 0, values)

def create_steering_status(packer):
  return packer.make_can_msg("ES_LKAS_State", 0, {})

def create_es_distance(packer, es_distance_msg, pcm_cancel_cmd):

  values = copy.copy(es_distance_msg)
  if pcm_cancel_cmd:
    values["Cruise_Cancel"] = 1

  return packer.make_can_msg("ES_Distance", 0, values)

def create_es_lkas(packer, es_lkas_msg, enabled, visual_alert, left_line, right_line, left_lane_depart, right_lane_depart):

  values = copy.copy(es_lkas_msg)

  # Filter the stock LKAS "Keep hands on wheel" alert
  if values["LKAS_Alert_Msg"] == 1:
    values["LKAS_Alert_Msg"] = 0

  # Filter the stock LKAS sending an audible alert when it turns off LKAS
  if values["LKAS_Alert"] == 27:
    values["LKAS_Alert"] = 0

  # Show Keep hands on wheel alert for openpilot steerRequired alert
  if visual_alert == VisualAlert.steerRequired:
    values["LKAS_Alert_Msg"] = 1

  # Ensure we don't overwrite potentially more important alerts from stock (e.g. FCW)
  if visual_alert == VisualAlert.ldw and values["LKAS_Alert"] == 0:
    if left_lane_depart:
      values["LKAS_Alert"] = 12 # Left lane departure dash alert
    elif right_lane_depart:
      values["LKAS_Alert"] = 11 # Right lane departure dash alert

  if enabled:
    values["LKAS_ACTIVE"] = 1 # Show LKAS lane lines
    values["LKAS_Dash_State"] = 2 # Green enabled indicator
  else:
    values["LKAS_Dash_State"] = 0 # LKAS Not enabled

  values["LKAS_Left_Line_Visible"] = int(left_line)
  values["LKAS_Right_Line_Visible"] = int(right_line)

  return packer.make_can_msg("ES_LKAS_State", 0, values)

# *** Subaru Pre-global ***

def subaru_preglobal_checksum(packer, values, addr):
  dat = packer.make_can_msg(addr, 0, values)[2]
  return (sum(dat[:7])) % 256

def create_preglobal_steering_control(packer, apply_steer):
  values = {
    "LKAS_Command": apply_steer,
    "LKAS_Active": 1 if apply_steer != 0 else 0
  }
  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_LKAS")

  return packer.make_can_msg("ES_LKAS", 0, values)

def create_preglobal_es_distance(packer, cruise_button, es_distance_msg):

  values = copy.copy(es_distance_msg)
  values["Cruise_Button"] = cruise_button

  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_Distance")

  return packer.make_can_msg("ES_Distance", 0, values)
