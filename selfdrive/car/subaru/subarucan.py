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

def create_es_distance(packer, es_distance_msg, bus, pcm_cancel_cmd, long_active, brake_cmd, brake_value, cruise_throttle):

  values = copy.copy(es_distance_msg)
  if long_active:
    values["Cruise_Throttle"] = cruise_throttle
  if pcm_cancel_cmd:
    values["COUNTER"] = (values["COUNTER"] + 1) % 0x10
    values["Cruise_Cancel"] = 1
  if brake_cmd:
    values["Cruise_Throttle"] = 808 if brake_value >= 35 else 1818
    values["Cruise_Brake_Active"] = 1
  # Do not disable openpilot on Eyesight Soft Disable
  values["Cruise_Soft_Disable"] = 0

  return packer.make_can_msg("ES_Distance", bus, values)

def create_es_lkas_state(packer, es_lkas_msg, enabled, visual_alert, left_line, right_line, left_lane_depart, right_lane_depart):

  values = copy.copy(es_lkas_msg)

  # Filter the stock LKAS "Keep hands on wheel" alert
  if values["LKAS_Alert_Msg"] == 1:
    values["LKAS_Alert_Msg"] = 0

  # Filter the stock LKAS sending an audible alert when it turns off LKAS
  if values["LKAS_Alert"] == 27:
    values["LKAS_Alert"] = 0

  # Filter the stock LKAS sending an audible alert when "Keep hands on wheel" alert is active (2020+ models)
  if values["LKAS_Alert"] == 28 and values["LKAS_Alert_Msg"] == 7:
    values["LKAS_Alert"] = 0

  # Filter the stock LKAS sending an audible alert when "Keep hands on wheel OFF" alert is active (2020+ models)
  if values["LKAS_Alert"] == 30:
    values["LKAS_Alert"] = 0

  # Filter the stock LKAS sending "Keep hands on wheel OFF" alert (2020+ models)
  if values["LKAS_Alert_Msg"] == 7:
    values["LKAS_Alert_Msg"] = 0

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

def create_es_dashstatus(packer, es_dashstatus_msg, enabled, long_active, lead_visible):

  values = copy.copy(es_dashstatus_msg)
  if enabled and long_active:
    values["Cruise_State"] = 0
    values["Cruise_Activated"] = 1
    values["Cruise_Disengaged"] = 0
    values["Car_Follow"] = int(lead_visible)

  # Filter stock LKAS disabled and Keep hands on steering wheel OFF alerts
  if values["LKAS_State_Msg"] in [2, 3]:
    values["LKAS_State_Msg"] = 0

  return packer.make_can_msg("ES_DashStatus", 0, values)

def create_es_brake(packer, es_brake_msg, enabled, brake_cmd, brake_value):

  values = copy.copy(es_brake_msg)
  if enabled:
    values["Cruise_Activated"] = 1
  if brake_cmd:
    values["Brake_Pressure"] = brake_value
    values["Cruise_Brake_Active"] = 1
    values["Cruise_Brake_Lights"] = 1 if brake_value >= 70 else 0

  return packer.make_can_msg("ES_Brake", 0, values)

def create_es_status(packer, es_status_msg, long_active, cruise_rpm):

  values = copy.copy(es_status_msg)
  if long_active:
    values["Cruise_Activated"] = 1
    values["Cruise_RPM"] = cruise_rpm

  return packer.make_can_msg("ES_Status", 0, values)

# disable cruise_activated feedback to eyesight to keep ready state
def create_cruise_control(packer, cruise_control_msg):

  values = copy.copy(cruise_control_msg)
  values["Cruise_Activated"] = 0

  return packer.make_can_msg("CruiseControl", 2, values)

# disable es_brake feedback to eyesight, exempt AEB
def create_brake_status(packer, brake_status_msg, aeb):

  values = copy.copy(brake_status_msg)
  if not aeb:
    values["ES_Brake"] = 0

  return packer.make_can_msg("Brake_Status", 2, values)

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
