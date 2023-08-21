from cereal import car
from openpilot.selfdrive.car.subaru.values import CanBus

VisualAlert = car.CarControl.HUDControl.VisualAlert


def create_steering_control(packer, apply_steer, steer_req):
  values = {
    "LKAS_Output": apply_steer,
    "LKAS_Request": steer_req,
    "SET_1": 1
  }
  return packer.make_can_msg("ES_LKAS", 0, values)


def create_steering_status(packer):
  return packer.make_can_msg("ES_LKAS_State", 0, {})

def create_es_distance(packer, es_distance_msg, bus, pcm_cancel_cmd, long_enabled = False, brake_cmd = False, cruise_throttle = 0):
  values = {s: es_distance_msg[s] for s in [
    "CHECKSUM",
    "COUNTER",
    "Signal1",
    "Cruise_Fault",
    "Cruise_Throttle",
    "Signal2",
    "Car_Follow",
    "Signal3",
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

  if long_enabled:
    values["Cruise_Throttle"] = cruise_throttle

    # Do not disable openpilot on Eyesight Soft Disable, if openpilot is controlling long
    values["Cruise_Soft_Disable"] = 0

    if brake_cmd:
      values["Cruise_Brake_Active"] = 1

  if pcm_cancel_cmd:
    values["Cruise_Cancel"] = 1
    values["Cruise_Throttle"] = 1818 # inactive throttle

  return packer.make_can_msg("ES_Distance", bus, values)


def create_es_lkas_state(packer, es_lkas_state_msg, enabled, visual_alert, left_line, right_line, left_lane_depart, right_lane_depart):
  values = {s: es_lkas_state_msg[s] for s in [
    "CHECKSUM",
    "COUNTER",
    "LKAS_Alert_Msg",
    "Signal1",
    "LKAS_ACTIVE",
    "LKAS_Dash_State",
    "Signal2",
    "Backward_Speed_Limit_Menu",
    "LKAS_Left_Line_Enable",
    "LKAS_Left_Line_Light_Blink",
    "LKAS_Right_Line_Enable",
    "LKAS_Right_Line_Light_Blink",
    "LKAS_Left_Line_Visible",
    "LKAS_Right_Line_Visible",
    "LKAS_Alert",
    "Signal3",
  ]}

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
      values["LKAS_Alert"] = 12  # Left lane departure dash alert
    elif right_lane_depart:
      values["LKAS_Alert"] = 11  # Right lane departure dash alert

  values["LKAS_ACTIVE"] = 1  # Show LKAS lane lines
  values["LKAS_Dash_State"] = 2 if enabled else 0  # Green enabled indicator

  values["LKAS_Left_Line_Visible"] = int(left_line)
  values["LKAS_Right_Line_Visible"] = int(right_line)

  return packer.make_can_msg("ES_LKAS_State", CanBus.main, values)

def create_es_dashstatus(packer, dashstatus_msg, enabled, long_enabled, long_active, lead_visible):
  values = {s: dashstatus_msg[s] for s in [
    "CHECKSUM",
    "COUNTER",
    "PCB_Off",
    "LDW_Off",
    "Signal1",
    "Cruise_State_Msg",
    "LKAS_State_Msg",
    "Signal2",
    "Cruise_Soft_Disable",
    "Cruise_Status_Msg",
    "Signal3",
    "Cruise_Distance",
    "Signal4",
    "Conventional_Cruise",
    "Signal5",
    "Cruise_Disengaged",
    "Cruise_Activated",
    "Signal6",
    "Cruise_Set_Speed",
    "Cruise_Fault",
    "Cruise_On",
    "Display_Own_Car",
    "Brake_Lights",
    "Car_Follow",
    "Signal7",
    "Far_Distance",
    "Cruise_State",
  ]}

  if enabled and long_active:
    values["Cruise_State"] = 0
    values["Cruise_Activated"] = 1
    values["Cruise_Disengaged"] = 0
    values["Car_Follow"] = int(lead_visible)

  if long_enabled:
    values["PCB_Off"] = 1 # AEB is not presevered, so show the PCB_Off on dash

  # Filter stock LKAS disabled and Keep hands on steering wheel OFF alerts
  if values["LKAS_State_Msg"] in (2, 3):
    values["LKAS_State_Msg"] = 0

  return packer.make_can_msg("ES_DashStatus", CanBus.main, values)

def create_es_brake(packer, es_brake_msg, enabled, brake_value):
  values = {s: es_brake_msg[s] for s in [
    "CHECKSUM",
    "COUNTER",
    "Signal1",
    "Brake_Pressure",
    "AEB_Status",
    "Cruise_Brake_Lights",
    "Cruise_Brake_Fault",
    "Cruise_Brake_Active",
    "Cruise_Activated",
    "Signal3",
  ]}

  if enabled:
    values["Cruise_Activated"] = 1

  values["Brake_Pressure"] = brake_value

  if brake_value > 0:
    values["Cruise_Brake_Active"] = 1
    values["Cruise_Brake_Lights"] = 1 if brake_value >= 70 else 0

  return packer.make_can_msg("ES_Brake", CanBus.main, values)

def create_es_status(packer, es_status_msg, long_enabled, long_active, cruise_rpm):
  values = {s: es_status_msg[s] for s in [
    "CHECKSUM",
    "COUNTER",
    "Signal1",
    "Cruise_Fault",
    "Cruise_RPM",
    "Signal2",
    "Cruise_Activated",
    "Brake_Lights",
    "Cruise_Hold",
    "Signal3",
  ]}

  if long_enabled:
    values["Cruise_RPM"] = cruise_rpm

    if long_active:
      values["Cruise_Activated"] = 1

  return packer.make_can_msg("ES_Status", CanBus.main, values)


def create_es_infotainment(packer, es_infotainment_msg, visual_alert):
  # Filter stock LKAS disabled and Keep hands on steering wheel OFF alerts
  values = {s: es_infotainment_msg[s] for s in [
    "CHECKSUM",
    "COUNTER",
    "LKAS_State_Infotainment",
    "LKAS_Blue_Lines",
    "Signal1",
    "Signal2",
  ]}
  if values["LKAS_State_Infotainment"] in (3, 4):
    values["LKAS_State_Infotainment"] = 0

  # Show Keep hands on wheel alert for openpilot steerRequired alert
  if visual_alert == VisualAlert.steerRequired:
    values["LKAS_State_Infotainment"] = 3

  # Show Obstacle Detected for fcw
  if visual_alert == VisualAlert.fcw:
    values["LKAS_State_Infotainment"] = 2

  return packer.make_can_msg("ES_Infotainment", CanBus.main, values)


# *** Subaru Pre-global ***

def subaru_preglobal_checksum(packer, values, addr):
  dat = packer.make_can_msg(addr, 0, values)[2]
  return (sum(dat[:7])) % 256


def create_preglobal_steering_control(packer, apply_steer, steer_req):
  values = {
    "LKAS_Command": apply_steer,
    "LKAS_Active": steer_req,
  }
  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_LKAS")

  return packer.make_can_msg("ES_LKAS", CanBus.main, values)


def create_preglobal_es_distance(packer, cruise_button, es_distance_msg):
  values = {s: es_distance_msg[s] for s in [
    "Cruise_Throttle",
    "Signal1",
    "Car_Follow",
    "Signal2",
    "Brake_On",
    "Distance_Swap",
    "Standstill",
    "Signal3",
    "Close_Distance",
    "Signal4",
    "Standstill_2",
    "Cruise_Fault",
    "Signal5",
    "COUNTER",
    "Signal6",
    "Cruise_Button",
    "Signal7",
  ]}

  values["Cruise_Button"] = cruise_button
  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_Distance")

  return packer.make_can_msg("ES_Distance", CanBus.main, values)
