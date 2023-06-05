import copy
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

def create_es_distance(packer, es_distance_msg, pcm_cancel_cmd, long_active, brake_cmd, brake_value, cruise_throttle, cruise_button, low_speed, bus):
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

  if cruise_button == Buttons.RES_INC:
    values["Cruise_Resume"] = 1
  if cruise_button == Buttons.SET_DEC:
    values["Cruise_Set"] = 1
  if cruise_button == Buttons.ACC_TOGGLE:
    values["Cruise_Cancel"] = 1

  if pcm_cancel_cmd:
    values["Cruise_Cancel"] = pcm_cancel_cmd

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


def create_es_lkas_state(packer, es_lkas_state_msg, enabled, hud_control, cruise_on):
  if es_lkas_state_msg is None:
    values = {
      "LKAS_Alert_Msg": 0,
      "Signal1": 0,
      "LKAS_ACTIVE": 0,
      "LKAS_Dash_State": 0,
      "Signal2": 0,
      "Backward_Speed_Limit_Menu": 0,
      "LKAS_Left_Line_Enable": 0,
      "LKAS_Left_Line_Light_Blink": 0,
      "LKAS_Right_Line_Enable": 0,
      "LKAS_Right_Line_Light_Blink": 0,
      "LKAS_Left_Line_Visible": 0,
      "LKAS_Right_Line_Visible": 0,
      "LKAS_Alert": 0,
      "Signal3": 400
    }
  else:
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
  if hud_control.visualAlert == VisualAlert.steerRequired:
    values["LKAS_Alert_Msg"] = 1

  # Ensure we don't overwrite potentially more important alerts from stock (e.g. FCW)
  if hud_control.visualAlert == VisualAlert.ldw and values["LKAS_Alert"] == 0:
    if hud_control.leftLaneDepart:
      values["LKAS_Alert"] = 12  # Left lane departure dash alert
    elif hud_control.rightLaneDepart:
      values["LKAS_Alert"] = 11  # Right lane departure dash alert

  values["LKAS_ACTIVE"] = 1  # Show LKAS lane lines

  values["LKAS_Dash_State"] = 0 if not cruise_on else (1 if not enabled else 2)

  values["LKAS_Left_Line_Visible"] = int(hud_control.leftLaneVisible)
  values["LKAS_Right_Line_Visible"] = int(hud_control.rightLaneVisible)

  return packer.make_can_msg("ES_LKAS_State", 0, values)

def create_es_dashstatus(packer, dashstatus_msg, enabled, long_active, hud_control, cruise_on): 
  if dashstatus_msg is None:
    values = {
      "PCB_Off": 0,
      "LDW_Off": 0,
      "Signal1": 0,
      "Signal2": 0,
      "Signal3": 0,
      "Cruise_Distance": 10,
      "Signal4": 1,
      "Conventional_Cruise": 0,
      "Signal5": 0,
      "Cruise_Disengaged": 0,
      "Cruise_Activated": 0,
      "Signal6": 0,
      "Cruise_Set_Speed": 0,
      "Cruise_Fault": 0,
      "Cruise_On": 0,
      "Display_Own_Car": 1,
      "Brake_Lights": 0,
      "Car_Follow": 0,
      "Signal7": 0,
      "Far_Distance": 10,
      "Cruise_State": 0
    }
  else:
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

  if values["LKAS_State_Msg"] in (2, 3):
    values["LKAS_State_Msg"] = 0

  if enabled and long_active:
    values["Cruise_State"] = 2
    values["Car_Follow"] = int(hud_control.leadVisible)
    values["Cruise_Fault"] = 0

  values["Cruise_Activated"] = enabled
  values["Cruise_Disengaged"] = not enabled
  values["Conventional_Cruise"] = False
  values["Cruise_On"] = cruise_on
  values["Cruise_Set_Speed"] = hud_control.setSpeed * CV.MS_TO_MPH if hud_control.speedVisible else 0 # TODO: handle kph on dash
  
  return packer.make_can_msg("ES_DashStatus", 0, values)

def create_es_brake(packer, es_brake_msg, enabled, brake_cmd, brake_value, bus):
  if es_brake_msg is None:
    values = {}
  else:
    values = {s: es_brake_msg[s] for s in [
      "Brake_Pressure",
      "Cruise_Brake_Active",
      "Cruise_Brake_Lights"
    ]}

  values["Cruise_Activated"] = enabled

  if brake_cmd:
    values["Brake_Pressure"] = brake_value
    values["Cruise_Brake_Active"] = 1
    values["Cruise_Brake_Lights"] = 1 if brake_value >= 70 else 0

  return packer.make_can_msg("ES_Brake", bus, values)

def create_es_status(packer, es_status_msg, long_active, cruise_rpm, bus):
  if es_status_msg is None:
    values = {}
  else:
    # Filter stock LKAS disabled and Keep hands on steering wheel OFF alerts
    values = {s: es_status_msg[s] for s in [
      "CHECKSUM",
      "COUNTER",
      "ES_Brake"
    ]}
  
  if long_active:
    values["Cruise_Activated"] = 1
    values["Cruise_RPM"] = cruise_rpm
    values["Cruise_Fault"] = 0

  return packer.make_can_msg("ES_Status", bus, values)

# # disable cruise_activated feedback to eyesight to keep ready state
# def create_cruise_control(packer, cruise_control_msg):
#   if cruise_control_msg is None:
#     values = {
#     }
#   else:
#     # Filter stock LKAS disabled and Keep hands on steering wheel OFF alerts
#     values = {s: cruise_control_msg[s] for s in [
#       "CHECKSUM",
#       "COUNTER",
#       "ES_Brake"
#     ]}
#   values["Cruise_Activated"] = 0

#   return packer.make_can_msg("CruiseControl", 2, values)

# disable es_brake feedback to eyesight, exempt AEB
def create_brake_status(packer, es_brake_msg, aeb, bus):
  if es_brake_msg is None:
    values = {
      "Signal1": 0,
      "ES_Brake": 0,
      "Signal2": 0,
      "Brake": 0,
      "Signal3": 0
    }
  else:
    # Filter stock LKAS disabled and Keep hands on steering wheel OFF alerts
    values = {s: es_brake_msg[s] for s in [
      "CHECKSUM",
      "COUNTER",
      "ES_Brake"
    ]}
  
  if not aeb:
    values["ES_Brake"] = 0

  return packer.make_can_msg("Brake_Status", bus, values)


def create_infotainmentstatus(packer, infotainmentstatus_msg, visual_alert):
  if infotainmentstatus_msg is not None:
    values = {s: infotainmentstatus_msg[s] for s in [
      "CHECKSUM",
      "COUNTER",
      "LKAS_State_Infotainment",
      "LKAS_Blue_Lines",
      "Signal1",
      "Signal2",
    ]}
  else:
    values = {
      "LKAS_State_Infotainment": 0
    }

  # Filter stock LKAS disabled and Keep hands on steering wheel OFF alerts
  if values["LKAS_State_Infotainment"] in (3, 4):
    values["LKAS_State_Infotainment"] = 0

  # Show Keep hands on wheel alert for openpilot steerRequired alert
  if visual_alert == VisualAlert.steerRequired:
    values["LKAS_State_Infotainment"] = 3

  # Show Obstacle Detected for fcw
  if visual_alert == VisualAlert.fcw:
    values["LKAS_State_Infotainment"] = 2

  return packer.make_can_msg("INFOTAINMENT_STATUS", 0, values)


def create_unknown_1(packer, bus):
  values = {
    "Signal1": 0x000080FF7F00
  }
  return packer.make_can_msg("ES_UNKNOWN_1", bus, values)


def create_unknown_2(packer, bus):
  values = {
    "Signal1": 0xC00000000000
  }
  return packer.make_can_msg("ES_UNKNOWN_2", bus, values)


def create_unknown_3(packer, bus):
  values = {
    "Signal1": 0x010000000000
  }
  return packer.make_can_msg("ES_UNKNOWN_3", bus, values)

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


def create_preglobal_es_distance(packer, cruise_button, es_distance_msg, bus):
  values = copy.copy(es_distance_msg)
  values["Cruise_Button"] = cruise_button
  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_Distance")

  return packer.make_can_msg("ES_Distance", bus, values)