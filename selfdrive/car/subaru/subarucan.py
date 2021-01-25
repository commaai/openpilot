import copy
from cereal import car

VisualAlert = car.CarControl.HUDControl.VisualAlert

def create_steering_control(packer, apply_steer, frame, steer_step):

  idx = (frame / steer_step) % 16

  values = {
    "Counter": idx,
    "LKAS_Output": apply_steer,
    "LKAS_Request": 1 if apply_steer != 0 else 0,
    "SET_1": 1
  }

  return packer.make_can_msg("ES_LKAS", 0, values)

def create_steering_status(packer, apply_steer, frame, steer_step):
  return packer.make_can_msg("ES_LKAS_State", 0, {})

def create_es_distance(packer, es_distance_msg, pcm_cancel_cmd):

  values = copy.copy(es_distance_msg)
  if pcm_cancel_cmd:
    values["Cruise_Cancel"] = 1

  return packer.make_can_msg("ES_Distance", 0, values)

#------------------TRICK NON-EPB------------------
def create_brake_pedal(packer, brake_pedal_msg, fake_wheel_speed):

  values = copy.copy(brake_pedal_msg)
  #Only modify Speed if the fake speed is not -1
  if fake_wheel_speed != -1:
    values["Speed"] = fake_wheel_speed

  return packer.make_can_msg("Brake_Pedal", 2, values)   
#------------------------------------------------- 

#---------------SUBARU STOP AND GO-----------------
def create_throttle(packer, throttle_msg, throttle_cmd):

  values = copy.copy(throttle_msg)
  #Only modify throttle_pedal if command is not -1, otherwise, just bypass/forward original message
  #For safety, limit our throttle command to 20 (out of 255)
  if throttle_cmd != -1 and throttle_cmd <= 10:
    values["Throttle_Pedal"] = throttle_cmd

  return packer.make_can_msg("Throttle", 2, values)
#---------------------------------------------------  

def create_es_lkas(packer, es_lkas_msg, visual_alert, left_line, right_line):

  values = copy.copy(es_lkas_msg)
  if visual_alert == VisualAlert.steerRequired:
    values["Keep_Hands_On_Wheel"] = 1

  values["LKAS_Left_Line_Visible"] = int(left_line)
  values["LKAS_Right_Line_Visible"] = int(right_line)

  return packer.make_can_msg("ES_LKAS_State", 0, values)

# *** Subaru Pre-global ***

def subaru_preglobal_checksum(packer, values, addr):
  dat = packer.make_can_msg(addr, 0, values)[2]
  return (sum(dat[:7])) % 256

def create_preglobal_throttle_control(packer, throttle_msg, throttle_cmd):

  values = copy.copy(throttle_msg)

  if throttle_cmd != -1 and throttle_cmd <= 10:
    values["Throttle_Pedal"] = throttle_cmd

  #values["Checksum"] = subaru_preglobal_checksum(packer, values, "Throttle")

  return packer.make_can_msg("Throttle", 2, values)  

def create_preglobal_steering_control(packer, apply_steer, frame, steer_step):

  idx = (frame / steer_step) % 8

  values = {
    "Counter": idx,
    "LKAS_Command": apply_steer,
    "LKAS_Active": 1 if apply_steer != 0 else 0
  }
  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_LKAS")

  return packer.make_can_msg("ES_LKAS", 0, values)

def create_es_throttle_control(packer, fake_button, es_accel_msg):

  values = copy.copy(es_accel_msg)
  values["Cruise_Button"] = fake_button

  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_CruiseThrottle")

  return packer.make_can_msg("ES_CruiseThrottle", 0, values)
