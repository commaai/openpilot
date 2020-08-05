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
  values["Button"] = fake_button

  values["Checksum"] = subaru_preglobal_checksum(packer, values, "ES_CruiseThrottle")

  return packer.make_can_msg("ES_CruiseThrottle", 0, values)
