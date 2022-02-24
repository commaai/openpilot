from common.numpy_fast import clip
from selfdrive.car.ford.values import MAX_ANGLE


def create_steer_command(packer, angle_cmd, enabled, lkas_state, angle_steers, curvature, lkas_action):
  """Creates a CAN message for the Ford Steer Command."""

  #if enabled and lkas available:
  if enabled and lkas_state in (2, 3):  # and (frame % 500) >= 3:
    action = lkas_action
  else:
    action = 0xf
    angle_cmd = angle_steers/MAX_ANGLE

  angle_cmd = clip(angle_cmd * MAX_ANGLE, - MAX_ANGLE, MAX_ANGLE)

  values = {
    "Lkas_Action": action,
    "Lkas_Alert": 0xf,             # no alerts
    "Lane_Curvature": clip(curvature, -0.01, 0.01),   # is it just for debug?
    #"Lane_Curvature": 0,   # is it just for debug?
    "Steer_Angle_Req": angle_cmd
  }
  return packer.make_can_msg("Lane_Keep_Assist_Control", 0, values)


def create_lkas_ui(packer, main_on, enabled, steer_alert):
  """Creates a CAN message for the Ford Steer Ui."""

  if not main_on:
    lines = 0xf
  elif enabled:
    lines = 0x3
  else:
    lines = 0x6

  values = {
    "Set_Me_X80": 0x80,
    "Set_Me_X45": 0x45,
    "Set_Me_X30": 0x30,
    "Lines_Hud": lines,
    "Hands_Warning_W_Chime": steer_alert,
  }
  return packer.make_can_msg("Lane_Keep_Assist_Ui", 0, values)

def spam_cancel_button(packer):
  values = {
    "Cancel": 1
  }
  return packer.make_can_msg("Steering_Buttons", 0, values)
