from selfdrive.car import make_can_msg

def create_lkas_command(packer, apply_steer, counter, steer_command_bit):
  values = {
    "LKAS_COMMAND": apply_steer,
    "LKAS_CONTROL_BIT": steer_command_bit,
    "COUNTER": counter
  }

  return packer.make_can_msg("DASM_LKAS_COMMAND", 0, values)

# TODO: might need to do counter sync, or maybe just filter/forward since we're in position to do so
def create_wheel_buttons(packer, frame, cancel=False):
  values = {
    "CANCEL": cancel,
    "COUNTER": frame % 16
  }
  return packer.make_can_msg("ACC_BUTTONS", 2, values)


# def create_lkas_hud(packer, enabled, leftLaneVisible, rightLaneVisible, autoHighBeamBit):
# the HUD message contains the auto high beam bit, and needs to be written in before enabling this again.
# FCA came up with this scheme, not me
#  if enabled:
#    if leftLaneVisible:
#      if rightLaneVisible:
#        lane_visibility_signal = 0x3  # Both sides white
#      else:
#        lane_visibility_signal = 0x9  # Left only white (GUESS, trying yellows for fun)
#    elif rightLaneVisible:
#      lane_visibility_signal = 0xA    # Right only white (GUESS, trying yellows for fun)
#    else:
#      lane_visibility_signal = 0x4    # Neither lane border shown
#  else:
#    lane_visibility_signal = 0x4      # Neither lane border shown
#  if CS.out.autoHighBeamBit == 1:
#    autoHighBeamBit = 1
#  else: autoHighBeamBit = 0

#  values = {
#    "LKAS_HUD": lane_visibility_signal,
#    "AUTO_HIGH_BEAM_BIT": autoHighBeamBit,
#  }

#  return packer.make_can_msg("FORWARD_CAMERA_HUD", 0, values)
