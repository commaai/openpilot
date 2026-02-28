# Opel Corsa F (PSA CMP Platform) - CAN message helpers
# Creates properly formatted CAN messages for the PSA platform


def create_psa_steering_control(packer, bus, apply_steer, idx, lkas_enabled):
  """Create LKAS steering control CAN message for PSA EPS."""
  values = {
    "LKAS_Active": lkas_enabled,
    "LKAS_Available": 1,
    "LKAS_Standby": not lkas_enabled,
    "SteerTorque": abs(apply_steer),
    "SteerDirection": 1 if apply_steer < 0 else 0,
    "SET_ME_1": 1,
    "COUNTER": idx,
  }
  return packer.make_can_msg("PSA_LKAS", bus, values)


def create_psa_hud_control(packer, bus, enabled, steering_pressed, hud_alert,
                            left_lane_visible, right_lane_visible,
                            left_lane_depart, right_lane_depart):
  """Create LDW/LKAS HUD control CAN message for PSA display."""
  # Lane color encoding:
  # 0 - off (LKAS disabled)
  # 1 - gray (LKAS enabled, no lane)
  # 2 - green (LKAS enabled, lane detected)
  # 3 - red (lane departure detected)
  values = {
    "LKAS_LED_Yellow": 1 if enabled and steering_pressed else 0,
    "LKAS_LED_Green": 1 if enabled and not steering_pressed else 0,
    "LaneStatus_Left": 3 if left_lane_depart else 1 + left_lane_visible,
    "LaneStatus_Right": 3 if right_lane_depart else 1 + right_lane_visible,
    "HUD_Alert": hud_alert,
  }
  return packer.make_can_msg("PSA_LDW_HUD", bus, values)


def create_psa_acc_buttons_control(packer, bus, buttonStatesToSend, CS, idx):
  """Create ACC button control CAN message for PSA cruise control."""
  values = {
    "MainSwitch": CS.accMainSwitch,
    "CancelButton": buttonStatesToSend["cancel"],
    "SetButton": buttonStatesToSend["setCruise"],
    "AccelButton": buttonStatesToSend["accelCruise"],
    "DecelButton": buttonStatesToSend["decelCruise"],
    "ResumeButton": buttonStatesToSend["resumeCruise"],
    "GapAdjustButton": 1 if buttonStatesToSend["gapAdjustCruise"] else 0,
    "COUNTER": idx,
  }
  return packer.make_can_msg("PSA_ACC_Buttons", bus, values)
