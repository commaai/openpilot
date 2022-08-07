def create_steering_control(packer, bus, apply_steer, lkas_enabled):
  values = {
    "LKAS_TORQUE_ACTIVE": lkas_enabled,
    "LKAS_TORQUE": abs(apply_steer) if lkas_enabled else 1022,
    "LKAS_TORQUE_DIRECTION": 1 if apply_steer < 0 else 0,
    "LKAS_ACTIVE": lkas_enabled,
    "MAYBE_HUD_LANE_STATES": 3 if lkas_enabled else 1,
  }
  return packer.make_can_msg("LKAS", bus, values)
