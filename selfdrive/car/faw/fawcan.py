def create_steering_control(packer, bus, apply_steer, lkas_enabled):
  values = {
    "LKAS_TORQUE_ACTIVE": lkas_enabled,
    "LKAS_TORQUE": abs(apply_steer),
    "LKAS_TORQUE_DIRECTION": 1 if apply_steer < 0 else 0,
    "LKAS_ACTIVE": lkas_enabled,
    "MAYBE_LANE_STATUS": 0,  # TODO: populate this later
  }
  return packer.make_can_msg("LKAS", bus, values)
