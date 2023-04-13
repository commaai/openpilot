
def create_steering_control(packer, car_fingerprint, apply_steer):

  values = {
    "LKAS_REQUEST": apply_steer,
  }

  return packer.make_can_msg("CAM_LKAS", 0, values)
