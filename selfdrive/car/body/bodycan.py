def create_control(packer, torque_l, torque_r):
  values = {
    "TORQUE_L": torque_l,
    "TORQUE_R": torque_r,
  }

  return packer.make_can_msg("TORQUE_CMD", 0, values)
