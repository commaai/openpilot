def create_control(packer, torque_l, torque_r):
  values = {
    "TORQUE_L": torque_l,
    "TORQUE_R": torque_r,
  }

  return packer.make_can_msg("TORQUE_CMD", 0, values)

def create_rpm_limit(packer, limit_l, limit_r, idx):
  values = {
    "MAX_RPM_L": limit_l,
    "MAX_RPM_R": limit_r,
  }

  return packer.make_can_msg("MAX_MOTOR_RPM_CMD", 0, values)

def create_knee_control(packer, torque_l, torque_r):
  values = {
    "TORQUE_L": torque_l,
    "TORQUE_R": torque_r,
  }

  return packer.make_can_msg("KNEE_TORQUE_CMD", 0, values)

def create_knee_rpm_limit(packer, limit_l, limit_r):
  values = {
    "MAX_RPM_L": limit_l,
    "MAX_RPM_R": limit_r,
  }

  return packer.make_can_msg("KNEE_MAX_MOTOR_RPM_CMD", 0, values)
