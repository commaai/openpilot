from selfdrive.car import make_can_msg

def create_steering_control(packer, bus, apply_steer, idx, lkas_active):

  values = {
    "LKASteeringCmdActive": lkas_active,
    "LKASteeringCmd": apply_steer,
    "RollingCounter": idx,
    "LKASteeringCmdChecksum": 0x1000 - (lkas_active << 11) - (apply_steer & 0x7ff) - idx
  }

  return packer.make_can_msg("ASCMLKASteeringCmd", bus, values)

def create_adas_keepalive(bus):
  dat = b"\x00\x00\x00\x00\x00\x00\x00"
  return [make_can_msg(0x409, dat, bus), make_can_msg(0x40a, dat, bus)]

def create_gas_regen_command(packer, bus, throttle, idx, acc_engaged, at_full_stop):
  values = {
    "GasRegenCmdActive": acc_engaged,
    "RollingCounter": idx,
    "GasRegenCmdActiveInv": 1 - acc_engaged,
    "GasRegenCmd": throttle,
    "GasRegenFullStopActive": at_full_stop,
    "GasRegenAlwaysOne": 1,
    "GasRegenAlwaysOne2": 1,
    "GasRegenAlwaysOne3": 1,
  }

  dat = packer.make_can_msg("ASCMGasRegenCmd", bus, values)[2]
  values["GasRegenChecksum"] = (((0xff - dat[1]) & 0xff) << 16) | \
                               (((0xff - dat[2]) & 0xff) << 8) | \
                               ((0x100 - dat[3] - idx) & 0xff)

  return packer.make_can_msg("ASCMGasRegenCmd", bus, values)

def create_friction_brake_command(packer, bus, apply_brake, idx, near_stop, at_full_stop):
  mode = 0x1
  if apply_brake > 0:
    mode = 0xa
    if at_full_stop:
      mode = 0xd

    # TODO: this is to have GM bringing the car to complete stop,
    # but currently it conflicts with OP controls, so turned off.
    #elif near_stop:
    #  mode = 0xb

  brake = (0x1000 - apply_brake) & 0xfff
  checksum = (0x10000 - (mode << 12) - brake - idx) & 0xffff

  values = {
    "RollingCounter" : idx,
    "FrictionBrakeMode" : mode,
    "FrictionBrakeChecksum": checksum,
    "FrictionBrakeCmd" : -apply_brake
  }

  return packer.make_can_msg("EBCMFrictionBrakeCmd", bus, values)

def create_acc_dashboard_command(packer, bus, acc_engaged, target_speed_kph, lead_car_in_sight, fcw):
  # Not a bit shift, dash can round up based on low 4 bits.
  target_speed = int(target_speed_kph * 16) & 0xfff

  values = {
    "ACCAlwaysOne" : 1,
    "ACCResumeButton" : 0,
    "ACCSpeedSetpoint" : target_speed,
    "ACCGapLevel" : 3 * acc_engaged,  # 3 "far", 0 "inactive"
    "ACCCmdActive" : acc_engaged,
    "ACCAlwaysOne2" : 1,
    "ACCLeadCar" : lead_car_in_sight,
    "FCWAlert": 0x3 if fcw else 0
  }

  return packer.make_can_msg("ASCMActiveCruiseControlStatus", bus, values)

def create_adas_time_status(bus, tt, idx):
  dat = [(tt >> 20) & 0xff, (tt >> 12) & 0xff, (tt >> 4) & 0xff,
    ((tt & 0xf) << 4) + (idx << 2)]
  chksum = 0x1000 - dat[0] - dat[1] - dat[2] - dat[3]
  chksum = chksum & 0xfff
  dat += [0x40 + (chksum >> 8), chksum & 0xff, 0x12]
  return make_can_msg(0xa1, bytes(dat), bus)

def create_adas_steering_status(bus, idx):
  dat = [idx << 6, 0xf0, 0x20, 0, 0, 0]
  chksum = 0x60 + sum(dat)
  dat += [chksum >> 8, chksum & 0xff]
  return make_can_msg(0x306, bytes(dat), bus)

def create_adas_accelerometer_speed_status(bus, speed_ms, idx):
  spd = int(speed_ms * 16) & 0xfff
  accel = 0 & 0xfff
  # 0 if in park/neutral, 0x10 if in reverse, 0x08 for D/L
  #stick = 0x08
  near_range_cutoff = 0x27
  near_range_mode = 1 if spd <= near_range_cutoff else 0
  far_range_mode = 1 - near_range_mode
  dat = [0x08, spd >> 4, ((spd & 0xf) << 4) | (accel >> 8), accel & 0xff, 0]
  chksum = 0x62 + far_range_mode + (idx << 2) + dat[0] + dat[1] + dat[2] + dat[3] + dat[4]
  dat += [(idx << 5) + (far_range_mode << 4) + (near_range_mode << 3) + (chksum >> 8), chksum & 0xff]
  return make_can_msg(0x308, bytes(dat), bus)

def create_adas_headlights_status(packer, bus):
  values = {
    "Always42": 0x42,
    "Always4": 0x4,
  }
  return packer.make_can_msg("ASCMHeadlight", bus, values)

def create_lka_icon_command(bus, active, critical, steer):
  if active and steer == 1:
    if critical:
      dat = b"\x50\xc0\x14"
    else:
      dat = b"\x50\x40\x18"
  elif active:
    if critical:
      dat = b"\x40\xc0\x14"
    else:
      dat = b"\x40\x40\x18"
  else:
    dat = b"\x00\x00\x00"
  return make_can_msg(0x104c006c, dat, bus)
