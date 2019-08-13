def create_steering_control(packer, bus, apply_steer, idx, lkas_active):

  values = {
    "LKASteeringCmdActive": lkas_active,
    "LKASteeringCmd": apply_steer,
    "RollingCounter": idx,
    "LKASteeringCmdChecksum": 0x1000 - (lkas_active << 11) - (apply_steer & 0x7ff) - idx
  }

  return packer.make_can_msg("ASCMLKASteeringCmd", bus, values)

def create_steering_control_ct6(packer, canbus, apply_steer, v_ego, idx, enabled):

  values = {
    "LKASteeringCmdActive": 1 if enabled else 0,
    "LKASteeringCmd": apply_steer,
    "RollingCounter": idx,
    "SetMe1": 1,
    "LKASVehicleSpeed": abs(v_ego * 3.6),
    "LKASMode": 2 if enabled else 0,
    "LKASteeringCmdChecksum": 0  # assume zero and then manually compute it
  }

  dat = packer.make_can_msg("ASCMLKASteeringCmd", 0, values)[2]
  # the checksum logic is weird
  values['LKASteeringCmdChecksum'] = (0x2a +
                                      sum([ord(i) for i in dat][:4]) +
                                      values['LKASMode']) & 0x3ff
  # pack again with checksum
  dat = packer.make_can_msg("ASCMLKASteeringCmd", 0, values)[2]

  return [0x152, 0, dat, canbus.powertrain], \
         [0x154, 0, dat, canbus.powertrain], \
         [0x151, 0, dat, canbus.chassis], \
         [0x153, 0, dat, canbus.chassis]


def create_adas_keepalive(bus):
  dat = "\x00\x00\x00\x00\x00\x00\x00"
  return [[0x409, 0, dat, bus], [0x40a, 0, dat, bus]]

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
  values["GasRegenChecksum"] = (((0xff - ord(dat[1])) & 0xff) << 16) | \
                               (((0xff - ord(dat[2])) & 0xff) << 8) | \
                               ((0x100 - ord(dat[3]) - idx) & 0xff)

  return packer.make_can_msg("ASCMGasRegenCmd", bus, values)

def create_friction_brake_command(packer, bus, apply_brake, idx, near_stop, at_full_stop):

  if apply_brake == 0:
    mode = 0x1
  else:
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

def create_acc_dashboard_command(packer, bus, acc_engaged, target_speed_kph, lead_car_in_sight):
  # Not a bit shift, dash can round up based on low 4 bits.
  target_speed = int(target_speed_kph * 16) & 0xfff

  values = {
    "ACCAlwaysOne" : 1,
    "ACCResumeButton" : 0,
    "ACCSpeedSetpoint" : target_speed,
    "ACCGapLevel" : 3 * acc_engaged, # 3 "far", 0 "inactive"
    "ACCCmdActive" : acc_engaged,
    "ACCAlwaysOne2" : 1,
    "ACCLeadCar" : lead_car_in_sight
  }

  return packer.make_can_msg("ASCMActiveCruiseControlStatus", bus, values)

def create_adas_time_status(bus, tt, idx):
  dat = [(tt >> 20) & 0xff, (tt >> 12) & 0xff, (tt >> 4) & 0xff,
    ((tt & 0xf) << 4) + (idx << 2)]
  chksum = 0x1000 - dat[0] - dat[1] - dat[2] - dat[3]
  chksum = chksum & 0xfff
  dat += [0x40 + (chksum >> 8), chksum & 0xff, 0x12]
  return [0xa1, 0, "".join(map(chr, dat)), bus]

def create_adas_steering_status(bus, idx):
  dat = [idx << 6, 0xf0, 0x20, 0, 0, 0]
  chksum = 0x60 + sum(dat)
  dat += [chksum >> 8, chksum & 0xff]
  return [0x306, 0, "".join(map(chr, dat)), bus]

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
  return [0x308, 0, "".join(map(chr, dat)), bus]

def create_adas_headlights_status(bus):
  return [0x310, 0, "\x42\x04", bus]

def create_lka_icon_command(bus, active, critical, steer):
  if active and steer == 1:
    if critical:
      dat = "\x50\xc0\x14"
    else:
      dat = "\x50\x40\x18"
  elif active:
    if critical:
      dat = "\x40\xc0\x14"
    else:
      dat = "\x40\x40\x18"
  else:
    dat = "\x00\x00\x00"
  return [0x104c006c, 0, dat, bus]

# TODO: WIP
'''
def create_friction_brake_command_ct6(packer, bus, apply_brake, idx, near_stop, at_full_stop):

  # counters loops across [0, 29, 42, 55] but checksum only considers 0, 1, 2, 3
  cntrs = [0, 29, 42, 55]
  if apply_brake == 0:
    mode = 0x1
  else:
    mode = 0xa

    if at_full_stop:
      mode = 0xd
    elif near_stop:
      mode = 0xb

  brake = (0x1000 - apply_brake) & 0xfff
  checksum = (0x10000 - (mode << 12) - brake - idx) & 0xffff

  values = {
    "RollingCounter" : cntrs[idx],
    "FrictionBrakeMode" : mode,
    "FrictionBrakeChecksum": checksum,
    "FrictionBrakeCmd" : -apply_brake
  }

  dat = packer.make_can_msg("EBCMFrictionBrakeCmd", 0, values)[2]
  # msg is 0x315 but we are doing the panda forwarding
  return [0x314, 0, dat, 2]

def create_gas_regen_command_ct6(bus, throttle, idx, acc_engaged, at_full_stop):
  cntrs = [0, 7, 10, 13]
  eng_bit = 1 if acc_engaged else 0
  gas_high = (throttle >> 8) | 0x80
  gas_low = (throttle) & 0xff
  full_stop = 0x20 if at_full_stop else 0

  chk1 = (0x100 - gas_high - 1) & 0xff
  chk2 = (0x100 - gas_low - idx) & 0xff
  dat = [(idx << 6) | eng_bit, 0xc2 | full_stop, gas_high, gas_low,
         (1 - eng_bit) | (cntrs[idx] << 1), 0x5d - full_stop, chk1, chk2]
  return [0x2cb, 0, "".join(map(chr, dat)), bus]

'''
