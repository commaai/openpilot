from opendbc.car.can_definitions import CanData
from opendbc.car.gm.values import CAR


def create_buttons(packer, bus, idx, button):
  values = {
    "ACCButtons": button,
    "RollingCounter": idx,
    "ACCAlwaysOne": 1,
    "DistanceButton": 0,
  }

  checksum = 240 + int(values["ACCAlwaysOne"] * 0xf)
  checksum += values["RollingCounter"] * (0x4ef if values["ACCAlwaysOne"] != 0 else 0x3f0)
  checksum -= int(values["ACCButtons"] - 1) << 4  # not correct if value is 0
  checksum -= 2 * values["DistanceButton"]

  values["SteeringButtonChecksum"] = checksum
  return packer.make_can_msg("ASCMSteeringButton", bus, values)


def create_pscm_status(packer, bus, pscm_status):
  values = {s: pscm_status[s] for s in [
    "HandsOffSWDetectionMode",
    "HandsOffSWlDetectionStatus",
    "LKATorqueDeliveredStatus",
    "LKADriverAppldTrq",
    "LKATorqueDelivered",
    "LKATotalTorqueDelivered",
    "RollingCounter",
    "PSCMStatusChecksum",
  ]}
  checksum_mod = int(1 - values["HandsOffSWlDetectionStatus"]) << 5
  values["HandsOffSWlDetectionStatus"] = 1
  values["PSCMStatusChecksum"] += checksum_mod
  return packer.make_can_msg("PSCMStatus", bus, values)


def create_steering_control(packer, bus, apply_torque, idx, lkas_active):
  values = {
    "LKASteeringCmdActive": lkas_active,
    "LKASteeringCmd": apply_torque,
    "RollingCounter": idx,
    "LKASteeringCmdChecksum": 0x1000 - (lkas_active << 11) - (apply_torque & 0x7ff) - idx
  }

  return packer.make_can_msg("ASCMLKASteeringCmd", bus, values)


def create_adas_keepalive(bus):
  dat = b"\x00\x00\x00\x00\x00\x00\x00"
  return [CanData(0x409, dat, bus), CanData(0x40a, dat, bus)]


def create_gas_regen_command(packer, bus, throttle, idx, enabled, at_full_stop):
  values = {
    "GasRegenCmdActive": enabled,
    "RollingCounter": idx,
    "GasRegenCmd": throttle,
    "GasRegenFullStopActive": at_full_stop,
    "GasRegenAccType": 1,
  }

  dat = packer.make_can_msg("ASCMGasRegenCmd", bus, values)[1]
  values["GasRegenChecksum"] = ((1 - enabled) << 24) | \
                               (((0xff - dat[1]) & 0xff) << 16) | \
                               (((0xff - dat[2]) & 0xff) << 8) | \
                               ((0x100 - dat[3] - idx) & 0xff)

  return packer.make_can_msg("ASCMGasRegenCmd", bus, values)


def create_friction_brake_command(packer, bus, apply_brake, idx, enabled, near_stop, at_full_stop, CP):
  mode = 0x1

  # TODO: Understand this better. Volts and ICE Camera ACC cars are 0x1 when enabled with no brake
  if enabled and CP.carFingerprint in (CAR.CHEVROLET_BOLT_EUV,):
    mode = 0x9

  if apply_brake > 0:
    mode = 0xa
    if at_full_stop:
      mode = 0xd

    # TODO: this is to have GM bringing the car to complete stop,
    # but currently it conflicts with OP controls, so turned off. Not set by all cars
    #elif near_stop:
    #  mode = 0xb

  brake = (0x1000 - apply_brake) & 0xfff
  checksum = (0x10000 - (mode << 12) - brake - idx) & 0xffff

  values = {
    "RollingCounter": idx,
    "FrictionBrakeMode": mode,
    "FrictionBrakeChecksum": checksum,
    "FrictionBrakeCmd": -apply_brake
  }

  return packer.make_can_msg("EBCMFrictionBrakeCmd", bus, values)


def create_acc_dashboard_command(packer, bus, enabled, target_speed_kph, hud_control, fcw):
  target_speed = min(target_speed_kph, 255)

  values = {
    "ACCAlwaysOne": 1,
    "ACCResumeButton": 0,
    "ACCSpeedSetpoint": target_speed,
    "ACCGapLevel": hud_control.leadDistanceBars * enabled,  # 3 "far", 0 "inactive"
    "ACCCmdActive": enabled,
    "ACCAlwaysOne2": 1,
    "ACCLeadCar": hud_control.leadVisible,
    "FCWAlert": 0x3 if fcw else 0
  }

  return packer.make_can_msg("ASCMActiveCruiseControlStatus", bus, values)


def create_adas_time_status(bus, tt, idx):
  dat = [(tt >> 20) & 0xff, (tt >> 12) & 0xff, (tt >> 4) & 0xff,
         ((tt & 0xf) << 4) + (idx << 2)]
  chksum = 0x1000 - dat[0] - dat[1] - dat[2] - dat[3]
  chksum = chksum & 0xfff
  dat += [0x40 + (chksum >> 8), chksum & 0xff, 0x12]
  return CanData(0xa1, bytes(dat), bus)


def create_adas_steering_status(bus, idx):
  dat = [idx << 6, 0xf0, 0x20, 0, 0, 0]
  chksum = 0x60 + sum(dat)
  dat += [chksum >> 8, chksum & 0xff]
  return CanData(0x306, bytes(dat), bus)


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
  return CanData(0x308, bytes(dat), bus)


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
  return CanData(0x104c006c, dat, bus)
