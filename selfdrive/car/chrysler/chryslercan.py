from cereal import car
from selfdrive.car import make_can_msg


GearShifter = car.CarState.GearShifter
VisualAlert = car.CarControl.HUDControl.VisualAlert

def calc_checksum(data):
  """This function does not want the checksum byte in the input data.

  jeep chrysler canbus checksum from http://illmatics.com/Remote%20Car%20Hacking.pdf
  """
  checksum = 0xFF
  for curr in data[:-1]:
    shift = 0x80
    for _ in range(0, 8):
      bit_sum = curr & shift
      temp_chk = checksum & 0x80
      if (bit_sum != 0):
        bit_sum = 0x1C
        if (temp_chk != 0):
          bit_sum = 1
        checksum = checksum << 1
        temp_chk = checksum | 1
        bit_sum ^= temp_chk
      else:
        if (temp_chk != 0):
          bit_sum = 0x1D
        checksum = checksum << 1
        bit_sum ^= checksum
      checksum = bit_sum
      shift = shift >> 1
  return ~checksum & 0xFF


def create_lkas_hud(packer, gear, lkas_active, hud_alert, hud_count, lkas_car_model):
  # LKAS_HUD 0x2a6 (678) Controls what lane-keeping icon is displayed.

  if hud_alert == VisualAlert.steerRequired:
    msg = b'\x00\x00\x00\x03\x00\x00\x00\x00'
    return make_can_msg(0x2a6, msg, 0)

  color = 1  # default values are for park or neutral in 2017 are 0 0, but trying 1 1 for 2019
  lines = 1
  alerts = 0

  if hud_count < (1 *4):  # first 3 seconds, 4Hz
    alerts = 1
  # CAR.PACIFICA_2018_HYBRID and CAR.PACIFICA_2019_HYBRID
  # had color = 1 and lines = 1 but trying 2017 hybrid style for now.
  if gear in (GearShifter.drive, GearShifter.reverse, GearShifter.low):
    if lkas_active:
      color = 2  # control active, display green.
      lines = 6
    else:
      color = 1  # control off, display white.
      lines = 1

  values = {
    "LKAS_ICON_COLOR": color,  # byte 0, last 2 bits
    "CAR_MODEL": lkas_car_model,  # byte 1
    "LKAS_LANE_LINES": lines,  # byte 2, last 4 bits
    "LKAS_ALERTS": alerts,  # byte 3, last 4 bits
    }

  return packer.make_can_msg("LKAS_HUD", 0, values)  # 0x2a6


def create_lkas_command(packer, apply_steer, moving_fast, frame):
  # LKAS_COMMAND 0x292 (658) Lane-keeping signal to turn the wheel.
  values = {
    "LKAS_STEERING_TORQUE": apply_steer,
    "LKAS_HIGH_TORQUE": int(moving_fast),
    "COUNTER": frame % 0x10,
  }

  dat = packer.make_can_msg("LKAS_COMMAND", 0, values)[2]
  checksum = calc_checksum(dat)

  values["CHECKSUM"] = checksum
  return packer.make_can_msg("LKAS_COMMAND", 0, values)


def create_wheel_buttons(frame):
  # WHEEL_BUTTONS (571) Message sent to cancel ACC.
  start = b"\x01"  # acc cancel set
  counter = (frame % 10) << 4
  dat = start + counter.to_bytes(1, 'little') + b"\x00"
  dat = dat[:-1] + calc_checksum(dat).to_bytes(1, 'little')
  return make_can_msg(0x23b, dat, 0)
