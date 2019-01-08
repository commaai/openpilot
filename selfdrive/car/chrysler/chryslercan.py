from cereal import car
from selfdrive.car.chrysler.values import CAR


VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = car.CarControl.HUDControl.AudibleAlert


def calc_checksum(data):
  """This function does not want the checksum byte in the input data.

  jeep chrysler canbus checksum from http://illmatics.com/Remote%20Car%20Hacking.pdf
  """
  end_index = len(data)
  index = 0
  checksum = 0xFF
  temp_chk = 0;
  bit_sum = 0;
  if(end_index <= index):
    return False
  for index in range(0, end_index):
    shift = 0x80
    curr = data[index]
    iterate = 8
    while(iterate > 0):
      iterate -= 1
      bit_sum = curr & shift;
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


def make_can_msg(addr, dat):
  return [addr, 0, dat, 0]

def create_lkas_heartbit(car_fingerprint):
  # LKAS_HEARTBIT (729) Lane-keeping heartbeat.
  msg = '0000000820'.decode('hex')  # 2017
  return make_can_msg(0x2d9, msg)

def create_lkas_hud(gear, lkas_active, hud_alert, car_fingerprint):
  # LKAS_HUD (678) Controls what lane-keeping icon is displayed.

  if hud_alert == VisualAlert.steerRequired:
    msg = msg = '0000000300000000'.decode('hex')
    return make_can_msg(0x2a6, msg)

  # TODO: use can packer
  msg = '0000000000000000'.decode('hex')  # park or neutral
  if car_fingerprint == CAR.PACIFICA_2018:
    msg = '0064000000000000'.decode('hex')  # Have not verified 2018 park with a real car.
  elif car_fingerprint == CAR.JEEP_CHEROKEE:
    msg = '00a4000000000000'.decode('hex')  # Have not verified 2018 park with a real car.
  elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
    msg = '01a8010000000000'.decode('hex')
  if (gear == 'drive' or gear == 'reverse'):
    if lkas_active:
      msg = '0200060000000000'.decode('hex') # control active, display green.
      if car_fingerprint == CAR.PACIFICA_2018:
        msg = '0264060000000000'.decode('hex')
      elif car_fingerprint == CAR.JEEP_CHEROKEE:
        msg = '02a4060000000000'.decode('hex')
      elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
        msg = '02a8060000000000'.decode('hex')
    else:
      msg = '0100010000000000'.decode('hex') # control off, display white.
      if car_fingerprint == CAR.PACIFICA_2018:
        msg = '0164010000000000'.decode('hex')
      elif car_fingerprint == CAR.JEEP_CHEROKEE:
        msg = '01a4010000000000'.decode('hex')
      elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
        msg = '01a8010000000000'.decode('hex')

  return make_can_msg(0x2a6, msg)


def create_lkas_command(packer, apply_steer, frame):
  # LKAS_COMMAND (658) Lane-keeping signal to turn the wheel.
  values = {
    "LKAS_STEERING_TORQUE": apply_steer,
    "LKAS_HIGH_TORQUE": 1,
    "COUNTER": frame % 0x10,
  }

  dat = packer.make_can_msg("LKAS_COMMAND", 0, values)[2]
  dat = [ord(i) for i in dat][:-1]
  checksum = calc_checksum(dat)

  values["CHECKSUM"] = checksum
  return packer.make_can_msg("LKAS_COMMAND", 0, values)


def create_chimes(audible_alert):
  # '0050' nothing, chime '4f55'
  if audible_alert == AudibleAlert.none:
    msg = '0050'.decode('hex')
  else:
    msg = '4f55'.decode('hex')
  return make_can_msg(0x339, msg)


def create_wheel_buttons(frame):
  # WHEEL_BUTTONS (571) Message sent to cancel ACC.
  start = [0x01]  # acc cancel set
  counter = (frame % 10) << 4
  dat = start + [counter]
  dat = dat + [calc_checksum(dat)]
  return make_can_msg(0x23b, str(bytearray(dat)))
