import struct
import binascii
from common.numpy_fast import clip
from selfdrive.car.chrysler.values import CAR


# *** Chrysler specific ***

def calc_checksum(data):
  """This version does not want checksum byte in input data.

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


def make_can_msg(addr, dat, alt=0, cks=False, counter=None):
  # We're not actually using cks and counter, just doing manually in create_ TODO
  # TODO what is the alt parameter? setting it to 0 works. 1 does not.
  #      for alt, look at can_list_to_can_capnp. looks like messaging.new_message() .src
  if counter != None:
    dat = dat  # TODO!!! verify 0..15 and put counter in as high nibble
  if cks:
    dat = dat + struct.pack("B", calc_checksum(dat))
  # if addr == 0x292:
  #   print ('make_can_msg:%s  len:%d  %s' % ('0x{:02x}'.format(addr), len(dat),
  #                                           ' '.join('{:02x}'.format(ord(c)) for c in dat)))
  return [addr, 0, dat, alt]

def create_2d9(car_fingerprint):
  msg = '0000000820'.decode('hex')  # 2017
  if car_fingerprint == CAR.PACIFICA_2018:
    msg = '0000000020'.decode('hex')
  elif car_fingerprint == CAR.CHEROKEE:
    msg = '0000000040'.decode('hex')
  elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
    msg = '0000000440'.decode('hex')
  return make_can_msg(0x2d9, msg)

def create_2a6(gear, apply_steer, moving_fast, car_fingerprint):
  msg = '0000000000000000'.decode('hex')  # park or neutral
  if car_fingerprint == CAR.PACIFICA_2018:
    msg = '0064000000000000'.decode('hex')  # Have not verified 2018 park with a real car.
  elif car_fingerprint == CAR.CHEROKEE:
    msg = '00a4000000000000'.decode('hex')  # Have not verified 2018 park with a real car.
  elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
    msg = '01a8010000000000'.decode('hex')
  if (gear == 'drive' or gear == 'reverse'):
    if moving_fast:
      msg = '0200060000000000'.decode('hex') # moving fast, display green.
      if car_fingerprint == CAR.PACIFICA_2018:
        msg = '0264060000000000'.decode('hex')
      elif car_fingerprint == CAR.CHEROKEE:
        msg = '02a4060000000000'.decode('hex')
      elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
        msg = '02a8060000000000'.decode('hex')
    else:
      msg = '0100010000000000'.decode('hex') # moving slowly, display white.
      if car_fingerprint == CAR.PACIFICA_2018:
        msg = '0164010000000000'.decode('hex')
      elif car_fingerprint == CAR.CHEROKEE:
        msg = '01a4010000000000'.decode('hex')
      elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
        msg = '01a8010000000000'.decode('hex')
  if apply_steer > 0:  # steering left
    msg = '03000a0000000000'.decode('hex')  # when torqueing, display yellow.
    if car_fingerprint == CAR.PACIFICA_2018:
      msg = '03640a0000000000'.decode('hex')
    elif car_fingerprint == CAR.CHEROKEE:
      msg = '03a40a0000000000'.decode('hex')
    elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
      msg = '03a80a0000000000'.decode('hex')
  elif apply_steer < 0:  # steering right
    msg = '0300080000000000'.decode('hex')  # when torqueing, display yellow.
    if car_fingerprint == CAR.PACIFICA_2018:
      msg = '0364080000000000'.decode('hex')
    elif car_fingerprint == CAR.CHEROKEE:
      msg = '03a4080000000000'.decode('hex')
    elif car_fingerprint == CAR.PACIFICA_2018_HYBRID:
      msg = '03a8080000000000'.decode('hex')
  return make_can_msg(0x2a6, msg)

LIMIT = 230-3  # 230 is documented limit # 171 is max from main example
STEP = 3  # 3 is stock. originally 20. 100 is fine. 200 is too much it seems.
_prev_angle = 0  # TODO if this is needed, refactor it.

def create_292(apply_angle, frame, moving_fast):
  global _prev_angle, LIMIT, STEP
  apply_angle = int(apply_angle)
  if apply_angle > LIMIT:
    apply_angle = LIMIT
  if apply_angle < -LIMIT:
    apply_angle = -LIMIT
  apply_angle = clip(apply_angle, _prev_angle - STEP, _prev_angle + STEP)
  _prev_angle = apply_angle
  combined_torque = apply_angle + 1024  # 1024 is straight. more is left, less is right.
  high_status = 0x10  #!!  0x00 here is more correct, but can_game_sticky uses 0x10
  if moving_fast:
    high_status = 0x10
  start = [high_status | (combined_torque >> 8), combined_torque & 0xff, 00, 00]
  counter = (frame % 0x10) << 4
  dat = start + [counter]
  dat = dat + [calc_checksum(dat)]  # this calc_checksum does not include the length
  return make_can_msg(0x292, str(bytearray(dat)))
  
