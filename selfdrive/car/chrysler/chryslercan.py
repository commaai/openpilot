import struct
import binascii


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


def make_can_msg(addr, dat, alt, cks=False, counter=None):
  # We're not actually using cks and counter, just doing manually in create_ TODO
  if counter != None:
    dat = dat  # TODO!!! verify 0..15 and put counter in as high nibble
  if cks:
    dat = dat + struct.pack("B", calc_checksum(dat))
  print 'make_can_msg:%s  len:%d  %s' % ('0x{:02x}'.format(addr), len(dat), binascii.hexlify(dat))
  return [addr, 0, dat, alt]
  # TODO what is alt? look at can_list_to_can_capnp. looks like messaging.new_message() .src

def create_2d9():
  msg = '0000000820'.decode('hex')
  return make_can_msg(0x2d9, msg, 1)

def create_2a6(gear, steering):
  msg = '0000000000000000'.decode('hex')  # park or neutral
  if (gear == 'drive' or gear == 'reverse'):
    # msg = '0100010000000000'.decode('hex') # moving slowly, display white.
    msg = '0200060000000000'.decode('hex') # moving fast, display green.
  if steering:
    msg = '03000a0000000000'.decode('hex')  # when torqueing, display yellow.
  return make_can_msg(0x2a6, msg, 1)

def create_292(apply_angle, frame):
  combined_torque = apply_angle + 1024  # 1024 is straight. more is left, less is right.
  start = [0x10 | (combined_torque >> 8), combined_torque & 0xff, 00, 00]
  counter = (frame % 0x10) << 4
  dat = start + [counter]
  dat = dat + [calc_checksum(dat)]  # this calc_checksum does not include the length
  return make_can_msg(0x292, str(bytearray(dat)), 1)
  
