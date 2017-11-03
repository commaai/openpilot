import struct
import common.numpy_fast as np
from selfdrive.config import Conversions as CV


# *** Toyota specific ***

def fix(msg, addr):
  checksum = 0
  idh = (addr & 0xff00) >> 8
  idl = (addr & 0xff)
  
  checksum = idh + idl + len(msg) + 1
  for d_byte in msg:
    checksum += ord(d_byte)
  
  #return msg + chr(checksum & 0xFF)
  return msg + struct.pack("B", checksum & 0xFF)
  
    
def make_can_msg(addr, dat, alt, cks=False):
  if cks:
    dat = fix(dat, addr)
  return [addr, 0, dat, alt]
  
  
def create_video_target(frame, addr):
  counter = frame & 0xff
  msg = struct.pack("!BBBBBBB", counter, 0x03, 0xff, 0x00, 0x00, 0x00, 0x00)
  return make_can_msg(addr, msg, 1, True)


def create_ipas_steer_command(steer):

  """Creates a CAN message for the Toyota Steer Command."""
  if steer < 0:
    move = 0x60
    steer = 0xfff + steer + 1
  elif steer > 0:
    move = 0x20
  else:
    move = 0x40

  mode = 0x30 if steer else 0x10
  
  steer_h = (steer & 0xF00) >> 8
  steer_l = steer & 0xff

  msg = struct.pack("!BBBBBBB", mode | steer_h, steer_l, 0x10, 0x00, move, 0x40, 0x00)

  return make_can_msg(0x266, msg, 0, True)

def create_steer_command(steer, raw_cnt):
  """Creates a CAN message for the Toyota Steer Command."""
  # from 0x80 to 0xff
  counter = ((raw_cnt & 0x3f) << 1) | 0x80
  if steer != 0:
    counter |= 1
    
  # hud
  # 00 => Regular
  # 40 => Actively Steering (with beep)
  # 80 => Actively Steering (without beep) 
  hud = 0x00

  msg = struct.pack("!BhB", counter, steer, hud)

  return make_can_msg(0x2e4, msg, 0, True)


def create_accel_command(accel, pcm_cancel):
  # TODO: find the exact canceling bit
  state = 0xc0 + pcm_cancel # this allows automatic restart from hold without driver cmd

  msg = struct.pack("!hBBBBB", accel, 0x63, state, 0x00, 0x00, 0x00)

  return make_can_msg(0x343, msg, 0, True)


def create_ui_command(hud1, hud2):
  msg = struct.pack('!BBBBBBBB', 0x54, 0x04 + hud1 + (hud2 << 4), 0x0C, 
                                 0x00, 0x00, 0x2C, 0x38, 0x02) 
  return make_can_msg(0x412, msg, 0, False)
