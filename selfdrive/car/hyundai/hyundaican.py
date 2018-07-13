import struct

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



def create_steer_command(packer, steer, car_fingerprint, idx):
  """Creates a CAN message for the Hyundai  Steering and LKAS UI command."""
  lkas_hud_values = {
    #checksum and counter are calculated elsewhere
    'CF_Lkas_LdwsSysState' : hud.lanes,
    'CF_Lkas_SysWarning' : hud.steer_required,
    'CF_Lkas_LdwsLHWarning' : 0x0,
    'CF_Lkas_LdwsRHWarning' : 0x0,
    'CF_Lkas_HbaLamp' : 0x0,
    'CF_Lkas_FcwBasReq' : 0x0,
    'CR_Lkas_StrToqReq' : steer, #actual torque request
    'CF_Lkas_ActToi': steer != 0, #the torque request bit
    'CF_Lkas_ToiFlt' : 0x0,
    'CF_Lkas_HbaSysState' : 0x1,
    'CF_Lkas_FcwOpt' : 0x0,
    'CF_Lkas_HbaOpt' : 0x1,
    'CF_Lkas_FcwSysState' : 0x0,
    'CF_Lkas_FcwCollisionWarning' : 0x0,
    'CF_Lkas_FusionState' : 0x0,
    'CF_Lkas_FcwOpt_USM' : 0x0,
    'CF_Lkas_LdwsOpt_USM' : 0x3,
  }

  return packer.make_can_msg("LKAS11", 0, lkas_hud_values, idx)


def create_accel_command(packer, accel, pcm_cancel, standstill_req):
  # TODO: find the exact canceling bit
  #values = {
  #  "ACCEL_CMD": accel,
  #  "SET_ME_X63": 0x63,
  #  "SET_ME_1": 1,
  #  "RELEASE_STANDSTILL": not standstill_req,
  #  "CANCEL_REQ": pcm_cancel,
  #}
  #return packer.make_can_msg("ACC_CONTROL", 0, values)
  return -1


def create_fcw_command(packer, fcw):
 # values = {
 #   "FCW": fcw,
 #   "SET_ME_X20": 0x20,
 #   "SET_ME_X10": 0x10,
 #   "SET_ME_X80": 0x80,
 # }
 # return packer.make_can_msg("ACC_HUD", 0, values)
  return -1
