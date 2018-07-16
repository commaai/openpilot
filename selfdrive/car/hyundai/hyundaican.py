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



def create_steer_command(packer, car_fingerprint, steer, idx, checksum):
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

def create_lkas11(packer, byte0, byte1, steer, steer_required, nibble5, \
  byte4, byte5, checksum, byte7):
  """Creates a CAN message for the Hyundai LKAS11."""
  values = {
    'Byte0' : byte0,
    'Byte1' : byte1,
    'CR_Lkas_StrToqReq' : steer, #actual torque request
    'CF_Lkas_ActToi': steer_required, #the torque request bit
    'Nibble5' : nibble5,
    'Byte4' : byte4,
    'Byte5' : byte5,
    'CF_Lkas_Chksum' : checksum,
    'Byte7' : byte7,
  }

  return packer.make_can_msg("LKAS11", 2, values)




def create_lkas12(packer):
  values = {
    'Byte0' : 0x00,
    'Byte1' : 0x00,
    'Byte2' : 0x00,
    'Byte3' : 0x00,
    'Byte4' : 0x20,
    'Byte5' : 0x00,
  }

  return packer.make_can_msg("LKAS12", 2, values)
