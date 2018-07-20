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



def create_lkas11_b(packer, lanes, steer, steer_required, idx, checksum):
  """Creates a CAN message for the Hyundai  Steering and LKAS UI command."""
  values = {
    #checksum and counter are calculated elsewhere
    'CF_Lkas_LdwsSysState' : lanes,
    'CR_Lkas_StrToqReq' : steer, #actual torque request
    'CF_Lkas_ActToi': steer_required, #the torque request bit
    'CF_Lkas_LdwsOpt_USM' : 0x6,
    'CF_Lkas_Chksum' : checksum,
    'CF_Lkas_MsgCount' : idx,
  }

  return packer.make_can_msg("LKAS11", 2, values)

def create_lkas11(packer, byte0, byte1, byte2, byte3, \
  byte4, byte5, byte6, byte7):
  """Creates a CAN message for the Hyundai LKAS11."""
  values = {
    'Byte0' : byte0,
    'Byte1' : byte1,
    'Byte2' : byte2,
    'Byte3' : byte3,
    'Byte4' : byte4,
    'Byte5' : byte5,
    'Byte6' : byte6,
    'Byte7' : byte7,
  }

  return packer.make_can_msg("LKAS11", 0, values)




def create_lkas12(packer, byte4, byte5):
  values = {
    'Byte0' : 0x00,
    'Byte1' : 0x00,
    'Byte2' : 0x00,
    'Byte3' : 0x00,
    'Byte4' : byte4,
    'Byte5' : byte5,
  }

  return packer.make_can_msg("LKAS12", 0, values)
