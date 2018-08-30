import crcmod

hyundai_checksum = crcmod.mkCrcFun(0x11D, initCrc=0xFD, rev=False, xorOut=0xdf)

def make_can_msg(addr, dat, alt):
  return [addr, 0, dat, alt]


def create_lkas11(packer, apply_steer, steer_req, cnt, enabled):

  values = {
    "CF_Lkas_Icon": 3 if enabled else 0,
    "CF_Lkas_LdwsSysState": 1,
    "CF_Lkas_SysWarning": 0,
    "CF_Lkas_LdwsLHWarning": 0,
    "CF_Lkas_LdwsRHWarning": 0,
    "CF_Lkas_HbaLamp": 0,
    "CF_Lkas_FcwBasReq": 0,
    "CR_Lkas_StrToqReq": apply_steer,
    "CF_Lkas_ActToi": steer_req,
    "CF_Lkas_ToiFlt": 0,
    "CF_Lkas_HbaSysState": 1,
    "CF_Lkas_FcwOpt": 0,
    "CF_Lkas_HbaOpt": 3,
    "CF_Lkas_MsgCount": cnt,
    "CF_Lkas_FcwSysState": 0,
    "CF_Lkas_FcwCollisionWarning": 0,
    "CF_Lkas_FusionState": 0,
    "CF_Lkas_Chksum": 0,
    "CF_Lkas_FcwOpt_USM": 2 if enabled else 1,
    "CF_Lkas_LdwsOpt_USM": 3,
  }

  dat = packer.make_can_msg("LKAS11", 0, values)[2]
  dat = dat[:6] + dat[7]
  checksum = hyundai_checksum(dat)

  values["CF_Lkas_Chksum"] = checksum

  return packer.make_can_msg("LKAS11", 0, values)


def create_lkas12():
  return make_can_msg(1342, "\x00\x00\x00\x00\x60\x05", 0)


def create_1191():
  return make_can_msg(1191, "\x01\x00", 0)


def create_1156():
  return make_can_msg(1156, "\x08\x20\xfe\x3f\x00\xe0\xfd\x3f", 0)
