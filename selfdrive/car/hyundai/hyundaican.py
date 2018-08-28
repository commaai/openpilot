import crcmod

hyundai_checksum = crcmod.mkCrcFun(0x11D, initCrc=0xFD, rev=False, xorOut=0xdf)

def make_can_msg(addr, dat, alt, cks=False):
  if cks:
    dat = fix(dat, addr)
  return [addr, 0, dat, alt]


def create_lkas11(packer, apply_steer, steer_req, cnt, enabled):

##  values = {
##    "CF_Lkas_LdwsSysState": lkas11.CF_Lkas_LdwsSysState,
##    "CF_Lkas_SysWarning": lkas11.CF_Lkas_SysWarning,
##    "CF_Lkas_LdwsLHWarning": lkas11.CF_Lkas_LdwsLHWarning,
##    "CF_Lkas_LdwsRHWarning": lkas11.CF_Lkas_LdwsRHWarning,
##    "CF_Lkas_HbaLamp": lkas11.CF_Lkas_HbaLamp,
##    "CF_Lkas_FcwBasReq": lkas11.CF_Lkas_FcwBasReq,
##    "CR_Lkas_StrToqReq": apply_steer,
##    "CF_Lkas_ActToi": steer_req,
##    "CF_Lkas_ToiFlt": lkas11.CF_Lkas_ToiFlt,
##    "CF_Lkas_HbaSysState": lkas11.CF_Lkas_HbaSysState,
##    "CF_Lkas_FcwOpt": lkas11.CF_Lkas_FcwOpt,
##    "CF_Lkas_HbaOpt": lkas11.CF_Lkas_HbaOpt,
##    "CF_Lkas_MsgCount": cnt,
##    "CF_Lkas_FcwSysState": lkas11.CF_Lkas_FcwSysState,
##    "CF_Lkas_FcwCollisionWarning": lkas11.CF_Lkas_FcwCollisionWarning,
##    "CF_Lkas_FusionState": lkas11.CF_Lkas_FusionState,
##    "CF_Lkas_Chksum": 0,
##    "CF_Lkas_FcwOpt_USM": lkas11.CF_Lkas_FcwOpt_USM,
##    "CF_Lkas_LdwsOpt_USM": lkas11.CF_Lkas_LdwsOpt_USM,
##  }

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
  return make_can_msg(1342, "\x00\x00\x00\x00\x60\x05", 0, False)


def create_1191():
  return make_can_msg(1191, "\x01\x00", 0, False)


def create_1156():
  return make_can_msg(1156, "\x08\x20\xfe\x3f\x00\xe0\xfd\x3f", 0, False)


##def create_mdps12(packer, steer_req, cnt, mdps12):
##
##  values = {
##    "CR_Mdps_StrColTq": mdps12.CR_Mdps_StrColTq,
##    "CF_Mdps_Def": mdps12.CF_Mdps_Def,
##    "CF_Mdps_ToiUnavail": mdps12.CF_Mdps_ToiUnavail,
##    "CF_Mdps_ToiActive": mdps12.CF_Mdps_ToiActive,
##    "CF_Mdps_ToiFlt": mdps12.CF_Mdps_ToiFlt,
##    "CF_Mdps_FailStat": mdps12.CF_Mdps_FailStat,
##    "CF_Mdps_MsgCount2": cnt,
##    "CF_Mdps_Chksum2": 0,
##    "CF_Mdps_SErr": mdps12.CF_Mdps_SErr,
##    "CR_Mdps_StrTq": mdps12.CR_Mdps_StrTq,
##    "CR_Mdps_OutTq": mdps12.CR_Mdps_OutTq,
##  }
##
##  dat = packer.make_can_msg("MDPS12", 0, values)[2]
##  values["CF_Mdps_Chksum2"] = sum([ord(i) for i in dat]) & 0xff
##
##  return packer.make_can_msg("MDPS12", 2, values)
