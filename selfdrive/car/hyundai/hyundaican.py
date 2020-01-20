import crcmod
from selfdrive.car.hyundai.values import CAR, CHECKSUM

hyundai_checksum = crcmod.mkCrcFun(0x11D, initCrc=0xFD, rev=False, xorOut=0xdf)

def create_lkas11(packer, car_fingerprint, apply_steer, steer_req, cnt, enabled, lkas11, hud_alert,
                                   lane_visible, left_lane_depart, right_lane_depart, keep_stock=False):
  values = {
    "CF_Lkas_Bca_R": lkas11["CF_Lkas_Bca_R"] if keep_stock else 3,
    "CF_Lkas_LdwsSysState": lane_visible,
    "CF_Lkas_SysWarning": hud_alert,
    "CF_Lkas_LdwsLHWarning": left_lane_depart,
    "CF_Lkas_LdwsRHWarning": right_lane_depart,
    "CF_Lkas_HbaLamp": lkas11["CF_Lkas_HbaLamp"] if keep_stock else 0,
    "CF_Lkas_FcwBasReq": lkas11["CF_Lkas_FcwBasReq"] if keep_stock else 0,
    "CR_Lkas_StrToqReq": apply_steer,
    "CF_Lkas_ActToi": steer_req,
    "CF_Lkas_ToiFlt": 0,
    "CF_Lkas_HbaSysState": lkas11["CF_Lkas_HbaSysState"] if keep_stock else 1,
    "CF_Lkas_FcwOpt": lkas11["CF_Lkas_FcwOpt"] if keep_stock else 0,
    "CF_Lkas_HbaOpt": lkas11["CF_Lkas_HbaOpt"] if keep_stock else 3,
    "CF_Lkas_MsgCount": cnt,
    "CF_Lkas_FcwSysState": lkas11["CF_Lkas_FcwSysState"] if keep_stock else 0,
    "CF_Lkas_FcwCollisionWarning": lkas11["CF_Lkas_FcwCollisionWarning"] if keep_stock else 0,
    "CF_Lkas_FusionState": lkas11["CF_Lkas_FusionState"] if keep_stock else 0,
    "CF_Lkas_Chksum": 0,
    "CF_Lkas_FcwOpt_USM": lkas11["CF_Lkas_FcwOpt_USM"] if keep_stock else 2,
    "CF_Lkas_LdwsOpt_USM": lkas11["CF_Lkas_LdwsOpt_USM"] if keep_stock else 3,
  }
  if car_fingerprint == CAR.GENESIS:
    values["CF_Lkas_Bca_R"] = 2
    values["CF_Lkas_HbaSysState"] = lkas11["CF_Lkas_HbaSysState"] if keep_stock else 0
    values["CF_Lkas_HbaOpt"] = lkas11["CF_Lkas_HbaOpt"] if keep_stock else 1
    values["CF_Lkas_FcwOpt_USM"] = lkas11["CF_Lkas_FcwOpt_USM"] if keep_stock else 2
    values["CF_Lkas_LdwsOpt_USM"] = lkas11["CF_Lkas_LdwsOpt_USM"] if keep_stock else 0
  if car_fingerprint == CAR.KIA_OPTIMA:
    values["CF_Lkas_Bca_R"] = 0
    values["CF_Lkas_HbaOpt"] = lkas11["CF_Lkas_HbaOpt"] if keep_stock else 1
    values["CF_Lkas_FcwOpt_USM"] = lkas11["CF_Lkas_FcwOpt_USM"] if keep_stock else 0

  dat = packer.make_can_msg("LKAS11", 0, values)[2]

  if car_fingerprint in CHECKSUM["crc8"]:
    # CRC Checksum as seen on 2019 Hyundai Santa Fe
    dat = dat[:6] + dat[7:8]
    checksum = hyundai_checksum(dat)
  elif car_fingerprint in CHECKSUM["6B"]:
    # Checksum of first 6 Bytes, as seen on 2018 Kia Sorento
    checksum = sum(dat[:6]) % 256
  else:
    # Checksum of first 6 Bytes and last Byte as seen on 2018 Kia Stinger
    checksum = (sum(dat[:6]) + dat[7]) % 256

  values["CF_Lkas_Chksum"] = checksum

  return packer.make_can_msg("LKAS11", 0, values)

def create_clu11(packer, clu11, button, cnt):
  values = {
    "CF_Clu_CruiseSwState": button,
    "CF_Clu_CruiseSwMain": clu11["CF_Clu_CruiseSwMain"],
    "CF_Clu_SldMainSW": clu11["CF_Clu_SldMainSW"],
    "CF_Clu_ParityBit1": clu11["CF_Clu_ParityBit1"],
    "CF_Clu_VanzDecimal": clu11["CF_Clu_VanzDecimal"],
    "CF_Clu_Vanz": clu11["CF_Clu_Vanz"],
    "CF_Clu_SPEED_UNIT": clu11["CF_Clu_SPEED_UNIT"],
    "CF_Clu_DetentOut": clu11["CF_Clu_DetentOut"],
    "CF_Clu_RheostatLevel": clu11["CF_Clu_RheostatLevel"],
    "CF_Clu_CluInfo": clu11["CF_Clu_CluInfo"],
    "CF_Clu_AmpInfo": clu11["CF_Clu_AmpInfo"],
    "CF_Clu_AliveCnt1": cnt,
  }

  return packer.make_can_msg("CLU11", 0, values)
