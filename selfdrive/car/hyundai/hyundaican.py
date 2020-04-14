import crcmod
from selfdrive.car.hyundai.values import CAR, CHECKSUM

hyundai_checksum = crcmod.mkCrcFun(0x11D, initCrc=0xFD, rev=False, xorOut=0xdf)


def create_lkas11(packer, car_fingerprint, apply_steer, steer_req, cnt, enabled, lkas11, hud_alert,
                  lane_visible, left_lane_depart, right_lane_depart):
  values = lkas11
  values["CF_Lkas_LdwsSysState"] = lane_visible
  values["CF_Lkas_SysWarning"] = hud_alert
  values["CF_Lkas_LdwsLHWarning"] = left_lane_depart
  values["CF_Lkas_LdwsRHWarning"] = right_lane_depart
  values["CR_Lkas_StrToqReq"] = apply_steer
  values["CF_Lkas_ActToi"] = steer_req
  values["CF_Lkas_ToiFlt"] = 0
  values["CF_Lkas_MsgCount"] = cnt
  values["CF_Lkas_Chksum"] = 0

  # This field is actually LdwsActivemode
  # Genesis and Optima fault when forwarding while engaged
  if car_fingerprint == CAR.HYUNDAI_GENESIS:
    values["CF_Lkas_Bca_R"] = 2
  if car_fingerprint == CAR.KIA_OPTIMA:
    values["CF_Lkas_Bca_R"] = 0

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
  values = clu11
  values["CF_Clu_CruiseSwState"] = button
  values["CF_Clu_CruiseSwState"] = cnt
  return packer.make_can_msg("CLU11", 0, values)
