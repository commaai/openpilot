import crcmod
from selfdrive.car.hyundai.values import CAR, CHECKSUM

hyundai_checksum = crcmod.mkCrcFun(0x11D, initCrc=0xFD, rev=False, xorOut=0xdf)


def create_lkas11(packer, frame, car_fingerprint, apply_steer, steer_req,
                  lkas11, sys_warning, sys_state, enabled,
                  left_lane, right_lane,
                  left_lane_depart, right_lane_depart, bus):
  values = lkas11
  values["CF_Lkas_LdwsSysState"] = sys_state
  values["CF_Lkas_SysWarning"] = 3 if sys_warning else 0
  values["CF_Lkas_LdwsLHWarning"] = left_lane_depart
  values["CF_Lkas_LdwsRHWarning"] = right_lane_depart
  values["CR_Lkas_StrToqReq"] = apply_steer
  values["CF_Lkas_ActToi"] = steer_req
  values["CF_Lkas_ToiFlt"] = 0
  values["CF_Lkas_MsgCount"] = frame % 0x10
  values["CF_Lkas_Chksum"] = 0

  if car_fingerprint in [CAR.SONATA, CAR.PALISADE, CAR.SONATA_H, CAR.SANTA_FE]:
    values["CF_Lkas_Bca_R"] = int(left_lane) + (int(right_lane) << 1)
    values["CF_Lkas_LdwsOpt_USM"] = 2

    # FcwOpt_USM 5 = Orange blinking car + lanes
    # FcwOpt_USM 4 = Orange car + lanes
    # FcwOpt_USM 3 = Green blinking car + lanes
    # FcwOpt_USM 2 = Green car + lanes
    # FcwOpt_USM 1 = White car + lanes
    # FcwOpt_USM 0 = No car + lanes
    values["CF_Lkas_FcwOpt_USM"] = 2 if enabled else 1

    # SysWarning 4 = keep hands on wheel
    # SysWarning 5 = keep hands on wheel (red)
    # SysWarning 6 = keep hands on wheel (red) + beep
    # Note: the warning is hidden while the blinkers are on
    values["CF_Lkas_SysWarning"] = 4 if sys_warning else 0

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

  return packer.make_can_msg("LKAS11", bus, values)

def create_clu11(packer, frame, bus, clu11, button, speed):
  values = clu11
  values["CF_Clu_CruiseSwState"] = button
  values["CF_Clu_Vanz"] = speed
  values["CF_Clu_AliveCnt1"] = frame // 2 % 0x10
  return packer.make_can_msg("CLU11", bus, values)

def create_lfa_mfa(packer, frame, enabled):
  values = {
    "ACTIVE": enabled,
    "HDA_USM": 2,
  }

  # ACTIVE 1 = Green steering wheel icon

  # LFA_USM 2 & 3 = LFA cancelled, fast loud beeping
  # LFA_USM 0 & 1 = No mesage

  # LFA_SysWarning 1 = "Switching to HDA", short beep
  # LFA_SysWarning 2 = "Switching to Smart Cruise control", short beep
  # LFA_SysWarning 3 =  LFA error

  # ACTIVE2: nothing
  # HDA_USM: nothing

  return packer.make_can_msg("LFAHDA_MFC", 0, values)

def create_mdps12(packer, frame, mdps12):
  values = mdps12
  values["CF_Mdps_ToiActive"] = 0
  values["CF_Mdps_ToiUnavail"] = 1
  values["CF_Mdps_MsgCount2"] = frame % 0x100
  values["CF_Mdps_Chksum2"] = 0

  dat = packer.make_can_msg("MDPS12", 2, values)[2]
  checksum = sum(dat) % 256
  values["CF_Mdps_Chksum2"] = checksum

  return packer.make_can_msg("MDPS12", 2, values)

def create_scc11(packer, frame, enabled, set_speed, lead_visible, scc_live, scc11):
  values = scc11
  values["AliveCounterACC"] = frame // 2 % 0x10
  if not scc_live:
    values["MainMode_ACC"] = 1
    values["VSetDis"] = set_speed
    values["ObjValid"] = 1 if enabled else 0
#  values["ACC_ObjStatus"] = lead_visible

  return packer.make_can_msg("SCC11", 0, values)

def create_scc12(packer, apply_accel, enabled, cnt, scc_live, scc12):
  values = scc12
  values["aReqRaw"] = apply_accel if enabled else 0 #aReqMax
  values["aReqValue"] = apply_accel if enabled else 0 #aReqMin
  values["CR_VSM_Alive"] = cnt
  values["CR_VSM_ChkSum"] = 0
  if not scc_live:
    values["ACCMode"] = 1  if enabled else 0 # 2 if gas padel pressed

  dat = packer.make_can_msg("SCC12", 0, values)[2]
  values["CR_VSM_ChkSum"] = 16 - sum([sum(divmod(i, 16)) for i in dat]) % 16

  return packer.make_can_msg("SCC12", 0, values)

def create_scc13(packer, scc13):
  values = scc13
  return packer.make_can_msg("SCC13", 0, values)

def create_scc14(packer, enabled, scc14):
  values = scc14
  if enabled:
    values["JerkUpperLimit"] = 3.2
    values["JerkLowerLimit"] = 0.1
    values["SCCMode"] = 1
    values["ComfortBandUpper"] = 0.24
    values["ComfortBandLower"] = 0.24

  return packer.make_can_msg("SCC14", 0, values)

def create_spas11(packer, car_fingerprint, frame, en_spas, apply_steer, bus):
  values = {
    "CF_Spas_Stat": en_spas,
    "CF_Spas_TestMode": 0,
    "CR_Spas_StrAngCmd": apply_steer,
    "CF_Spas_BeepAlarm": 0,
    "CF_Spas_Mode_Seq": 2,
    "CF_Spas_AliveCnt": frame % 0x200,
    "CF_Spas_Chksum": 0,
    "CF_Spas_PasVol": 0,
  }
  dat = packer.make_can_msg("SPAS11", 0, values)[2]
  if car_fingerprint in CHECKSUM["crc8"]:
    dat = dat[:6]
    values["CF_Spas_Chksum"] = hyundai_checksum(dat)
  else:
    values["CF_Spas_Chksum"] = sum(dat[:6]) % 256
  return packer.make_can_msg("SPAS11", bus, values)

def create_spas12(bus):
  return [1268, 0, "\x00\x00\x00\x00\x00\x00\x00\x00", bus]

def create_ems11(packer, ems11, enabled):
  values = ems11
  if enabled:
    values["VS"] = 0
  return packer.make_can_msg("values", 1, ems11)
