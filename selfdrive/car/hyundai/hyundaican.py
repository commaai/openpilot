import crcmod
from selfdrive.car.hyundai.values import CAR, CHECKSUM

hyundai_checksum = crcmod.mkCrcFun(0x11D, initCrc=0xFD, rev=False, xorOut=0xdf)

def create_scc11(packer, enabled, count):
  objValid = 0
  objStatus = 0
  objDist = 150
  if enabled:
    objValid = 1
    objStatus = 1
    objDist = 3
  values = {
    "MainMode_ACC": enabled,
    "SCCInfoDisplay": 0,
    "AliveCounterACC": count,
    "VSetDis": 0,  # km/h velosity
    "ObjValid": objValid,
    "DriverAlertDisplay": 0,
    "TauGapSet": 1,
    "Navi_SCC_Curve_Status": 0,
    "Navi_SCC_Curve_Act": 0,
    "Navi_SCC_Camera_Act": 0,
    "Navi_SCC_Camera_Status": 0,
    "ACC_ObjStatus": objStatus,
    "ACC_ObjDist": objDist,
    "ACC_ObjLatPos":0,
    "ACC_ObjRelSpd":0,
  }
  return packer.make_can_msg("SCC11", 0, values)

def create_scc12(packer, apply_accel, enabled, cnt, scc12):
  values = {
    "CF_VSM_Prefill": 0,
    "CF_VSM_DecCmdAct": 0,
    "CF_VSM_HBACmd": 0,
    "CF_VSM_Warn": 0,
    "CF_VSM_Stat": 0,
    "CF_VSM_BeltCmd": 0,
    "ACCFailInfo": 0,
    "ACCMode": enabled,
    "StopReq": 0,
    "CR_VSM_DecCmd": 0,
    "aReqMax": apply_accel+3.0 if enabled else 0,
    "TakeOverReq": 0,
    "PreFill": 0,
    "aReqMin": apply_accel+3.0 if enabled else -10.23,
    "CF_VSM_ConfMode": 0,
    "AEB_Failinfo": 0,
    "AEB_Status": 0,
    "AEB_CmdAct": 0,
    "AEB_StopReq": 0,
    "CR_VSM_Alive": cnt,
    "CR_VSM_ChkSum": 0,
  }
  dat = packer.make_can_msg("SCC12", 0, values)[2]
  values["CR_VSM_ChkSum"] = 16 - sum([sum(divmod(i, 16)) for i in dat]) % 16

  return packer.make_can_msg("SCC12", 0, values)

def create_scc13(packer):
  values = {
    "SCCDrvModeRValue" : 2,
    "SCC_Equip" : 1,
    "AebDrvSetStatus" : 0,
  }
  return packer.make_can_msg("SCC13", 0, values)

def create_scc14(packer, enabled):
  if enabled:
    values = {
      "JerkUpperLimit" : 3.2,
      "JerkLowerLimit" : 0.1,
      "SCCMode2" : 1,
      "ComfortBandUpper" : 0.24,
      "ComfortBandLower" : 0.24,
    }
  else:
    values = {
      "JerkUpperLimit" : 0,
      "JerkLowerLimit" : 0,
      "SCCMode2" : 0,
      "ComfortBandUpper" : 0,
      "ComfortBandLower" : 0,
    }
  return packer.make_can_msg("SCC14", 0, values)

def create_4a2SCC(packer):
  values = {
    "Paint_1": 1
  }
  return packer.make_can_msg("4a2SCC", 0, values)

def create_lkas11(packer, frame, car_fingerprint, apply_steer, steer_req,
                  lkas11, sys_warning, sys_state, enabled,
                  left_lane, right_lane,
                  left_lane_depart, right_lane_depart):
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

  if car_fingerprint in [CAR.SONATA, CAR.PALISADE]:
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

  elif car_fingerprint == CAR.HYUNDAI_GENESIS:
    # This field is actually LdwsActivemode
    # Genesis and Optima fault when forwarding while engaged
    values["CF_Lkas_Bca_R"] = 2
  elif car_fingerprint == CAR.KIA_OPTIMA:
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


def create_clu11(packer, frame, clu11, button):
  values = clu11
  values["CF_Clu_CruiseSwState"] = button
  values["CF_Clu_AliveCnt1"] = frame % 0x10
  return packer.make_can_msg("CLU11", 0, values)


def create_lfa_mfa(packer, frame, enabled):
  values = {
    "ACTIVE": enabled,
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
