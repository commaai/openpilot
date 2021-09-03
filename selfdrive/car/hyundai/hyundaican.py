import crcmod
from selfdrive.swaglog import cloudlog
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from selfdrive.car.hyundai.values import CAR, CHECKSUM

RADAR_ADDR = 0x7D0
EXT_DIAG_REQUEST = b'\x10\x03'
EXT_DIAG_RESPONSE = b'\x50\x03'
COM_CONT_REQUEST = b'\x28\x83\x01'
COM_CONT_RESPONSE = b''

hyundai_checksum = crcmod.mkCrcFun(0x11D, initCrc=0xFD, rev=False, xorOut=0xdf)

# TODO: merge with honda.hondacan.disable_radar
def disable_radar(logcan, sendcan, bus=0, timeout=0.1, debug=False):
  """Silence the radar by disabling sending and receiving messages using UDS 0x28.
  The radar will stay silent as long as openpilot keeps sending Tester Present.
  Openpilot will emulate the radar. WARNING: THIS DISABLES AEB!"""
  cloudlog.warning(f"radar disable {hex(RADAR_ADDR)} ...")

  try:
    query = IsoTpParallelQuery(sendcan, logcan, bus, [RADAR_ADDR], [EXT_DIAG_REQUEST], [EXT_DIAG_RESPONSE], debug=debug)

    for _, _ in query.get_data(timeout).items():
      cloudlog.warning("radar communication control disable tx/rx ...")

      query = IsoTpParallelQuery(sendcan, logcan, bus, [RADAR_ADDR], [COM_CONT_REQUEST], [COM_CONT_RESPONSE], debug=debug)
      query.get_data(0)

      cloudlog.warning("radar disabled")
      return

  except Exception:
    cloudlog.exception("radar disable exception")
  cloudlog.warning("radar disable failed")


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
  values["CF_Lkas_MsgCount"] = frame % 0x10

  if car_fingerprint in [CAR.SONATA, CAR.PALISADE, CAR.KIA_NIRO_EV, CAR.KIA_NIRO_HEV_2021, CAR.SANTA_FE,
                         CAR.IONIQ_EV_2020, CAR.IONIQ_PHEV, CAR.KIA_SELTOS, CAR.ELANTRA_2021,
                         CAR.ELANTRA_HEV_2021, CAR.SONATA_HYBRID, CAR.KONA_EV, CAR.KONA_HEV]:
    values["CF_Lkas_LdwsActivemode"] = int(left_lane) + (int(right_lane) << 1)
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
    values["CF_Lkas_LdwsActivemode"] = 2
  elif car_fingerprint == CAR.KIA_OPTIMA:
    values["CF_Lkas_LdwsActivemode"] = 0

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


def create_lfahda_mfc(packer, enabled, hda_set_speed=0):
  values = {
    "LFA_Icon_State": 2 if enabled else 0,
    "HDA_Active": 1 if hda_set_speed else 0,
    "HDA_Icon_State": 2 if hda_set_speed else 0,
    "HDA_VSetReq": hda_set_speed,
  }
  return packer.make_can_msg("LFAHDA_MFC", 0, values)

def create_acc_commands(packer, enabled, accel, idx, lead_visible, set_speed, stopping):
  commands = []

  scc11_values = {
    "MainMode_ACC": 1,
    "TauGapSet": 4,
    "VSetDis": set_speed if enabled else 0,
    "AliveCounterACC": idx % 0x10,
  }
  commands.append(packer.make_can_msg("SCC11", 0, scc11_values))

  scc12_values = {
    "ACCMode": 1 if enabled else 0,
    "StopReq": 1 if enabled and stopping else 0,
    "aReqRaw": accel if enabled else 0,
    "aReqValue": accel if enabled else 0, # stock ramps up and down respecting jerk limit until it reaches aReqRaw
    "CR_VSM_Alive": idx % 0xF,
  }
  scc12_dat = packer.make_can_msg("SCC12", 0, scc12_values)[2]
  scc12_values["CR_VSM_ChkSum"] = 0x10 - sum([sum(divmod(i, 16)) for i in scc12_dat]) % 0x10

  commands.append(packer.make_can_msg("SCC12", 0, scc12_values))

  scc14_values = {
    "ComfortBandUpper": 0.0, # stock usually is 0 but sometimes uses higher values
    "ComfortBandLower": 0.0, # stock usually is 0 but sometimes uses higher values
    "JerkUpperLimit": 12.7 if enabled else 0, # stock usually is 1.0 but sometimes uses higher values
    "JerkLowerLimit": 12.7 if enabled else 0, # stock usually is 0.5 but sometimes uses higher values
    "ACCMode": 1 if enabled else 4, # stock will always be 4 instead of 0 after first disengage
    "ObjGap": 3 if lead_visible else 0, # TODO: 1-5 based on distance to lead vehicle or 0 if no lead
  }
  commands.append(packer.make_can_msg("SCC14", 0, scc14_values))

  fca11_values = {
    # seems to count 2,1,0,3,2,1,0,3,2,1,0,3,2,1,0,repeat...
    # (where first value is aligned to Supplemental_Counter == 0)
    # test: [(idx % 0xF, -((idx % 0xF) + 2) % 4) for idx in range(0x14)]
    "CR_FCA_Alive": ((-((idx % 0xF) + 2) % 4) << 2) + 1,
    "Supplemental_Counter": idx % 0xF,
  }
  fca11_dat = packer.make_can_msg("FCA11", 0, fca11_values)[2]
  fca11_values["CR_FCA_ChkSum"] = 0x10 - sum([sum(divmod(i, 16)) for i in fca11_dat]) % 0x10
  commands.append(packer.make_can_msg("FCA11", 0, fca11_values))

  return commands

def create_acc_opt(packer):
  commands = []

  scc13_values = {
    "SCCDrvModeRValue": 2,
    "SCC_Equip": 1,
    "Lead_Veh_Dep_Alert_USM": 2,
  }
  commands.append(packer.make_can_msg("SCC13", 0, scc13_values))

  fca12_values = {
    # stock values may be needed if openpilot has vision based AEB some day
    # for now we are not setting these because there is no AEB for vision only
    # "FCA_USM": 3,
    # "FCA_DrvSetState": 2,
  }
  commands.append(packer.make_can_msg("FCA12", 0, fca12_values))

  return commands

def create_frt_radar_opt(packer):
  frt_radar11_values = {
    "CF_FCA_Equip_Front_Radar": 1,
  }
  return packer.make_can_msg("FRT_RADAR11", 0, frt_radar11_values)
