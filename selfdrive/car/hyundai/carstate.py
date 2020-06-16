from cereal import car
from selfdrive.car.hyundai.values import DBC, STEER_THRESHOLD, FEATURES, EV_HYBRID
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.config import Conversions as CV

GearShifter = car.CarState.GearShifter

def get_can_parser(CP):

  signals = [
    # sig_name, sig_address, default
    ("WHL_SPD_FL", "WHL_SPD11", 0),
    ("WHL_SPD_FR", "WHL_SPD11", 0),
    ("WHL_SPD_RL", "WHL_SPD11", 0),
    ("WHL_SPD_RR", "WHL_SPD11", 0),

    ("YAW_RATE", "ESP12", 0),

    ("CF_Gway_DrvSeatBeltInd", "CGW4", 1),

    ("CF_Gway_DrvSeatBeltSw", "CGW1", 0), # Driver Seatbelt
    ("CF_Gway_DrvDrSw", "CGW1", 0),       # Driver Door is open
    ("CF_Gway_AstDrSw", "CGW1", 0),       # Passenger door is open
    ("CF_Gway_RLDrSw", "CGW2", 0),        # Rear reft door is open
    ("CF_Gway_RRDrSw", "CGW2", 0),        # Rear right door is open
    ("CF_Gway_TSigLHSw", "CGW1", 0),
    ("CF_Gway_TurnSigLh", "CGW1", 0),
    ("CF_Gway_TSigRHSw", "CGW1", 0),
    ("CF_Gway_TurnSigRh", "CGW1", 0),
    ("CF_Gway_ParkBrakeSw", "CGW1", 0),   # Parking Brake



    ("CYL_PRES", "ESP12", 0),

    ("CF_Clu_CruiseSwState", "CLU11", 0),
    ("CF_Clu_CruiseSwMain", "CLU11", 0),
    ("CF_Clu_SldMainSW", "CLU11", 0),
    ("CF_Clu_ParityBit1", "CLU11", 0),
    ("CF_Clu_VanzDecimal" , "CLU11", 0),
    ("CF_Clu_Vanz", "CLU11", 0),
    ("CF_Clu_SPEED_UNIT", "CLU11", 0),
    ("CF_Clu_DetentOut", "CLU11", 0),
    ("CF_Clu_RheostatLevel", "CLU11", 0),
    ("CF_Clu_CluInfo", "CLU11", 0),
    ("CF_Clu_AmpInfo", "CLU11", 0),
    ("CF_Clu_AliveCnt1", "CLU11", 0),

    ("ACCEnable", "TCS13", 0),
    ("ACC_REQ", "TCS13", 0),
    ("BrakeLight", "TCS13", 0),
    ("DriverBraking", "TCS13", 0),
    ("DriverOverride", "TCS13", 0),

    ("ESC_Off_Step", "TCS15", 0),

    ("CF_Lvr_GearInf", "LVR11", 0),        #Transmission Gear (0 = N or P, 1-8 = Fwd, 14 = Rev)

    ("CF_Lca_Stat", "LCA11", 0),
    ("CF_Lca_IndLeft", "LCA11", 0),
    ("CF_Lca_IndRight", "LCA11", 0),
  ]

  checks = [
    # address, frequency
    ("TCS15", 10),
    ("CLU11", 50),
    ("ESP12", 100),
    ("CGW1", 10),
    ("CGW4", 5),
    ("WHL_SPD11", 50),
  ]
  if not CP.mdpsBus:
    signals += [
      ("CR_Mdps_StrColTq", "MDPS12", 0),
      ("CF_Mdps_Def", "MDPS12", 0),
      ("CF_Mdps_ToiActive", "MDPS12", 0),
      ("CF_Mdps_ToiUnavail", "MDPS12", 0),
      ("CF_Mdps_MsgCount2", "MDPS12", 0),
      ("CF_Mdps_Chksum2", "MDPS12", 0),
      ("CF_Mdps_ToiFlt", "MDPS12", 0),
      ("CF_Mdps_SErr", "MDPS12", 0),
      ("CR_Mdps_StrTq", "MDPS12", 0),
      ("CF_Mdps_FailStat", "MDPS12", 0),
      ("CR_Mdps_OutTq", "MDPS12", 0)
    ]
    checks += [
      ("MDPS12", 50)
    ]
  if not CP.sasBus:
    signals += [
      ("SAS_Angle", "SAS11", 0),
      ("SAS_Speed", "SAS11", 0),
    ]
    checks += [
      ("SAS11", 100)
    ]
  if CP.sccBus == -1:
    signals += [
      ("CRUISE_LAMP_M", "EMS16", 0),
      ("CF_Lvr_CruiseSet", "LVR12", 0),
    ]
  elif not CP.sccBus:
    signals += [
      ("MainMode_ACC", "SCC11", 0),
      ("VSetDis", "SCC11", 0),
      ("SCCInfoDisplay", "SCC11", 0),
      ("ACC_ObjDist", "SCC11", 0),
      ("TauGapSet", "SCC11", 0),

      ("ACCMode", "SCC12", 0),
      ("CF_VSM_Prefill", "SCC12", 0),
      ("CF_VSM_DecCmdAct", "SCC12", 0),
      ("CF_VSM_HBACmd", "SCC12", 0),
      ("CF_VSM_Warn", "SCC12", 0),
      ("CF_VSM_Stat", "SCC12", 0),
      ("CF_VSM_BeltCmd", "SCC12", 0),
      ("ACCFailInfo", "SCC12", 0),
      ("ACCMode", "SCC12", 0),
      ("StopReq", "SCC12", 0),
      ("CR_VSM_DecCmd", "SCC12", 0),
      ("aReqMax", "SCC12", 0),
      ("TakeOverReq", "SCC12", 0),
      ("PreFill", "SCC12", 0),
      ("aReqMin", "SCC12", 0),
      ("CF_VSM_ConfMode", "SCC12", 0),
      ("AEB_Failinfo", "SCC12", 0),
      ("AEB_Status", "SCC12", 0),
      ("AEB_CmdAct", "SCC12", 0),
      ("AEB_StopReq", "SCC12", 0),
      ("CR_VSM_Alive", "SCC12", 0),
      ("CR_VSM_ChkSum", "SCC12", 0),
    ]
    checks += [
      ("SCC11", 50),
      ("SCC12", 50),
    ]
  if CP.carFingerprint in EV_HYBRID:
    signals += [
      ("Accel_Pedal_Pos", "E_EMS11", 0),
    ]
    checks += [
      ("E_EMS11", 50),
    ]
  else:
    signals += [
      ("BRAKE_ACT", "EMS12", 0),
      ("PV_AV_CAN", "EMS12", 0),
      ("TPS", "EMS12", 0),
    ]
    checks += [
      ("EMS12", 100),
    ]
  if CP.carFingerprint in FEATURES["use_cluster_gears"]:
    signals += [
      ("CF_Clu_InhibitD", "CLU15", 0),
      ("CF_Clu_InhibitP", "CLU15", 0),
      ("CF_Clu_InhibitN", "CLU15", 0),
      ("CF_Clu_InhibitR", "CLU15", 0),
    ]
  elif CP.carFingerprint in FEATURES["use_tcu_gears"]:
    signals += [
      ("CUR_GR", "TCU12",0),
    ]
  elif CP.carFingerprint in FEATURES["use_elect_gears"]:
    signals += [
      ("Elect_Gear_Shifter", "ELECT_GEAR", 0),
	]
  else:
    signals += [
      ("CF_Lvr_Gear","LVR12",0),
    ]
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

def get_can2_parser(CP):
  signals = []
  checks = []
  if CP.mdpsBus == 1:
    signals += [
      ("CR_Mdps_StrColTq", "MDPS12", 0),
      ("CF_Mdps_Def", "MDPS12", 0),
      ("CF_Mdps_ToiActive", "MDPS12", 0),
      ("CF_Mdps_ToiUnavail", "MDPS12", 0),
      ("CF_Mdps_MsgCount2", "MDPS12", 0),
      ("CF_Mdps_Chksum2", "MDPS12", 0),
      ("CF_Mdps_ToiFlt", "MDPS12", 0),
      ("CF_Mdps_SErr", "MDPS12", 0),
      ("CR_Mdps_StrTq", "MDPS12", 0),
      ("CF_Mdps_FailStat", "MDPS12", 0),
      ("CR_Mdps_OutTq", "MDPS12", 0)
    ]
    checks += [
      ("MDPS12", 50)
    ]
  if CP.sasBus == 1:
    signals += [
      ("SAS_Angle", "SAS11", 0),
      ("SAS_Speed", "SAS11", 0),
    ]
    checks += [
      ("SAS11", 100)
    ]
  if CP.sccBus == 1:
    signals += [
      ("MainMode_ACC", "SCC11", 0),
      ("VSetDis", "SCC11", 0),
      ("SCCInfoDisplay", "SCC11", 0),
      ("ACC_ObjDist", "SCC11", 0),
      ("TauGapSet", "SCC11", 0),

      ("ACCMode", "SCC12", 0),
      ("CF_VSM_Prefill", "SCC12", 0),
      ("CF_VSM_DecCmdAct", "SCC12", 0),
      ("CF_VSM_HBACmd", "SCC12", 0),
      ("CF_VSM_Warn", "SCC12", 0),
      ("CF_VSM_Stat", "SCC12", 0),
      ("CF_VSM_BeltCmd", "SCC12", 0),
      ("ACCFailInfo", "SCC12", 0),
      ("ACCMode", "SCC12", 0),
      ("StopReq", "SCC12", 0),
      ("CR_VSM_DecCmd", "SCC12", 0),
      ("aReqMax", "SCC12", 0),
      ("TakeOverReq", "SCC12", 0),
      ("PreFill", "SCC12", 0),
      ("aReqMin", "SCC12", 0),
      ("CF_VSM_ConfMode", "SCC12", 0),
      ("AEB_Failinfo", "SCC12", 0),
      ("AEB_Status", "SCC12", 0),
      ("AEB_CmdAct", "SCC12", 0),
      ("AEB_StopReq", "SCC12", 0),
      ("CR_VSM_Alive", "SCC12", 0),
      ("CR_VSM_ChkSum", "SCC12", 0)
    ]
    checks += [
      ("SCC11", 50),
      ("SCC12", 50),
    ]
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 1)

def get_camera_parser(CP):

  signals = [
    # sig_name, sig_address, default
    ("CF_Lkas_Bca_R", "LKAS11", 0),
    ("CF_Lkas_LdwsSysState", "LKAS11", 0),
    ("CF_Lkas_SysWarning", "LKAS11", 0),
    ("CF_Lkas_LdwsLHWarning", "LKAS11", 0),
    ("CF_Lkas_LdwsRHWarning", "LKAS11", 0),
    ("CF_Lkas_HbaLamp", "LKAS11", 0),
    ("CF_Lkas_FcwBasReq", "LKAS11", 0),
    ("CF_Lkas_ToiFlt", "LKAS11", 0),
    ("CF_Lkas_HbaSysState", "LKAS11", 0),
    ("CF_Lkas_FcwOpt", "LKAS11", 0),
    ("CF_Lkas_HbaOpt", "LKAS11", 0),
    ("CF_Lkas_FcwSysState", "LKAS11", 0),
    ("CF_Lkas_FcwCollisionWarning", "LKAS11", 0),
    ("CF_Lkas_MsgCount", "LKAS11", 0),
    ("CF_Lkas_FusionState", "LKAS11", 0),
    ("CF_Lkas_FcwOpt_USM", "LKAS11", 0),
    ("CF_Lkas_LdwsOpt_USM", "LKAS11", 0)
  ]

  checks = []
  if CP.sccBus == 2:
    signals += [
      ("MainMode_ACC", "SCC11", 0),
      ("VSetDis", "SCC11", 0),
      ("SCCInfoDisplay", "SCC11", 0),
      ("ACC_ObjDist", "SCC11", 0),
      ("TauGapSet", "SCC11", 0),

      ("ACCMode", "SCC12", 0),
      ("CF_VSM_Prefill", "SCC12", 0),
      ("CF_VSM_DecCmdAct", "SCC12", 0),
      ("CF_VSM_HBACmd", "SCC12", 0),
      ("CF_VSM_Warn", "SCC12", 0),
      ("CF_VSM_Stat", "SCC12", 0),
      ("CF_VSM_BeltCmd", "SCC12", 0),
      ("ACCFailInfo", "SCC12", 0),
      ("ACCMode", "SCC12", 0),
      ("StopReq", "SCC12", 0),
      ("CR_VSM_DecCmd", "SCC12", 0),
      ("aReqMax", "SCC12", 0),
      ("TakeOverReq", "SCC12", 0),
      ("PreFill", "SCC12", 0),
      ("aReqMin", "SCC12", 0),
      ("CF_VSM_ConfMode", "SCC12", 0),
      ("AEB_Failinfo", "SCC12", 0),
      ("AEB_Status", "SCC12", 0),
      ("AEB_CmdAct", "SCC12", 0),
      ("AEB_StopReq", "SCC12", 0),
      ("CR_VSM_Alive", "SCC12", 0),
      ("CR_VSM_ChkSum", "SCC12", 0),
    ]
    checks += [
      ("SCC11", 50),
      ("SCC12", 50),
    ]
  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.no_radar = CP.sccBus == -1
    self.mdps_bus = CP.mdpsBus
    self.sas_bus = CP.sasBus
    self.scc_bus = CP.sccBus

  def update(self, cp, cp2, cp_cam):
    cp_mdps = cp2 if self.mdps_bus else cp
    cp_sas = cp2 if self.sas_bus else cp
    cp_scc = cp2 if self.scc_bus == 1 else cp_cam if self.scc_bus == 2 else cp

    # update prevs, update must run once per Loop
    self.prev_left_blinker_on = self.left_blinker_on
    self.prev_right_blinker_on = self.right_blinker_on
    
    self.door_all_closed = not any([cp.vl["CGW1"]['CF_Gway_DrvDrSw'],cp.vl["CGW1"]['CF_Gway_AstDrSw'],
                                   cp.vl["CGW2"]['CF_Gway_RLDrSw'], cp.vl["CGW2"]['CF_Gway_RRDrSw']])
    self.seatbelt = cp.vl["CGW1"]['CF_Gway_DrvSeatBeltSw']

    self.brake_pressed = cp.vl["TCS13"]['DriverBraking']
    self.esp_disabled = cp.vl["TCS15"]['ESC_Off_Step']
    self.park_brake = cp.vl["CGW1"]['CF_Gway_ParkBrakeSw']

    self.main_on = (cp_scc.vl["SCC11"]["MainMode_ACC"] != 0) if not self.no_radar else \
                                            cp.vl['EMS16']['CRUISE_LAMP_M']
    self.acc_active = (cp_scc.vl["SCC12"]['ACCMode'] != 0) if not self.no_radar else \
                                      (cp.vl["LVR12"]['CF_Lvr_CruiseSet'] != 0)
    self.pcm_acc_status = int(self.acc_active)

    self.v_wheel_fl = cp.vl["WHL_SPD11"]['WHL_SPD_FL'] * CV.KPH_TO_MS
    self.v_wheel_fr = cp.vl["WHL_SPD11"]['WHL_SPD_FR'] * CV.KPH_TO_MS
    self.v_wheel_rl = cp.vl["WHL_SPD11"]['WHL_SPD_RL'] * CV.KPH_TO_MS
    self.v_wheel_rr = cp.vl["WHL_SPD11"]['WHL_SPD_RR'] * CV.KPH_TO_MS
    self.v_ego_raw = (self.v_wheel_fl + self.v_wheel_fr + self.v_wheel_rl + self.v_wheel_rr) / 4.
    self.v_ego, self.a_ego = self.update_speed_kf(self.v_ego_raw)

    self.low_speed_lockout = self.v_ego_raw < 1.0

    self.is_set_speed_in_mph = int(cp.vl["CLU11"]["CF_Clu_SPEED_UNIT"])
    speed_conv = CV.MPH_TO_MS if self.is_set_speed_in_mph else CV.KPH_TO_MS
    self.cruise_set_speed = cp_scc.vl["SCC11"]['VSetDis'] * speed_conv if not self.no_radar else \
                                         (cp.vl["LVR12"]["CF_Lvr_CruiseSet"] * speed_conv)
    self.standstill = not self.v_ego_raw > 0.1

    self.angle_steers = cp_sas.vl["SAS11"]['SAS_Angle']
    self.angle_steers_rate = cp_sas.vl["SAS11"]['SAS_Speed']
    self.yaw_rate = cp.vl["ESP12"]['YAW_RATE']
    self.left_blinker_on = cp.vl["CGW1"]['CF_Gway_TSigLHSw']
    self.right_blinker_on = cp.vl["CGW1"]['CF_Gway_TSigRHSw']
    self.left_blinker_flash = cp.vl["CGW1"]['CF_Gway_TurnSigLh']
    self.right_blinker_flash = cp.vl["CGW1"]['CF_Gway_TurnSigRh']
    self.steer_override = abs(cp_mdps.vl["MDPS12"]['CR_Mdps_StrColTq']) > STEER_THRESHOLD
    self.steer_state = cp_mdps.vl["MDPS12"]['CF_Mdps_ToiActive'] #0 NOT ACTIVE, 1 ACTIVE
    self.steer_error = cp_mdps.vl["MDPS12"]['CF_Mdps_ToiUnavail']
    self.brake_error = 0
    self.steer_torque_driver = cp_mdps.vl["MDPS12"]['CR_Mdps_StrColTq']
    self.steer_torque_motor = cp_mdps.vl["MDPS12"]['CR_Mdps_OutTq']
    self.stopped = cp_scc.vl["SCC11"]['SCCInfoDisplay'] == 4. if not self.no_radar else False
    self.lead_distance = cp_scc.vl["SCC11"]['ACC_ObjDist'] if not self.no_radar else 0

    self.user_brake = 0

    self.brake_pressed = cp.vl["TCS13"]['DriverBraking']
    self.brake_lights = bool(cp.vl["TCS13"]['BrakeLight'] or self.brake_pressed)
    if self.CP.carFingerprint in EV_HYBRID:
      gas_press_amt = cp.vl["E_EMS11"]['Accel_Pedal_Pos'] / 100
      if cp.vl["E_EMS11"]['Accel_Pedal_Pos'] > 5:
        self.pedal_gas = gas_press_amt
      else:
        self.pedal_gas = 0
    else:
      if cp.vl["TCS13"]["DriverOverride"] == 0 and cp.vl["TCS13"]['ACC_REQ'] == 1:
        self.pedal_gas = 0
      else:
        self.pedal_gas = cp.vl["EMS12"]['TPS']
    if self.CP.carFingerprint in EV_HYBRID:
      self.car_gas = cp.vl["E_EMS11"]['Accel_Pedal_Pos'] / 100
    else:
      self.car_gas = cp.vl["EMS12"]['TPS']

    # Gear Selection via Cluster - For those Kia/Hyundai which are not fully discovered, we can use the Cluster Indicator for Gear Selection, as this seems to be standard over all cars, but is not the preferred method.
    if self.car_fingerprint in FEATURES["use_cluster_gears"]:
      if cp.vl["CLU15"]["CF_Clu_InhibitD"] == 1:
        self.gear_shifter = GearShifter.drive
      elif cp.vl["CLU15"]["CF_Clu_InhibitN"] == 1:
        self.gear_shifter = GearShifter.neutral
      elif cp.vl["CLU15"]["CF_Clu_InhibitP"] == 1:
        self.gear_shifter = GearShifter.park
      elif cp.vl["CLU15"]["CF_Clu_InhibitR"] == 1:
        self.gear_shifter = GearShifter.reverse
      else:
        self.gear_shifter = GearShifter.unknown
    # Gear Selecton via TCU12
    elif self.car_fingerprint in FEATURES["use_tcu_gears"]:
      gear = cp.vl["TCU12"]["CUR_GR"]
      if gear == 0:
        self.gear_shifter = GearShifter.park
      elif gear == 14:
        self.gear_shifter = GearShifter.reverse
      elif gear > 0 and gear < 9:    # unaware of anything over 8 currently
        self.gear_shifter = GearShifter.drive
      else:
        self.gear_shifter = GearShifter.unknown
    # Gear Selecton - This is only compatible with optima hybrid 2017
    elif self.car_fingerprint in FEATURES["use_elect_gears"]:
      gear = cp.vl["ELECT_GEAR"]["Elect_Gear_Shifter"]
      if gear in (5, 8): # 5: D, 8: sport mode
        self.gear_shifter = GearShifter.drive
      elif gear == 6:
        self.gear_shifter = GearShifter.neutral
      elif gear == 0:
        self.gear_shifter = GearShifter.park
      elif gear == 7:
        self.gear_shifter = GearShifter.reverse
      else:
        self.gear_shifter = GearShifter.unknown
    # Gear Selecton - This is not compatible with all Kia/Hyundai's, But is the best way for those it is compatible with
    else:
      gear = cp.vl["LVR12"]["CF_Lvr_Gear"]
      if gear in (5, 8): # 5: D, 8: sport mode
        self.gear_shifter = GearShifter.drive
      elif gear == 6:
        self.gear_shifter = GearShifter.neutral
      elif gear == 0:
        self.gear_shifter = GearShifter.park
      elif gear == 7:
        self.gear_shifter = GearShifter.reverse
      else:
        self.gear_shifter = GearShifter.unknown

    self.lkas_button_on = 7 > cp_cam.vl["LKAS11"]["CF_Lkas_LdwsSysState"] != 0
    self.lkas_error = cp_cam.vl["LKAS11"]["CF_Lkas_LdwsSysState"] == 7

    # Blind Spot Detection and Lane Change Assist signals
    self.lca_state = cp.vl["LCA11"]["CF_Lca_Stat"]
    self.lca_left = cp.vl["LCA11"]["CF_Lca_IndLeft"]
    self.lca_right = cp.vl["LCA11"]["CF_Lca_IndRight"]

    # save the entire LKAS11, CLU11, SCC12 and MDPS12
    self.lkas11 = cp_cam.vl["LKAS11"]
    self.clu11 = cp.vl["CLU11"]
    self.scc12 = cp_scc.vl["SCC12"]
    self.mdps12 = cp_mdps.vl["MDPS12"]
