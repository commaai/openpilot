from cereal import car
from selfdrive.car.hyundai.values import DBC, STEER_THRESHOLD, FEATURES
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

    ("CF_Gway_DrvSeatBeltSw", "CGW1", 0),
    ("CF_Gway_TSigLHSw", "CGW1", 0),
    ("CF_Gway_TurnSigLh", "CGW1", 0),
    ("CF_Gway_TSigRHSw", "CGW1", 0),
    ("CF_Gway_TurnSigRh", "CGW1", 0),
    ("CF_Gway_ParkBrakeSw", "CGW1", 0),

    ("BRAKE_ACT", "EMS12", 0),
    ("PV_AV_CAN", "EMS12", 0),
    ("TPS", "EMS12", 0),

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

    ("CF_Clu_InhibitD", "CLU15", 0),
    ("CF_Clu_InhibitP", "CLU15", 0),
    ("CF_Clu_InhibitN", "CLU15", 0),
    ("CF_Clu_InhibitR", "CLU15", 0),

    ("CF_Lvr_Gear", "LVR12",0),
    ("CUR_GR", "TCU12",0),

    ("ACCEnable", "TCS13", 0),
    ("ACC_REQ", "TCS13", 0),
    ("DriverBraking", "TCS13", 0),
    ("DriverOverride", "TCS13", 0),

    ("ESC_Off_Step", "TCS15", 0),

    ("CF_Lvr_GearInf", "LVR11", 0),        #Transmission Gear (0 = N or P, 1-8 = Fwd, 14 = Rev)

    ("CR_Mdps_DrvTq", "MDPS11", 0),

    ("CR_Mdps_StrColTq", "MDPS12", 0),
    ("CF_Mdps_ToiActive", "MDPS12", 0),
    ("CF_Mdps_ToiUnavail", "MDPS12", 0),
    ("CF_Mdps_FailStat", "MDPS12", 0),
    ("CR_Mdps_OutTq", "MDPS12", 0),

    ("VSetDis", "SCC11", 0),
    ("SCCInfoDisplay", "SCC11", 0),
    ("ACCMode", "SCC12", 1),

    ("SAS_Angle", "SAS11", 0),
    ("SAS_Speed", "SAS11", 0),
  ]

  checks = [
    # address, frequency
    ("MDPS12", 50),
    ("MDPS11", 100),
    ("TCS15", 10),
    ("TCS13", 50),
    ("CLU11", 50),
    ("ESP12", 100),
    ("EMS12", 100),
    ("CGW1", 10),
    ("CGW4", 5),
    ("WHL_SPD11", 50),
    ("SCC11", 50),
    ("SCC12", 50),
    ("SAS11", 100)
  ]

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)


def get_camera_parser(CP):

  signals = [
    # sig_name, sig_address, default
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
    ("CF_Lkas_FusionState", "LKAS11", 0),
    ("CF_Lkas_FcwOpt_USM", "LKAS11", 0),
    ("CF_Lkas_LdwsOpt_USM", "LKAS11", 0)
  ]

  checks = []

  return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)


class CarState(CarStateBase):

  def update(self, cp, cp_cam):
    ret = car.CarState.new_message()

    ret.doorOpen = False  # FIXME
    ret.seatbeltUnlatched = cp.vl["CGW1"]['CF_Gway_DrvSeatBeltSw'] == 0

    ret.wheelSpeeds.fl = cp.vl["WHL_SPD11"]['WHL_SPD_FL'] * CV.KPH_TO_MS
    ret.wheelSpeeds.fr = cp.vl["WHL_SPD11"]['WHL_SPD_FR'] * CV.KPH_TO_MS
    ret.wheelSpeeds.rl = cp.vl["WHL_SPD11"]['WHL_SPD_RL'] * CV.KPH_TO_MS
    ret.wheelSpeeds.rr = cp.vl["WHL_SPD11"]['WHL_SPD_RR'] * CV.KPH_TO_MS
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    ret.standstill = ret.vEgoRaw < 0.1

    ret.steeringAngle = cp.vl["SAS11"]['SAS_Angle']
    ret.steeringRate = cp.vl["SAS11"]['SAS_Speed']
    ret.yawRate = cp.vl["ESP12"]['YAW_RATE']
    ret.leftBlinker = cp.vl["CGW1"]['CF_Gway_TSigLHSw'] != 0
    ret.rightBlinker = cp.vl["CGW1"]['CF_Gway_TSigRHSw'] != 0
    ret.steeringTorque = cp.vl["MDPS11"]['CR_Mdps_DrvTq']
    ret.steeringTorqueEps = cp.vl["MDPS12"]['CR_Mdps_OutTq']
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD

    # cruise state
    ret.cruiseState.enabled = cp.vl["SCC12"]['ACCMode'] != 0
    ret.cruiseState.available = True
    ret.cruiseState.standstill = cp.vl["SCC11"]['SCCInfoDisplay'] == 4.
    if ret.cruiseState.enabled:
      is_set_speed_in_mph = int(cp.vl["CLU11"]["CF_Clu_SPEED_UNIT"])
      speed_conv = CV.MPH_TO_MS if is_set_speed_in_mph else CV.KPH_TO_MS
      ret.cruiseState.speed = cp.vl["SCC11"]['VSetDis'] * speed_conv
    else:
      ret.cruiseState.speed = 0

    ret.brake = 0  # FIXME
    ret.brakePressed = cp.vl["TCS13"]['DriverBraking'] != 0
    ret.brakeLights = ret.brakePressed
    if (cp.vl["TCS13"]["DriverOverride"] == 0 and cp.vl["TCS13"]['ACC_REQ'] == 1):
      pedal_gas = 0
    else:
      pedal_gas = cp.vl["EMS12"]['TPS']
    ret.gasPressed = pedal_gas > 1e-3
    ret.gas = cp.vl["EMS12"]['TPS']

    # Gear Selecton - This is not compatible with all Kia/Hyundai's, But is the best way for those it is compatible with
    gear = cp.vl["LVR12"]["CF_Lvr_Gear"]
    if gear == 5:
      gear_shifter = GearShifter.drive
    elif gear == 6:
      gear_shifter = GearShifter.neutral
    elif gear == 0:
      gear_shifter = GearShifter.park
    elif gear == 7:
      gear_shifter = GearShifter.reverse
    else:
      gear_shifter = GearShifter.unknown

    # Gear Selection via Cluster - For those Kia/Hyundai which are not fully discovered, we can use the Cluster Indicator for Gear Selection, as this seems to be standard over all cars, but is not the preferred method.
    if cp.vl["CLU15"]["CF_Clu_InhibitD"] == 1:
      gear_shifter_cluster = GearShifter.drive
    elif cp.vl["CLU15"]["CF_Clu_InhibitN"] == 1:
      gear_shifter_cluster = GearShifter.neutral
    elif cp.vl["CLU15"]["CF_Clu_InhibitP"] == 1:
      gear_shifter_cluster = GearShifter.park
    elif cp.vl["CLU15"]["CF_Clu_InhibitR"] == 1:
      gear_shifter_cluster = GearShifter.reverse
    else:
      gear_shifter_cluster = GearShifter.unknown

    # Gear Selecton via TCU12
    gear2 = cp.vl["TCU12"]["CUR_GR"]
    if gear2 == 0:
      gear_tcu = GearShifter.park
    elif gear2 == 14:
      gear_tcu = GearShifter.reverse
    elif gear2 > 0 and gear2 < 9:    # unaware of anything over 8 currently
      gear_tcu = GearShifter.drive
    else:
      gear_tcu = GearShifter.unknown

    # gear shifter
    if self.CP.carFingerprint in FEATURES["use_cluster_gears"]:
      ret.gearShifter = gear_shifter_cluster
    elif self.CP.carFingerprint in FEATURES["use_tcu_gears"]:
      ret.gearShifter = gear_tcu
    else:
      ret.gearShifter = gear_shifter

    # save the entire LKAS11 and CLU11
    self.lkas11 = cp_cam.vl["LKAS11"]
    self.clu11 = cp.vl["CLU11"]
    self.esp_disabled = cp.vl["TCS15"]['ESC_Off_Step']
    self.park_brake = cp.vl["CGW1"]['CF_Gway_ParkBrakeSw']
    self.steer_state = cp.vl["MDPS12"]['CF_Mdps_ToiActive'] #0 NOT ACTIVE, 1 ACTIVE
    self.steer_error = cp.vl["MDPS12"]['CF_Mdps_ToiUnavail']
    self.brake_error = 0

    return ret
