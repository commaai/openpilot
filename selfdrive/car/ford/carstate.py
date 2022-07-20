from typing import Dict

from cereal import car
from common.conversions import Conversions as CV
from opendbc.can.can_define import CANDefine
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.ford.values import CANBUS, DBC

GearShifter = car.CarState.GearShifter
TransmissionType = car.CarParams.TransmissionType


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    if CP.transmissionType == TransmissionType.automatic:
      self.shifter_values = can_define.dv["Gear_Shift_by_Wire_FD1"]["TrnGear_D_RqDrv"]

  def update(self, cp, cp_cam):
    ret = car.CarState.new_message()

    # car speed
    ret.vEgoRaw = cp.vl["EngVehicleSpThrottle2"]["Veh_V_ActlEng"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.yawRate = cp.vl["Yaw_Data_FD1"]["VehYaw_W_Actl"] * CV.RAD_TO_DEG
    ret.standstill = cp.vl["DesiredTorqBrk"]["VehStop_D_Stat"] == 1

    # gas pedal
    ret.gas = cp.vl["EngVehicleSpThrottle"]["ApedPos_Pc_ActlArb"] / 100.
    ret.gasPressed = ret.gas > 1e-6

    # brake pedal
    ret.brake = cp.vl["BrakeSnData_4"]["BrkTot_Tq_Actl"] / 32756.  # torque in Nm
    ret.brakePressed = cp.vl["EngBrakeData"]["BpedDrvAppl_D_Actl"] == 2
    ret.parkingBrake = cp.vl["DesiredTorqBrk"]["PrkBrkStatus"] in (1, 2)

    # steering wheel
    ret.steeringAngleDeg = cp.vl["SteeringPinion_Data"]["StePinComp_An_Est"]
    ret.steeringTorque = cp.vl["EPAS_INFO"]["SteeringColumnTorque"]
    ret.steeringPressed = cp.vl["Lane_Assist_Data3_FD1"]["LaHandsOff_B_Actl"] == 0
    ret.steerFaultTemporary = cp.vl["EPAS_INFO"]["EPAS_Failure"] == 1
    ret.steerFaultPermanent = cp.vl["EPAS_INFO"]["EPAS_Failure"] in (2, 3)
    # ret.espDisabled = False  # TODO: find traction control signal

    # cruise state
    ret.cruiseState.speed = cp.vl["EngBrakeData"]["Veh_V_DsplyCcSet"] * CV.MPH_TO_MS
    ret.cruiseState.enabled = cp.vl["EngBrakeData"]["CcStat_D_Actl"] in (4, 5)
    ret.cruiseState.available = cp.vl["EngBrakeData"]["CcStat_D_Actl"] in (3, 4, 5)

    # gear
    if self.CP.transmissionType == TransmissionType.automatic:
      gear = int(cp.vl["Gear_Shift_by_Wire_FD1"]["TrnGear_D_RqDrv"])
      ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(gear, None))
    elif self.CP.transmissionType == TransmissionType.manual:
      ret.clutchPressed = cp.vl["Engine_Clutch_Data"]["CluPdlPos_Pc_Meas"] > 0
      if bool(cp.vl["BCM_Lamp_Stat_FD1"]["RvrseLghtOn_B_Stat"]):
        ret.gearShifter = GearShifter.reverse
      else:
        ret.gearShifter = GearShifter.drive

    # safety
    ret.stockFcw = bool(cp_cam.vl["ACCDATA_3"]["FcwVisblWarn_B_Rq"])
    ret.stockAeb = ret.stockFcw and ret.cruiseState.enabled

    # button presses
    ret.leftBlinker = cp.vl["Steering_Data_FD1"]["TurnLghtSwtch_D_Stat"] == 1
    ret.rightBlinker = cp.vl["Steering_Data_FD1"]["TurnLghtSwtch_D_Stat"] == 2
    ret.genericToggle = bool(cp.vl["Steering_Data_FD1"]["TjaButtnOnOffPress"])

    # lock info
    ret.doorOpen = any([cp.vl["BodyInfo_3_FD1"]["DrStatDrv_B_Actl"], cp.vl["BodyInfo_3_FD1"]["DrStatPsngr_B_Actl"],
                        cp.vl["BodyInfo_3_FD1"]["DrStatRl_B_Actl"], cp.vl["BodyInfo_3_FD1"]["DrStatRr_B_Actl"]])
    ret.seatbeltUnlatched = cp.vl["RCMStatusMessage2_FD1"]["FirstRowBuckleDriver"] == 2

    # blindspot sensors
    if self.CP.enableBsm:
      ret.leftBlindspot = cp.vl["Side_Detect_L_Stat"]["SodDetctLeft_D_Stat"] != 0
      ret.rightBlindspot = cp.vl["Side_Detect_R_Stat"]["SodDetctRight_D_Stat"] != 0

    # Stock values from IPMA so that we can retain some stock functionality
    self.acc_tja_status_stock_values = cp_cam.vl["ACCDATA_3"]
    self.lkas_status_stock_values = cp_cam.vl["IPMA_Data"]

    return ret

  @staticmethod
  def parse_gear_shifter(gear: str) -> car.CarState.GearShifter:
    d: Dict[str, car.CarState.GearShifter] = {
        'Park': GearShifter.park, 'Reverse': GearShifter.reverse, 'Neutral': GearShifter.neutral,
        'Manual': GearShifter.manumatic, 'Drive': GearShifter.drive,
    }
    return d.get(gear, GearShifter.unknown)

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("Veh_V_ActlEng", "EngVehicleSpThrottle2"),            # ABS vehicle speed (kph)
      ("VehYaw_W_Actl", "Yaw_Data_FD1"),                     # ABS vehicle yaw rate (rad/s)
      ("VehStop_D_Stat", "DesiredTorqBrk"),                  # ABS vehicle stopped
      ("PrkBrkStatus", "DesiredTorqBrk"),                    # ABS park brake status
      ("ApedPos_Pc_ActlArb", "EngVehicleSpThrottle"),        # PCM throttle (pct)
      ("BrkTot_Tq_Actl", "BrakeSnData_4"),                   # ABS brake torque (Nm)
      ("BpedDrvAppl_D_Actl", "EngBrakeData"),                # PCM driver brake pedal pressed
      ("Veh_V_DsplyCcSet", "EngBrakeData"),                  # PCM ACC set speed (mph)
                                                             # The units might change with IPC settings?
      ("CcStat_D_Actl", "EngBrakeData"),                     # PCM ACC status
      ("StePinComp_An_Est", "SteeringPinion_Data"),          # PSCM estimated steering angle (deg)
                                                             # Calculates steering angle (and offset) from pinion
                                                             # angle and driving measurements.
                                                             # StePinRelInit_An_Sns is the pinion angle, initialised
                                                             # to zero at the beginning of the drive.
      ("SteeringColumnTorque", "EPAS_INFO"),                 # PSCM steering column torque (Nm)
      ("EPAS_Failure", "EPAS_INFO"),                         # PSCM EPAS status
      ("LaHandsOff_B_Actl", "Lane_Assist_Data3_FD1"),        # PSCM LKAS hands off wheel
      ("TurnLghtSwtch_D_Stat", "Steering_Data_FD1"),         # SCCM Turn signal switch
      ("TjaButtnOnOffPress", "Steering_Data_FD1"),           # SCCM ACC button, lane-centering/traffic jam assist toggle
      ("DrStatDrv_B_Actl", "BodyInfo_3_FD1"),                # BCM Door open, driver
      ("DrStatPsngr_B_Actl", "BodyInfo_3_FD1"),              # BCM Door open, passenger
      ("DrStatRl_B_Actl", "BodyInfo_3_FD1"),                 # BCM Door open, rear left
      ("DrStatRr_B_Actl", "BodyInfo_3_FD1"),                 # BCM Door open, rear right
      ("FirstRowBuckleDriver", "RCMStatusMessage2_FD1"),     # RCM Seatbelt status, driver
    ]

    checks = [
      # sig_address, frequency
      ("EngVehicleSpThrottle2", 50),
      ("Yaw_Data_FD1", 100),
      ("DesiredTorqBrk", 50),
      ("EngVehicleSpThrottle", 100),
      ("BrakeSnData_4", 50),
      ("EngBrakeData", 10),
      ("SteeringPinion_Data", 100),
      ("EPAS_INFO", 50),
      ("Lane_Assist_Data3_FD1", 33),
      ("Steering_Data_FD1", 10),
      ("BodyInfo_3_FD1", 2),
      ("RCMStatusMessage2_FD1", 10),
    ]

    if CP.transmissionType == TransmissionType.automatic:
      signals += [
        ("TrnGear_D_RqDrv", "Gear_Shift_by_Wire_FD1"),       # GWM transmission gear position
      ]
      checks += [
        ("Gear_Shift_by_Wire_FD1", 10),
      ]
    elif CP.transmissionType == TransmissionType.manual:
      signals += [
        ("CluPdlPos_Pc_Meas", "Engine_Clutch_Data"),         # PCM clutch (pct)
        ("RvrseLghtOn_B_Stat", "BCM_Lamp_Stat_FD1"),         # BCM reverse light
      ]
      checks += [
        ("Engine_Clutch_Data", 33),
        ("BCM_Lamp_Stat_FD1", 1),
      ]

    if CP.enableBsm:
      signals += [
        ("SodDetctLeft_D_Stat", "Side_Detect_L_Stat"),       # Blindspot sensor, left
        ("SodDetctRight_D_Stat", "Side_Detect_R_Stat"),      # Blindspot sensor, right
      ]
      checks += [
        ("Side_Detect_L_Stat", 5),
        ("Side_Detect_R_Stat", 5),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, CANBUS.main)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("HaDsply_No_Cs", "ACCDATA_3"),
      ("HaDsply_No_Cnt", "ACCDATA_3"),
      ("AccStopStat_D_Dsply", "ACCDATA_3"),         # ACC stopped status message
      ("AccTrgDist2_D_Dsply", "ACCDATA_3"),         # ACC target distance
      ("AccStopRes_B_Dsply", "ACCDATA_3"),
      ("TjaWarn_D_Rq", "ACCDATA_3"),                # TJA warning
      ("Tja_D_Stat", "ACCDATA_3"),                  # TJA status
      ("TjaMsgTxt_D_Dsply", "ACCDATA_3"),           # TJA text
      ("IaccLamp_D_Rq", "ACCDATA_3"),               # iACC status icon
      ("AccMsgTxt_D2_Rq", "ACCDATA_3"),             # ACC text
      ("FcwDeny_B_Dsply", "ACCDATA_3"),             # FCW disabled
      ("FcwMemStat_B_Actl", "ACCDATA_3"),           # FCW enabled setting
      ("AccTGap_B_Dsply", "ACCDATA_3"),             # ACC time gap display setting
      ("CadsAlignIncplt_B_Actl", "ACCDATA_3"),
      ("AccFllwMde_B_Dsply", "ACCDATA_3"),          # ACC follow mode display setting
      ("CadsRadrBlck_B_Actl", "ACCDATA_3"),
      ("CmbbPostEvnt_B_Dsply", "ACCDATA_3"),        # AEB event status
      ("AccStopMde_B_Dsply", "ACCDATA_3"),          # ACC stop mode display setting
      ("FcwMemSens_D_Actl", "ACCDATA_3"),           # FCW sensitivity setting
      ("FcwMsgTxt_D_Rq", "ACCDATA_3"),              # FCW text
      ("AccWarn_D_Dsply", "ACCDATA_3"),             # ACC warning
      ("FcwVisblWarn_B_Rq", "ACCDATA_3"),           # FCW visible alert
      ("FcwAudioWarn_B_Rq", "ACCDATA_3"),           # FCW audio alert
      ("AccTGap_D_Dsply", "ACCDATA_3"),             # ACC time gap
      ("AccMemEnbl_B_RqDrv", "ACCDATA_3"),          # ACC adaptive/normal setting
      ("FdaMem_B_Stat", "ACCDATA_3"),               # FDA enabled setting

      ("FeatConfigIpmaActl", "IPMA_Data"),
      ("FeatNoIpmaActl", "IPMA_Data"),
      ("PersIndexIpma_D_Actl", "IPMA_Data"),
      ("AhbcRampingV_D_Rq", "IPMA_Data"),           # AHB ramping
      ("LaActvStats_D_Dsply", "IPMA_Data"),         # LKAS status (lines)
      ("LaDenyStats_B_Dsply", "IPMA_Data"),         # LKAS error
      ("LaHandsOff_D_Dsply", "IPMA_Data"),          # LKAS hands on chime
      ("CamraDefog_B_Req", "IPMA_Data"),            # Windshield heater?
      ("CamraStats_D_Dsply", "IPMA_Data"),          # Camera status
      ("DasAlrtLvl_D_Dsply", "IPMA_Data"),          # DAS alert level
      ("DasStats_D_Dsply", "IPMA_Data"),            # DAS status
      ("DasWarn_D_Dsply", "IPMA_Data"),             # DAS warning
      ("AhbHiBeam_D_Rq", "IPMA_Data"),              # AHB status
      ("Set_Me_X1", "IPMA_Data"),
    ]

    checks = [
      # sig_address, frequency
      ("ACCDATA_3", 5),
      ("IPMA_Data", 1),
    ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, CANBUS.camera)
