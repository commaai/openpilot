from cereal import car
from common.conversions import Conversions as CV
from opendbc.can.can_define import CANDefine
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.ford.fordcan import CanBus
from selfdrive.car.ford.values import DBC, CarControllerParams

GearShifter = car.CarState.GearShifter
TransmissionType = car.CarParams.TransmissionType


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    if CP.transmissionType == TransmissionType.automatic:
      self.shifter_values = can_define.dv["Gear_Shift_by_Wire_FD1"]["TrnRng_D_RqGsm"]

    self.vehicle_sensors_valid = False
    self.hybrid_platform = False

  def update(self, cp, cp_cam):
    ret = car.CarState.new_message()

    # Hybrid variants experience a bug where a message from the PCM sends invalid checksums,
    # we do not support these cars at this time.
    # TrnAin_Tq_Actl and its quality flag are only set on ICE platform variants
    self.hybrid_platform = cp.vl["VehicleOperatingModes"]["TrnAinTq_D_Qf"] == 0

    # Occasionally on startup, the ABS module recalibrates the steering pinion offset, so we need to block engagement
    # The vehicle usually recovers out of this state within a minute of normal driving
    self.vehicle_sensors_valid = cp.vl["SteeringPinion_Data"]["StePinCompAnEst_D_Qf"] == 3

    # car speed
    ret.vEgoRaw = cp.vl["BrakeSysFeatures"]["Veh_V_ActlBrk"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.yawRate = cp.vl["Yaw_Data_FD1"]["VehYaw_W_Actl"]
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
    ret.steeringPressed = self.update_steering_pressed(abs(ret.steeringTorque) > CarControllerParams.STEER_DRIVER_ALLOWANCE, 5)
    ret.steerFaultTemporary = cp.vl["EPAS_INFO"]["EPAS_Failure"] == 1
    ret.steerFaultPermanent = cp.vl["EPAS_INFO"]["EPAS_Failure"] in (2, 3)
    # ret.espDisabled = False  # TODO: find traction control signal

    # cruise state
    ret.cruiseState.speed = cp.vl["EngBrakeData"]["Veh_V_DsplyCcSet"] * CV.MPH_TO_MS
    ret.cruiseState.enabled = cp.vl["EngBrakeData"]["CcStat_D_Actl"] in (4, 5)
    ret.cruiseState.available = cp.vl["EngBrakeData"]["CcStat_D_Actl"] in (3, 4, 5)
    ret.cruiseState.nonAdaptive = cp.vl["Cluster_Info1_FD1"]["AccEnbl_B_RqDrv"] == 0
    ret.cruiseState.standstill = cp.vl["EngBrakeData"]["AccStopMde_D_Rq"] == 3
    ret.accFaulted = cp.vl["EngBrakeData"]["CcStat_D_Actl"] in (1, 2)

    # gear
    if self.CP.transmissionType == TransmissionType.automatic:
      gear = self.shifter_values.get(cp.vl["Gear_Shift_by_Wire_FD1"]["TrnRng_D_RqGsm"])
      ret.gearShifter = self.parse_gear_shifter(gear)
    elif self.CP.transmissionType == TransmissionType.manual:
      ret.clutchPressed = cp.vl["Engine_Clutch_Data"]["CluPdlPos_Pc_Meas"] > 0
      if bool(cp.vl["BCM_Lamp_Stat_FD1"]["RvrseLghtOn_B_Stat"]):
        ret.gearShifter = GearShifter.reverse
      else:
        ret.gearShifter = GearShifter.drive

    # safety
    ret.stockFcw = bool(cp_cam.vl["ACCDATA_3"]["FcwVisblWarn_B_Rq"])
    ret.stockAeb = bool(cp_cam.vl["ACCDATA_2"]["CmbbBrkDecel_B_Rq"])

    # button presses
    ret.leftBlinker = cp.vl["Steering_Data_FD1"]["TurnLghtSwtch_D_Stat"] == 1
    ret.rightBlinker = cp.vl["Steering_Data_FD1"]["TurnLghtSwtch_D_Stat"] == 2
    # TODO: block this going to the camera otherwise it will enable stock TJA
    ret.genericToggle = bool(cp.vl["Steering_Data_FD1"]["TjaButtnOnOffPress"])

    # lock info
    ret.doorOpen = any([cp.vl["BodyInfo_3_FD1"]["DrStatDrv_B_Actl"], cp.vl["BodyInfo_3_FD1"]["DrStatPsngr_B_Actl"],
                        cp.vl["BodyInfo_3_FD1"]["DrStatRl_B_Actl"], cp.vl["BodyInfo_3_FD1"]["DrStatRr_B_Actl"]])
    ret.seatbeltUnlatched = cp.vl["RCMStatusMessage2_FD1"]["FirstRowBuckleDriver"] == 2

    # blindspot sensors
    if self.CP.enableBsm:
      ret.leftBlindspot = cp.vl["Side_Detect_L_Stat"]["SodDetctLeft_D_Stat"] != 0
      ret.rightBlindspot = cp.vl["Side_Detect_R_Stat"]["SodDetctRight_D_Stat"] != 0

    # Stock steering buttons so that we can passthru blinkers etc.
    self.buttons_stock_values = cp.vl["Steering_Data_FD1"]
    # Stock values from IPMA so that we can retain some stock functionality
    self.acc_tja_status_stock_values = cp_cam.vl["ACCDATA_3"]
    self.lkas_status_stock_values = cp_cam.vl["IPMA_Data"]

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("TrnAinTq_D_Qf", "VehicleOperatingModes"),            # Used to detect hybrid or ICE platform variant

      ("Veh_V_ActlBrk", "BrakeSysFeatures"),                 # ABS vehicle speed (kph)
      ("VehYaw_W_Actl", "Yaw_Data_FD1"),                     # ABS vehicle yaw rate (rad/s)
      ("VehStop_D_Stat", "DesiredTorqBrk"),                  # ABS vehicle stopped
      ("PrkBrkStatus", "DesiredTorqBrk"),                    # ABS park brake status
      ("ApedPos_Pc_ActlArb", "EngVehicleSpThrottle"),        # PCM throttle (pct)
      ("BrkTot_Tq_Actl", "BrakeSnData_4"),                   # ABS brake torque (Nm)
      ("BpedDrvAppl_D_Actl", "EngBrakeData"),                # PCM driver brake pedal pressed
      ("Veh_V_DsplyCcSet", "EngBrakeData"),                  # PCM ACC set speed (mph)
                                                             # The units might change with IPC settings?
      ("CcStat_D_Actl", "EngBrakeData"),                     # PCM ACC status
      ("AccStopMde_D_Rq", "EngBrakeData"),                   # PCM ACC standstill
      ("AccEnbl_B_RqDrv", "Cluster_Info1_FD1"),              # PCM ACC enable
      ("StePinComp_An_Est", "SteeringPinion_Data"),          # PSCM estimated steering angle (deg)
      ("StePinCompAnEst_D_Qf", "SteeringPinion_Data"),       # PSCM estimated steering angle (quality flag)
                                                             # Calculates steering angle (and offset) from pinion
                                                             # angle and driving measurements.
                                                             # StePinRelInit_An_Sns is the pinion angle, initialised
                                                             # to zero at the beginning of the drive.
      ("SteeringColumnTorque", "EPAS_INFO"),                 # PSCM steering column torque (Nm)
      ("EPAS_Failure", "EPAS_INFO"),                         # PSCM EPAS status
      ("TurnLghtSwtch_D_Stat", "Steering_Data_FD1"),         # SCCM Turn signal switch
      ("TjaButtnOnOffPress", "Steering_Data_FD1"),           # SCCM ACC button, lane-centering/traffic jam assist toggle
      ("DrStatDrv_B_Actl", "BodyInfo_3_FD1"),                # BCM Door open, driver
      ("DrStatPsngr_B_Actl", "BodyInfo_3_FD1"),              # BCM Door open, passenger
      ("DrStatRl_B_Actl", "BodyInfo_3_FD1"),                 # BCM Door open, rear left
      ("DrStatRr_B_Actl", "BodyInfo_3_FD1"),                 # BCM Door open, rear right
      ("FirstRowBuckleDriver", "RCMStatusMessage2_FD1"),     # RCM Seatbelt status, driver
      ("HeadLghtHiFlash_D_Stat", "Steering_Data_FD1"),       # SCCM Passthrough the remaining buttons
      ("WiprFront_D_Stat", "Steering_Data_FD1"),
      ("LghtAmb_D_Sns", "Steering_Data_FD1"),
      ("AccButtnGapDecPress", "Steering_Data_FD1"),
      ("AccButtnGapIncPress", "Steering_Data_FD1"),
      ("AslButtnOnOffCnclPress", "Steering_Data_FD1"),
      ("AslButtnOnOffPress", "Steering_Data_FD1"),
      ("LaSwtchPos_D_Stat", "Steering_Data_FD1"),
      ("CcAslButtnCnclResPress", "Steering_Data_FD1"),
      ("CcAslButtnDeny_B_Actl", "Steering_Data_FD1"),
      ("CcAslButtnIndxDecPress", "Steering_Data_FD1"),
      ("CcAslButtnIndxIncPress", "Steering_Data_FD1"),
      ("CcAslButtnOffCnclPress", "Steering_Data_FD1"),
      ("CcAslButtnOnOffCncl", "Steering_Data_FD1"),
      ("CcAslButtnOnPress", "Steering_Data_FD1"),
      ("CcAslButtnResDecPress", "Steering_Data_FD1"),
      ("CcAslButtnResIncPress", "Steering_Data_FD1"),
      ("CcAslButtnSetDecPress", "Steering_Data_FD1"),
      ("CcAslButtnSetIncPress", "Steering_Data_FD1"),
      ("CcAslButtnSetPress", "Steering_Data_FD1"),
      ("CcButtnOffPress", "Steering_Data_FD1"),
      ("CcButtnOnOffCnclPress", "Steering_Data_FD1"),
      ("CcButtnOnOffPress", "Steering_Data_FD1"),
      ("CcButtnOnPress", "Steering_Data_FD1"),
      ("HeadLghtHiFlash_D_Actl", "Steering_Data_FD1"),
      ("HeadLghtHiOn_B_StatAhb", "Steering_Data_FD1"),
      ("AhbStat_B_Dsply", "Steering_Data_FD1"),
      ("AccButtnGapTogglePress", "Steering_Data_FD1"),
      ("WiprFrontSwtch_D_Stat", "Steering_Data_FD1"),
      ("HeadLghtHiCtrl_D_RqAhb", "Steering_Data_FD1"),
    ]

    checks = [
      # sig_address, frequency
      ("VehicleOperatingModes", 100),
      ("BrakeSysFeatures", 50),
      ("Yaw_Data_FD1", 100),
      ("DesiredTorqBrk", 50),
      ("EngVehicleSpThrottle", 100),
      ("BrakeSnData_4", 50),
      ("EngBrakeData", 10),
      ("Cluster_Info1_FD1", 10),
      ("SteeringPinion_Data", 100),
      ("EPAS_INFO", 50),
      ("Lane_Assist_Data3_FD1", 33),
      ("Steering_Data_FD1", 10),
      ("BodyInfo_3_FD1", 2),
      ("RCMStatusMessage2_FD1", 10),
    ]

    if CP.transmissionType == TransmissionType.automatic:
      signals += [
        ("TrnRng_D_RqGsm", "Gear_Shift_by_Wire_FD1"),        # GWM transmission gear position
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

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, CanBus(CP).main)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("CmbbBrkDecel_B_Rq", "ACCDATA_2"),           # AEB actuation request bit

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
      ("AccFllwMde_B_Dsply", "ACCDATA_3"),          # ACC lead indicator
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
      ("LaDenyStats_B_Dsply", "IPMA_Data"),         # LKAS error
      ("CamraDefog_B_Req", "IPMA_Data"),            # Windshield heater?
      ("CamraStats_D_Dsply", "IPMA_Data"),          # Camera status
      ("DasAlrtLvl_D_Dsply", "IPMA_Data"),          # DAS alert level
      ("DasStats_D_Dsply", "IPMA_Data"),            # DAS status
      ("DasWarn_D_Dsply", "IPMA_Data"),             # DAS warning
      ("AhbHiBeam_D_Rq", "IPMA_Data"),              # AHB status
      ("Passthru_63", "IPMA_Data"),
      ("Passthru_48", "IPMA_Data"),
    ]

    checks = [
      # sig_address, frequency
      ("ACCDATA_2", 50),
      ("ACCDATA_3", 5),
      ("IPMA_Data", 1),
    ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, CanBus(CP).camera)
