import numpy as np
from cereal import car
from common.conversions import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.car.volkswagen.values import DBC_FILES, CANBUS, NetworkLocation, TransmissionType, GearShifter, BUTTON_STATES, CarControllerParams

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC_FILES.mqb)
    if CP.transmissionType == TransmissionType.automatic:
      self.shifter_values = can_define.dv["Getriebe_11"]["GE_Fahrstufe"]
    elif CP.transmissionType == TransmissionType.direct:
      self.shifter_values = can_define.dv["EV_Gearshift"]["GearPosition"]
    self.hca_status_values = can_define.dv["LH_EPS_03"]["EPS_HCA_Status"]
    self.buttonStates = BUTTON_STATES.copy()

  def update(self, pt_cp, cam_cp, ext_cp, trans_type):
    ret = car.CarState.new_message()
    # Update vehicle speed and acceleration from ABS wheel speeds.
    ret.wheelSpeeds = self.get_wheel_speeds(
      pt_cp.vl["ESP_19"]["ESP_VL_Radgeschw_02"],
      pt_cp.vl["ESP_19"]["ESP_VR_Radgeschw_02"],
      pt_cp.vl["ESP_19"]["ESP_HL_Radgeschw_02"],
      pt_cp.vl["ESP_19"]["ESP_HR_Radgeschw_02"],
    )

    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgo < 0.1

    # Update steering angle, rate, yaw rate, and driver input torque. VW send
    # the sign/direction in a separate signal so they must be recombined.
    ret.steeringAngleDeg = pt_cp.vl["LWI_01"]["LWI_Lenkradwinkel"] * (1, -1)[int(pt_cp.vl["LWI_01"]["LWI_VZ_Lenkradwinkel"])]
    ret.steeringRateDeg = pt_cp.vl["LWI_01"]["LWI_Lenkradw_Geschw"] * (1, -1)[int(pt_cp.vl["LWI_01"]["LWI_VZ_Lenkradw_Geschw"])]
    ret.steeringTorque = pt_cp.vl["LH_EPS_03"]["EPS_Lenkmoment"] * (1, -1)[int(pt_cp.vl["LH_EPS_03"]["EPS_VZ_Lenkmoment"])]
    ret.steeringPressed = abs(ret.steeringTorque) > CarControllerParams.STEER_DRIVER_ALLOWANCE
    ret.yawRate = pt_cp.vl["ESP_02"]["ESP_Gierrate"] * (1, -1)[int(pt_cp.vl["ESP_02"]["ESP_VZ_Gierrate"])] * CV.DEG_TO_RAD

    # Verify EPS readiness to accept steering commands
    hca_status = self.hca_status_values.get(pt_cp.vl["LH_EPS_03"]["EPS_HCA_Status"])
    ret.steerFaultPermanent = hca_status in ("DISABLED", "FAULT")
    ret.steerFaultTemporary = hca_status in ("INITIALIZING", "REJECTED")

    # Update gas, brakes, and gearshift.
    ret.gas = pt_cp.vl["Motor_20"]["MO_Fahrpedalrohwert_01"] / 100.0
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["ESP_05"]["ESP_Bremsdruck"] / 250.0  # FIXME: this is pressure in Bar, not sure what OP expects
    ret.brakePressed = bool(pt_cp.vl["ESP_05"]["ESP_Fahrer_bremst"])
    ret.parkingBrake = bool(pt_cp.vl["Kombi_01"]["KBI_Handbremse"])  # FIXME: need to include an EPB check as well
    self.esp_hold_confirmation = pt_cp.vl["ESP_21"]["ESP_Haltebestaetigung"]

    # Update gear and/or clutch position data.
    if trans_type == TransmissionType.automatic:
      ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(pt_cp.vl["Getriebe_11"]["GE_Fahrstufe"], None))
    elif trans_type == TransmissionType.direct:
      ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(pt_cp.vl["EV_Gearshift"]["GearPosition"], None))
    elif trans_type == TransmissionType.manual:
      ret.clutchPressed = not pt_cp.vl["Motor_14"]["MO_Kuppl_schalter"]
      if bool(pt_cp.vl["Gateway_72"]["BCM1_Rueckfahrlicht_Schalter"]):
        ret.gearShifter = GearShifter.reverse
      else:
        ret.gearShifter = GearShifter.drive

    # Update door and trunk/hatch lid open status.
    ret.doorOpen = any([pt_cp.vl["Gateway_72"]["ZV_FT_offen"],
                        pt_cp.vl["Gateway_72"]["ZV_BT_offen"],
                        pt_cp.vl["Gateway_72"]["ZV_HFS_offen"],
                        pt_cp.vl["Gateway_72"]["ZV_HBFS_offen"],
                        pt_cp.vl["Gateway_72"]["ZV_HD_offen"]])

    # Update seatbelt fastened status.
    ret.seatbeltUnlatched = pt_cp.vl["Airbag_02"]["AB_Gurtschloss_FA"] != 3

    # Consume blind-spot monitoring info/warning LED states, if available.
    # Infostufe: BSM LED on, Warnung: BSM LED flashing
    if self.CP.enableBsm:
      ret.leftBlindspot = bool(ext_cp.vl["SWA_01"]["SWA_Infostufe_SWA_li"]) or bool(ext_cp.vl["SWA_01"]["SWA_Warnung_SWA_li"])
      ret.rightBlindspot = bool(ext_cp.vl["SWA_01"]["SWA_Infostufe_SWA_re"]) or bool(ext_cp.vl["SWA_01"]["SWA_Warnung_SWA_re"])

    # Consume factory LDW data relevant for factory SWA (Lane Change Assist)
    # and capture it for forwarding to the blind spot radar controller
    self.ldw_stock_values = cam_cp.vl["LDW_02"] if self.CP.networkLocation == NetworkLocation.fwdCamera else {}

    # Stock FCW is considered active if the release bit for brake-jerk warning
    # is set. Stock AEB considered active if the partial braking or target
    # braking release bits are set.
    # Refer to VW Self Study Program 890253: Volkswagen Driver Assistance
    # Systems, chapter on Front Assist with Braking: Golf Family for all MQB
    ret.stockFcw = bool(ext_cp.vl["ACC_10"]["AWV2_Freigabe"])
    ret.stockAeb = bool(ext_cp.vl["ACC_10"]["ANB_Teilbremsung_Freigabe"]) or bool(ext_cp.vl["ACC_10"]["ANB_Zielbremsung_Freigabe"])

    # Update ACC radar status.
    self.tsk_status = pt_cp.vl["TSK_06"]["TSK_Status"]
    if self.tsk_status == 2:
      # ACC okay and enabled, but not currently engaged
      ret.cruiseState.available = True
      ret.cruiseState.enabled = False
    elif self.tsk_status in (3, 4, 5):
      # ACC okay and enabled, currently regulating speed (3) or driver accel override (4) or overrun coast-down (5)
      ret.cruiseState.available = True
      ret.cruiseState.enabled = True
    else:
      # ACC okay but disabled (1), or a radar visibility or other fault/disruption (6 or 7)
      ret.cruiseState.available = False
      ret.cruiseState.enabled = False

    # Update ACC setpoint. When the setpoint is zero or there's an error, the
    # radar sends a set-speed of ~90.69 m/s / 203mph.
    if self.CP.pcmCruise:
      ret.cruiseState.speed = ext_cp.vl["ACC_02"]["ACC_Wunschgeschw"] * CV.KPH_TO_MS
      if ret.cruiseState.speed > 90:
        ret.cruiseState.speed = 0

    # Update control button states for turn signals and ACC controls.
    self.buttonStates["accelCruise"] = bool(pt_cp.vl["GRA_ACC_01"]["GRA_Tip_Hoch"])
    self.buttonStates["decelCruise"] = bool(pt_cp.vl["GRA_ACC_01"]["GRA_Tip_Runter"])
    self.buttonStates["cancel"] = bool(pt_cp.vl["GRA_ACC_01"]["GRA_Abbrechen"])
    self.buttonStates["setCruise"] = bool(pt_cp.vl["GRA_ACC_01"]["GRA_Tip_Setzen"])
    self.buttonStates["resumeCruise"] = bool(pt_cp.vl["GRA_ACC_01"]["GRA_Tip_Wiederaufnahme"])
    self.buttonStates["gapAdjustCruise"] = bool(pt_cp.vl["GRA_ACC_01"]["GRA_Verstellung_Zeitluecke"])
    ret.leftBlinker = bool(pt_cp.vl["Blinkmodi_02"]["Comfort_Signal_Left"])
    ret.rightBlinker = bool(pt_cp.vl["Blinkmodi_02"]["Comfort_Signal_Right"])

    # Read ACC hardware button type configuration info that has to pass thru
    # to the radar. Ends up being different for steering wheel buttons vs
    # third stalk type controls.
    self.graHauptschalter = pt_cp.vl["GRA_ACC_01"]["GRA_Hauptschalter"]
    self.graTypHauptschalter = pt_cp.vl["GRA_ACC_01"]["GRA_Typ_Hauptschalter"]
    self.graButtonTypeInfo = pt_cp.vl["GRA_ACC_01"]["GRA_ButtonTypeInfo"]
    self.graTipStufe2 = pt_cp.vl["GRA_ACC_01"]["GRA_Tip_Stufe_2"]
    # Pick up the GRA_ACC_01 CAN message counter so we can sync to it for
    # later cruise-control button spamming.
    self.graMsgBusCounter = pt_cp.vl["GRA_ACC_01"]["COUNTER"]

    # Additional safety checks performed in CarInterface.
    ret.espDisabled = pt_cp.vl["ESP_21"]["ESP_Tastung_passiv"] != 0

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("LWI_Lenkradwinkel", "LWI_01"),           # Absolute steering angle
      ("LWI_VZ_Lenkradwinkel", "LWI_01"),        # Steering angle sign
      ("LWI_Lenkradw_Geschw", "LWI_01"),         # Absolute steering rate
      ("LWI_VZ_Lenkradw_Geschw", "LWI_01"),      # Steering rate sign
      ("ESP_VL_Radgeschw_02", "ESP_19"),         # ABS wheel speed, front left
      ("ESP_VR_Radgeschw_02", "ESP_19"),         # ABS wheel speed, front right
      ("ESP_HL_Radgeschw_02", "ESP_19"),         # ABS wheel speed, rear left
      ("ESP_HR_Radgeschw_02", "ESP_19"),         # ABS wheel speed, rear right
      ("ESP_Gierrate", "ESP_02"),                # Absolute yaw rate
      ("ESP_VZ_Gierrate", "ESP_02"),             # Yaw rate sign
      ("ZV_FT_offen", "Gateway_72"),             # Door open, driver
      ("ZV_BT_offen", "Gateway_72"),             # Door open, passenger
      ("ZV_HFS_offen", "Gateway_72"),            # Door open, rear left
      ("ZV_HBFS_offen", "Gateway_72"),           # Door open, rear right
      ("ZV_HD_offen", "Gateway_72"),             # Trunk or hatch open
      ("Comfort_Signal_Left", "Blinkmodi_02"),   # Left turn signal including comfort blink interval
      ("Comfort_Signal_Right", "Blinkmodi_02"),  # Right turn signal including comfort blink interval
      ("AB_Gurtschloss_FA", "Airbag_02"),        # Seatbelt status, driver
      ("AB_Gurtschloss_BF", "Airbag_02"),        # Seatbelt status, passenger
      ("ESP_Fahrer_bremst", "ESP_05"),           # Brake pedal pressed
      ("ESP_Bremsdruck", "ESP_05"),              # Brake pressure applied
      ("MO_Fahrpedalrohwert_01", "Motor_20"),    # Accelerator pedal value
      ("EPS_Lenkmoment", "LH_EPS_03"),           # Absolute driver torque input
      ("EPS_VZ_Lenkmoment", "LH_EPS_03"),        # Driver torque input sign
      ("EPS_HCA_Status", "LH_EPS_03"),           # EPS HCA control status
      ("ESP_Tastung_passiv", "ESP_21"),          # Stability control disabled
      ("ESP_Haltebestaetigung", "ESP_21"),       # ESP hold confirmation
      ("KBI_Handbremse", "Kombi_01"),            # Manual handbrake applied
      ("TSK_Status", "TSK_06"),                  # ACC engagement status from drivetrain coordinator
      ("GRA_Hauptschalter", "GRA_ACC_01"),       # ACC button, on/off
      ("GRA_Abbrechen", "GRA_ACC_01"),           # ACC button, cancel
      ("GRA_Tip_Setzen", "GRA_ACC_01"),          # ACC button, set
      ("GRA_Tip_Hoch", "GRA_ACC_01"),            # ACC button, increase or accel
      ("GRA_Tip_Runter", "GRA_ACC_01"),          # ACC button, decrease or decel
      ("GRA_Tip_Wiederaufnahme", "GRA_ACC_01"),  # ACC button, resume
      ("GRA_Verstellung_Zeitluecke", "GRA_ACC_01"),  # ACC button, time gap adj
      ("GRA_Typ_Hauptschalter", "GRA_ACC_01"),   # ACC main button type
      ("GRA_Tip_Stufe_2", "GRA_ACC_01"),         # unknown related to stalk type
      ("GRA_ButtonTypeInfo", "GRA_ACC_01"),      # unknown related to stalk type
      ("COUNTER", "GRA_ACC_01"),                 # GRA_ACC_01 CAN message counter
    ]

    checks = [
      # sig_address, frequency
      ("LWI_01", 100),      # From J500 Steering Assist with integrated sensors
      ("LH_EPS_03", 100),   # From J500 Steering Assist with integrated sensors
      ("ESP_19", 100),      # From J104 ABS/ESP controller
      ("ESP_05", 50),       # From J104 ABS/ESP controller
      ("ESP_21", 50),       # From J104 ABS/ESP controller
      ("Motor_20", 50),     # From J623 Engine control module
      ("TSK_06", 50),       # From J623 Engine control module
      ("ESP_02", 50),       # From J104 ABS/ESP controller
      ("GRA_ACC_01", 33),   # From J533 CAN gateway (via LIN from steering wheel controls)
      ("Gateway_72", 10),   # From J533 CAN gateway (aggregated data)
      ("Airbag_02", 5),     # From J234 Airbag control module
      ("Kombi_01", 2),      # From J285 Instrument cluster
      ("Blinkmodi_02", 1),  # From J519 BCM (sent at 1Hz when no lights active, 50Hz when active)
    ]

    if CP.transmissionType == TransmissionType.automatic:
      signals.append(("GE_Fahrstufe", "Getriebe_11"))  # Auto trans gear selector position
      checks.append(("Getriebe_11", 20))  # From J743 Auto transmission control module
    elif CP.transmissionType == TransmissionType.direct:
      signals.append(("GearPosition", "EV_Gearshift"))  # EV gear selector position
      checks.append(("EV_Gearshift", 10))  # From J??? unknown EV control module
    elif CP.transmissionType == TransmissionType.manual:
      signals += [("MO_Kuppl_schalter", "Motor_14"),  # Clutch switch
                  ("BCM1_Rueckfahrlicht_Schalter", "Gateway_72")]  # Reverse light from BCM
      checks.append(("Motor_14", 10))  # From J623 Engine control module

    if CP.networkLocation == NetworkLocation.fwdCamera:
      # Radars are here on CANBUS.pt
      signals += MqbExtraSignals.fwd_radar_signals
      checks += MqbExtraSignals.fwd_radar_checks
      if CP.enableBsm:
        signals += MqbExtraSignals.bsm_radar_signals
        checks += MqbExtraSignals.bsm_radar_checks

    return CANParser(DBC_FILES.mqb, signals, checks, CANBUS.pt)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = []
    checks = []

    if CP.networkLocation == NetworkLocation.fwdCamera:
      signals += [
        # sig_name, sig_address
        ("LDW_SW_Warnung_links", "LDW_02"),      # Blind spot in warning mode on left side due to lane departure
        ("LDW_SW_Warnung_rechts", "LDW_02"),     # Blind spot in warning mode on right side due to lane departure
        ("LDW_Seite_DLCTLC", "LDW_02"),          # Direction of most likely lane departure (left or right)
        ("LDW_DLC", "LDW_02"),                   # Lane departure, distance to line crossing
        ("LDW_TLC", "LDW_02"),                   # Lane departure, time to line crossing
      ]
      checks += [
        # sig_address, frequency
        ("LDW_02", 10)      # From R242 Driver assistance camera
      ]
    else:
      # Radars are here on CANBUS.cam
      signals += MqbExtraSignals.fwd_radar_signals
      checks += MqbExtraSignals.fwd_radar_checks
      if CP.enableBsm:
        signals += MqbExtraSignals.bsm_radar_signals
        checks += MqbExtraSignals.bsm_radar_checks

    return CANParser(DBC_FILES.mqb, signals, checks, CANBUS.cam)

class MqbExtraSignals:
  # Additional signal and message lists for optional or bus-portable controllers
  fwd_radar_signals = [
    ("ACC_Wunschgeschw", "ACC_02"),              # ACC set speed
    ("AWV2_Freigabe", "ACC_10"),                 # FCW brake jerk release
    ("ANB_Teilbremsung_Freigabe", "ACC_10"),     # AEB partial braking release
    ("ANB_Zielbremsung_Freigabe", "ACC_10"),     # AEB target braking release
  ]
  fwd_radar_checks = [
    ("ACC_10", 50),                                 # From J428 ACC radar control module
    ("ACC_02", 17),                                 # From J428 ACC radar control module
  ]
  bsm_radar_signals = [
    ("SWA_Infostufe_SWA_li", "SWA_01"),          # Blind spot object info, left
    ("SWA_Warnung_SWA_li", "SWA_01"),            # Blind spot object warning, left
    ("SWA_Infostufe_SWA_re", "SWA_01"),          # Blind spot object info, right
    ("SWA_Warnung_SWA_re", "SWA_01"),            # Blind spot object warning, right
  ]
  bsm_radar_checks = [
    ("SWA_01", 20),                                 # From J1086 Lane Change Assist
  ]
