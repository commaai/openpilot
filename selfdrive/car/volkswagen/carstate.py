import numpy as np
from cereal import car
from common.conversions import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.car.volkswagen.values import DBC, CANBUS, PQ_CARS, NetworkLocation, TransmissionType, GearShifter, \
                                            CarControllerParams


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.CCP = CarControllerParams(CP)
    self.button_states = {button.event_type: False for button in self.CCP.BUTTONS}
    self.esp_hold_confirmation = False
    self.upscale_lead_car_signal = False

  def create_button_events(self, pt_cp, buttons):
    button_events = []

    for button in buttons:
      state = pt_cp.vl[button.can_addr][button.can_msg] in button.values
      if self.button_states[button.event_type] != state:
        event = car.CarState.ButtonEvent.new_message()
        event.type = button.event_type
        event.pressed = state
        button_events.append(event)
      self.button_states[button.event_type] = state

    return button_events

  def update(self, pt_cp, cam_cp, ext_cp, trans_type):
    if self.CP.carFingerprint in PQ_CARS:
      return self.update_pq(pt_cp, cam_cp, ext_cp, trans_type)

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
    ret.standstill = ret.vEgoRaw == 0

    # Update steering angle, rate, yaw rate, and driver input torque. VW send
    # the sign/direction in a separate signal so they must be recombined.
    ret.steeringAngleDeg = pt_cp.vl["LWI_01"]["LWI_Lenkradwinkel"] * (1, -1)[int(pt_cp.vl["LWI_01"]["LWI_VZ_Lenkradwinkel"])]
    ret.steeringRateDeg = pt_cp.vl["LWI_01"]["LWI_Lenkradw_Geschw"] * (1, -1)[int(pt_cp.vl["LWI_01"]["LWI_VZ_Lenkradw_Geschw"])]
    ret.steeringTorque = pt_cp.vl["LH_EPS_03"]["EPS_Lenkmoment"] * (1, -1)[int(pt_cp.vl["LH_EPS_03"]["EPS_VZ_Lenkmoment"])]
    ret.steeringPressed = abs(ret.steeringTorque) > self.CCP.STEER_DRIVER_ALLOWANCE
    ret.yawRate = pt_cp.vl["ESP_02"]["ESP_Gierrate"] * (1, -1)[int(pt_cp.vl["ESP_02"]["ESP_VZ_Gierrate"])] * CV.DEG_TO_RAD

    # Verify EPS readiness to accept steering commands
    hca_status = self.CCP.hca_status_values.get(pt_cp.vl["LH_EPS_03"]["EPS_HCA_Status"])
    ret.steerFaultPermanent = hca_status in ("DISABLED", "FAULT")
    ret.steerFaultTemporary = hca_status in ("INITIALIZING", "REJECTED")

    # Update gas, brakes, and gearshift.
    ret.gas = pt_cp.vl["Motor_20"]["MO_Fahrpedalrohwert_01"] / 100.0
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["ESP_05"]["ESP_Bremsdruck"] / 250.0  # FIXME: this is pressure in Bar, not sure what OP expects
    brake_pedal_pressed = bool(pt_cp.vl["Motor_14"]["MO_Fahrer_bremst"])
    brake_pressure_detected = bool(pt_cp.vl["ESP_05"]["ESP_Fahrer_bremst"])
    ret.brakePressed = brake_pedal_pressed or brake_pressure_detected
    ret.parkingBrake = bool(pt_cp.vl["Kombi_01"]["KBI_Handbremse"])  # FIXME: need to include an EPB check as well

    # Update gear and/or clutch position data.
    if trans_type == TransmissionType.automatic:
      ret.gearShifter = self.parse_gear_shifter(self.CCP.shifter_values.get(pt_cp.vl["Getriebe_11"]["GE_Fahrstufe"], None))
    elif trans_type == TransmissionType.direct:
      ret.gearShifter = self.parse_gear_shifter(self.CCP.shifter_values.get(pt_cp.vl["EV_Gearshift"]["GearPosition"], None))
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
    self.acc_type = ext_cp.vl["ACC_06"]["ACC_Typ"]
    if pt_cp.vl["TSK_06"]["TSK_Status"] == 2:
      # ACC okay and enabled, but not currently engaged
      ret.cruiseState.available = True
      ret.cruiseState.enabled = False
    elif pt_cp.vl["TSK_06"]["TSK_Status"] in (3, 4, 5):
      # ACC okay and enabled, currently regulating speed (3) or driver accel override (4) or brake only (5)
      ret.cruiseState.available = True
      ret.cruiseState.enabled = True
    else:
      # ACC okay but disabled (1), or a radar visibility or other fault/disruption (6 or 7)
      ret.cruiseState.available = False
      ret.cruiseState.enabled = False
    self.esp_hold_confirmation = bool(pt_cp.vl["ESP_21"]["ESP_Haltebestaetigung"])
    ret.cruiseState.standstill = self.CP.pcmCruise and self.esp_hold_confirmation
    ret.accFaulted = pt_cp.vl["TSK_06"]["TSK_Status"] in (6, 7)

    # Update ACC setpoint. When the setpoint is zero or there's an error, the
    # radar sends a set-speed of ~90.69 m/s / 203mph.
    if self.CP.pcmCruise:
      ret.cruiseState.speed = ext_cp.vl["ACC_02"]["ACC_Wunschgeschw_02"] * CV.KPH_TO_MS
      if ret.cruiseState.speed > 90:
        ret.cruiseState.speed = 0

    # Update button states for turn signals and ACC controls, capture all ACC button state/config for passthrough
    ret.leftBlinker = bool(pt_cp.vl["Blinkmodi_02"]["Comfort_Signal_Left"])
    ret.rightBlinker = bool(pt_cp.vl["Blinkmodi_02"]["Comfort_Signal_Right"])
    ret.buttonEvents = self.create_button_events(pt_cp, self.CCP.BUTTONS)
    self.gra_stock_values = pt_cp.vl["GRA_ACC_01"]

    # Additional safety checks performed in CarInterface.
    ret.espDisabled = pt_cp.vl["ESP_21"]["ESP_Tastung_passiv"] != 0

    # Digital instrument clusters expect the ACC HUD lead car distance to be scaled differently
    self.upscale_lead_car_signal = bool(pt_cp.vl["Kombi_03"]["KBI_Variante"])

    return ret

  def update_pq(self, pt_cp, cam_cp, ext_cp, trans_type):
    ret = car.CarState.new_message()
    # Update vehicle speed and acceleration from ABS wheel speeds.
    ret.wheelSpeeds = self.get_wheel_speeds(
      pt_cp.vl["Bremse_3"]["Radgeschw__VL_4_1"],
      pt_cp.vl["Bremse_3"]["Radgeschw__VR_4_1"],
      pt_cp.vl["Bremse_3"]["Radgeschw__HL_4_1"],
      pt_cp.vl["Bremse_3"]["Radgeschw__HR_4_1"],
    )

    # vEgo obtained from Bremse_1 vehicle speed rather than Bremse_3 wheel speeds because Bremse_3 isn't present on NSF
    ret.vEgoRaw = pt_cp.vl["Bremse_1"]["Geschwindigkeit_neu__Bremse_1_"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw == 0

    # Update steering angle, rate, yaw rate, and driver input torque. VW send
    # the sign/direction in a separate signal so they must be recombined.
    ret.steeringAngleDeg = pt_cp.vl["Lenkhilfe_3"]["LH3_BLW"] * (1, -1)[int(pt_cp.vl["Lenkhilfe_3"]["LH3_BLWSign"])]
    ret.steeringRateDeg = pt_cp.vl["Lenkwinkel_1"]["Lenkradwinkel_Geschwindigkeit"] * (1, -1)[int(pt_cp.vl["Lenkwinkel_1"]["Lenkradwinkel_Geschwindigkeit_S"])]
    ret.steeringTorque = pt_cp.vl["Lenkhilfe_3"]["LH3_LM"] * (1, -1)[int(pt_cp.vl["Lenkhilfe_3"]["LH3_LMSign"])]
    ret.steeringPressed = abs(ret.steeringTorque) > self.CCP.STEER_DRIVER_ALLOWANCE
    ret.yawRate = pt_cp.vl["Bremse_5"]["Giergeschwindigkeit"] * (1, -1)[int(pt_cp.vl["Bremse_5"]["Vorzeichen_der_Giergeschwindigk"])] * CV.DEG_TO_RAD

    # Verify EPS readiness to accept steering commands
    hca_status = self.CCP.hca_status_values.get(pt_cp.vl["Lenkhilfe_2"]["LH2_Sta_HCA"])
    ret.steerFaultPermanent = hca_status in ("DISABLED", "FAULT")
    ret.steerFaultTemporary = hca_status in ("INITIALIZING", "REJECTED")

    # Update gas, brakes, and gearshift.
    ret.gas = pt_cp.vl["Motor_3"]["Fahrpedal_Rohsignal"] / 100.0
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["Bremse_5"]["Bremsdruck"] / 250.0  # FIXME: this is pressure in Bar, not sure what OP expects
    ret.brakePressed = bool(pt_cp.vl["Motor_2"]["Bremslichtschalter"])
    ret.parkingBrake = bool(pt_cp.vl["Kombi_1"]["Bremsinfo"])

    # Update gear and/or clutch position data.
    if trans_type == TransmissionType.automatic:
      ret.gearShifter = self.parse_gear_shifter(self.CCP.shifter_values.get(pt_cp.vl["Getriebe_1"]["Waehlhebelposition__Getriebe_1_"], None))
    elif trans_type == TransmissionType.manual:
      ret.clutchPressed = not pt_cp.vl["Motor_1"]["Kupplungsschalter"]
      reverse_light = bool(pt_cp.vl["Gate_Komf_1"]["GK1_Rueckfahr"])
      if reverse_light:
        ret.gearShifter = GearShifter.reverse
      else:
        ret.gearShifter = GearShifter.drive

    # Update door and trunk/hatch lid open status.
    ret.doorOpen = any([pt_cp.vl["Gate_Komf_1"]["GK1_Fa_Tuerkont"],
                        pt_cp.vl["Gate_Komf_1"]["BSK_BT_geoeffnet"],
                        pt_cp.vl["Gate_Komf_1"]["BSK_HL_geoeffnet"],
                        pt_cp.vl["Gate_Komf_1"]["BSK_HR_geoeffnet"],
                        pt_cp.vl["Gate_Komf_1"]["BSK_HD_Hauptraste"]])

    # Update seatbelt fastened status.
    ret.seatbeltUnlatched = not bool(pt_cp.vl["Airbag_1"]["Gurtschalter_Fahrer"])

    # Consume blind-spot monitoring info/warning LED states, if available.
    # Infostufe: BSM LED on, Warnung: BSM LED flashing
    if self.CP.enableBsm:
      ret.leftBlindspot = bool(ext_cp.vl["SWA_1"]["SWA_Infostufe_SWA_li"]) or bool(ext_cp.vl["SWA_1"]["SWA_Warnung_SWA_li"])
      ret.rightBlindspot = bool(ext_cp.vl["SWA_1"]["SWA_Infostufe_SWA_re"]) or bool(ext_cp.vl["SWA_1"]["SWA_Warnung_SWA_re"])

    # Consume factory LDW data relevant for factory SWA (Lane Change Assist)
    # and capture it for forwarding to the blind spot radar controller
    self.ldw_stock_values = cam_cp.vl["LDW_Status"] if self.CP.networkLocation == NetworkLocation.fwdCamera else {}

    # Stock FCW is considered active if the release bit for brake-jerk warning
    # is set. Stock AEB considered active if the partial braking or target
    # braking release bits are set.
    # Refer to VW Self Study Program 890253: Volkswagen Driver Assistance
    # Systems, chapters on Front Assist with Braking and City Emergency
    # Braking for the 2016 Passat NMS
    # TODO: deferred until we can collect data on pre-MY2016 behavior, AWV message may be shorter with fewer signals
    ret.stockFcw = False
    ret.stockAeb = False

    # Update ACC radar status.
    self.acc_type = ext_cp.vl["ACC_System"]["ACS_Typ_ACC"]
    ret.cruiseState.available = bool(pt_cp.vl["Motor_5"]["GRA_Hauptschalter"])
    ret.cruiseState.enabled = pt_cp.vl["Motor_2"]["GRA_Status"] in (1, 2)
    if self.CP.pcmCruise:
      ret.accFaulted = ext_cp.vl["ACC_GRA_Anzeige"]["ACA_StaACC"] in (6, 7)
    else:
      ret.accFaulted = pt_cp.vl["Motor_2"]["GRA_Status"] == 3

    # Update ACC setpoint. When the setpoint reads as 255, the driver has not
    # yet established an ACC setpoint, so treat it as zero.
    ret.cruiseState.speed = ext_cp.vl["ACC_GRA_Anzeige"]["ACA_V_Wunsch"] * CV.KPH_TO_MS
    if ret.cruiseState.speed > 70:  # 255 kph in m/s == no current setpoint
      ret.cruiseState.speed = 0

    # Update button states for turn signals and ACC controls, capture all ACC button state/config for passthrough
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_stalk(300, pt_cp.vl["Gate_Komf_1"]["GK1_Blinker_li"],
                                                                            pt_cp.vl["Gate_Komf_1"]["GK1_Blinker_re"])
    ret.buttonEvents = self.create_button_events(pt_cp, self.CCP.BUTTONS)
    self.gra_stock_values = pt_cp.vl["GRA_Neu"]

    # Additional safety checks performed in CarInterface.
    ret.espDisabled = bool(pt_cp.vl["Bremse_1"]["ESP_Passiv_getastet"])

    return ret

  @staticmethod
  def get_can_parser(CP):
    if CP.carFingerprint in PQ_CARS:
      return CarState.get_can_parser_pq(CP)

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
      ("ESP_Fahrer_bremst", "ESP_05"),           # Driver applied brake pressure over threshold
      ("MO_Fahrer_bremst", "Motor_14"),          # Brake pedal switch
      ("ESP_Bremsdruck", "ESP_05"),              # Brake pressure
      ("MO_Fahrpedalrohwert_01", "Motor_20"),    # Accelerator pedal value
      ("EPS_Lenkmoment", "LH_EPS_03"),           # Absolute driver torque input
      ("EPS_VZ_Lenkmoment", "LH_EPS_03"),        # Driver torque input sign
      ("EPS_HCA_Status", "LH_EPS_03"),           # EPS HCA control status
      ("ESP_Tastung_passiv", "ESP_21"),          # Stability control disabled
      ("ESP_Haltebestaetigung", "ESP_21"),       # ESP hold confirmation
      ("KBI_Handbremse", "Kombi_01"),            # Manual handbrake applied
      ("KBI_Variante", "Kombi_03"),              # Digital/full-screen instrument cluster installed
      ("TSK_Status", "TSK_06"),                  # ACC engagement status from drivetrain coordinator
      ("GRA_Hauptschalter", "GRA_ACC_01"),       # ACC button, on/off
      ("GRA_Abbrechen", "GRA_ACC_01"),           # ACC button, cancel
      ("GRA_Tip_Setzen", "GRA_ACC_01"),          # ACC button, set
      ("GRA_Tip_Hoch", "GRA_ACC_01"),            # ACC button, increase or accel
      ("GRA_Tip_Runter", "GRA_ACC_01"),          # ACC button, decrease or decel
      ("GRA_Tip_Wiederaufnahme", "GRA_ACC_01"),  # ACC button, resume
      ("GRA_Verstellung_Zeitluecke", "GRA_ACC_01"),  # ACC button, time gap adj
      ("GRA_Typ_Hauptschalter", "GRA_ACC_01"),   # ACC main button type
      ("GRA_Codierung", "GRA_ACC_01"),           # ACC button configuration/coding
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
      ("Motor_14", 10),     # From J623 Engine control module
      ("Airbag_02", 5),     # From J234 Airbag control module
      ("Kombi_01", 2),      # From J285 Instrument cluster
      ("Blinkmodi_02", 1),  # From J519 BCM (sent at 1Hz when no lights active, 50Hz when active)
      ("Kombi_03", 0),      # From J285 instrument cluster (not present on older cars, 1Hz when present)
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

    if CP.networkLocation == NetworkLocation.fwdCamera:
      # Radars are here on CANBUS.pt
      signals += MqbExtraSignals.fwd_radar_signals
      checks += MqbExtraSignals.fwd_radar_checks
      if CP.enableBsm:
        signals += MqbExtraSignals.bsm_radar_signals
        checks += MqbExtraSignals.bsm_radar_checks

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, CANBUS.pt)

  @staticmethod
  def get_cam_can_parser(CP):
    if CP.carFingerprint in PQ_CARS:
      return CarState.get_cam_can_parser_pq(CP)

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

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, CANBUS.cam)

  @staticmethod
  def get_can_parser_pq(CP):
    signals = [
      # sig_name, sig_address, default
      ("LH3_BLW", "Lenkhilfe_3"),                # Absolute steering angle
      ("LH3_BLWSign", "Lenkhilfe_3"),            # Steering angle sign
      ("LH3_LM", "Lenkhilfe_3"),                 # Absolute driver torque input
      ("LH3_LMSign", "Lenkhilfe_3"),             # Driver torque input sign
      ("LH2_Sta_HCA", "Lenkhilfe_2"),            # Steering rack HCA status
      ("Lenkradwinkel_Geschwindigkeit", "Lenkwinkel_1"),  # Absolute steering rate
      ("Lenkradwinkel_Geschwindigkeit_S", "Lenkwinkel_1"),  # Steering rate sign
      ("Geschwindigkeit_neu__Bremse_1_", "Bremse_1"),  # Vehicle speed from ABS
      ("Radgeschw__VL_4_1", "Bremse_3"),         # ABS wheel speed, front left
      ("Radgeschw__VR_4_1", "Bremse_3"),         # ABS wheel speed, front right
      ("Radgeschw__HL_4_1", "Bremse_3"),         # ABS wheel speed, rear left
      ("Radgeschw__HR_4_1", "Bremse_3"),         # ABS wheel speed, rear right
      ("Giergeschwindigkeit", "Bremse_5"),       # Absolute yaw rate
      ("Vorzeichen_der_Giergeschwindigk", "Bremse_5"),  # Yaw rate sign
      ("Gurtschalter_Fahrer", "Airbag_1"),       # Seatbelt status, driver
      ("Gurtschalter_Beifahrer", "Airbag_1"),    # Seatbelt status, passenger
      ("Bremstestschalter", "Motor_2"),          # Brake pedal pressed (brake light test switch)
      ("Bremslichtschalter", "Motor_2"),         # Brakes applied (brake light switch)
      ("Bremsdruck", "Bremse_5"),                # Brake pressure applied
      ("Vorzeichen_Bremsdruck", "Bremse_5"),     # Brake pressure applied sign (???)
      ("Fahrpedal_Rohsignal", "Motor_3"),        # Accelerator pedal value
      ("ESP_Passiv_getastet", "Bremse_1"),       # Stability control disabled
      ("GRA_Hauptschalter", "Motor_5"),          # ACC main switch
      ("GRA_Status", "Motor_2"),                 # ACC engagement status
      ("GK1_Fa_Tuerkont", "Gate_Komf_1"),        # Door open, driver
      ("BSK_BT_geoeffnet", "Gate_Komf_1"),       # Door open, passenger
      ("BSK_HL_geoeffnet", "Gate_Komf_1"),       # Door open, rear left
      ("BSK_HR_geoeffnet", "Gate_Komf_1"),       # Door open, rear right
      ("BSK_HD_Hauptraste", "Gate_Komf_1"),      # Trunk or hatch open
      ("GK1_Blinker_li", "Gate_Komf_1"),         # Left turn signal on
      ("GK1_Blinker_re", "Gate_Komf_1"),         # Right turn signal on
      ("Bremsinfo", "Kombi_1"),                  # Manual handbrake applied
      ("GRA_Hauptschalt", "GRA_Neu"),            # ACC button, on/off
      ("GRA_Typ_Hauptschalt", "GRA_Neu"),        # ACC button, momentary vs latching
      ("GRA_Kodierinfo", "GRA_Neu"),             # ACC button, configuration
      ("GRA_Abbrechen", "GRA_Neu"),              # ACC button, cancel
      ("GRA_Neu_Setzen", "GRA_Neu"),             # ACC button, set
      ("GRA_Up_lang", "GRA_Neu"),                # ACC button, increase or accel, long press
      ("GRA_Down_lang", "GRA_Neu"),              # ACC button, decrease or decel, long press
      ("GRA_Up_kurz", "GRA_Neu"),                # ACC button, increase or accel, short press
      ("GRA_Down_kurz", "GRA_Neu"),              # ACC button, decrease or decel, short press
      ("GRA_Recall", "GRA_Neu"),                 # ACC button, resume
      ("GRA_Zeitluecke", "GRA_Neu"),             # ACC button, time gap adj
      ("COUNTER", "GRA_Neu"),                    # ACC button, message counter
      ("GRA_Sender", "GRA_Neu"),                 # ACC button, CAN message originator
    ]

    checks = [
      # sig_address, frequency
      ("Bremse_1", 100),    # From J104 ABS/ESP controller
      ("Bremse_3", 100),    # From J104 ABS/ESP controller
      ("Lenkhilfe_3", 100),  # From J500 Steering Assist with integrated sensors
      ("Lenkwinkel_1", 100),  # From J500 Steering Assist with integrated sensors
      ("Motor_3", 100),     # From J623 Engine control module
      ("Airbag_1", 50),     # From J234 Airbag control module
      ("Bremse_5", 50),     # From J104 ABS/ESP controller
      ("GRA_Neu", 50),      # From J??? steering wheel control buttons
      ("Kombi_1", 50),      # From J285 Instrument cluster
      ("Motor_2", 50),      # From J623 Engine control module
      ("Motor_5", 50),      # From J623 Engine control module
      ("Lenkhilfe_2", 20),  # From J500 Steering Assist with integrated sensors
      ("Gate_Komf_1", 10),  # From J533 CAN gateway
    ]

    if CP.transmissionType == TransmissionType.automatic:
      signals += [("Waehlhebelposition__Getriebe_1_", "Getriebe_1", 0)]  # Auto trans gear selector position
      checks += [("Getriebe_1", 100)]  # From J743 Auto transmission control module
    elif CP.transmissionType == TransmissionType.manual:
      signals += [("Kupplungsschalter", "Motor_1", 0),  # Clutch switch
                  ("GK1_Rueckfahr", "Gate_Komf_1", 0)]  # Reverse light from BCM
      checks += [("Motor_1", 100)]  # From J623 Engine control module

    if CP.networkLocation == NetworkLocation.fwdCamera:
      # Extended CAN devices other than the camera are here on CANBUS.pt
      signals += PqExtraSignals.fwd_radar_signals
      checks += PqExtraSignals.fwd_radar_checks
      if CP.enableBsm:
        signals += PqExtraSignals.bsm_radar_signals
        checks += PqExtraSignals.bsm_radar_checks

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, CANBUS.pt)

  @staticmethod
  def get_cam_can_parser_pq(CP):

    signals = []
    checks = []

    if CP.networkLocation == NetworkLocation.fwdCamera:
      signals += [
        # sig_name, sig_address
        ("LDW_SW_Warnung_links", "LDW_Status"),      # Blind spot in warning mode on left side due to lane departure
        ("LDW_SW_Warnung_rechts", "LDW_Status"),     # Blind spot in warning mode on right side due to lane departure
        ("LDW_Seite_DLCTLC", "LDW_Status"),          # Direction of most likely lane departure (left or right)
        ("LDW_DLC", "LDW_Status"),                   # Lane departure, distance to line crossing
        ("LDW_TLC", "LDW_Status"),                   # Lane departure, time to line crossing
      ]
      checks += [
        # sig_address, frequency
        ("LDW_Status", 10)      # From R242 Driver assistance camera
      ]

    if CP.networkLocation == NetworkLocation.gateway:
      # Radars are here on CANBUS.cam
      signals += PqExtraSignals.fwd_radar_signals
      checks += PqExtraSignals.fwd_radar_checks
      if CP.enableBsm:
        signals += PqExtraSignals.bsm_radar_signals
        checks += PqExtraSignals.bsm_radar_checks

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, CANBUS.cam)


class MqbExtraSignals:
  # Additional signal and message lists for optional or bus-portable controllers
  fwd_radar_signals = [
    ("ACC_Wunschgeschw_02", "ACC_02"),           # ACC set speed
    ("ACC_Typ", "ACC_06"),                       # Basic vs FtS vs SnG
    ("AWV2_Freigabe", "ACC_10"),                 # FCW brake jerk release
    ("ANB_Teilbremsung_Freigabe", "ACC_10"),     # AEB partial braking release
    ("ANB_Zielbremsung_Freigabe", "ACC_10"),     # AEB target braking release
  ]
  fwd_radar_checks = [
    ("ACC_06", 50),                              # From J428 ACC radar control module
    ("ACC_10", 50),                              # From J428 ACC radar control module
    ("ACC_02", 17),                              # From J428 ACC radar control module
  ]
  bsm_radar_signals = [
    ("SWA_Infostufe_SWA_li", "SWA_01"),          # Blind spot object info, left
    ("SWA_Warnung_SWA_li", "SWA_01"),            # Blind spot object warning, left
    ("SWA_Infostufe_SWA_re", "SWA_01"),          # Blind spot object info, right
    ("SWA_Warnung_SWA_re", "SWA_01"),            # Blind spot object warning, right
  ]
  bsm_radar_checks = [
    ("SWA_01", 20),                              # From J1086 Lane Change Assist
  ]

class PqExtraSignals:
  # Additional signal and message lists for optional or bus-portable controllers
  fwd_radar_signals = [
    ("ACS_Typ_ACC", "ACC_System"),               # Basic vs FtS (no SnG support on PQ)
    ("ACA_StaACC", "ACC_GRA_Anzeige"),           # ACC drivetrain coordinator status
    ("ACA_V_Wunsch", "ACC_GRA_Anzeige"),         # ACC set speed
  ]
  fwd_radar_checks = [
    ("ACC_System", 50),                          # From J428 ACC radar control module
    ("ACC_GRA_Anzeige", 25),                     # From J428 ACC radar control module
  ]
  bsm_radar_signals = [
    ("SWA_Infostufe_SWA_li", "SWA_1"),           # Blind spot object info, left
    ("SWA_Warnung_SWA_li", "SWA_1"),             # Blind spot object warning, left
    ("SWA_Infostufe_SWA_re", "SWA_1"),           # Blind spot object info, right
    ("SWA_Warnung_SWA_re", "SWA_1"),             # Blind spot object warning, right
  ]
  bsm_radar_checks = [
    ("SWA_1", 20),                               # From J1086 Lane Change Assist
  ]
