import copy
from cereal import car
from selfdrive.car.tesla.values import DBC, CANBUS, GEAR_MAP, DOORS, BUTTONS
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.config import Conversions as CV

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.button_states = {button.event_type: False for button in BUTTONS}
    self.can_define = CANDefine(DBC[CP.carFingerprint]['chassis'])

    # Needed by carcontroller
    self.msg_stw_actn_req = None
    self.hands_on_level = 0
    self.steer_warning = None

  def update(self, cp, cp_cam):
    ret = car.CarState.new_message()

    # Vehicle speed
    ret.vEgoRaw = cp.vl["ESP_B"]["ESP_vehicleSpeed"] * CV.KPH_TO_MS
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = (ret.vEgo < 0.1)

    # Gas pedal
    ret.gas = cp.vl["DI_torque1"]["DI_pedalPos"] / 100.0
    ret.gasPressed = (ret.gas > 0)

    # Brake pedal
    ret.brake = 0
    ret.brakePressed = bool(cp.vl["BrakeMessage"]["driverBrakeStatus"] != 1)

    # Steering wheel
    self.hands_on_level = cp.vl["EPAS_sysStatus"]["EPAS_handsOnLevel"]
    self.steer_warning = self.can_define.dv["EPAS_sysStatus"]["EPAS_eacErrorCode"].get(int(cp.vl["EPAS_sysStatus"]["EPAS_eacErrorCode"]), None)
    steer_status = self.can_define.dv["EPAS_sysStatus"]["EPAS_eacStatus"].get(int(cp.vl["EPAS_sysStatus"]["EPAS_eacStatus"]), None)

    ret.steeringAngleDeg = -cp.vl["EPAS_sysStatus"]["EPAS_internalSAS"]
    ret.steeringRateDeg = -cp.vl["STW_ANGLHP_STAT"]["StW_AnglHP_Spd"] # This is from a different angle sensor, and at different rate
    ret.steeringTorque = -cp.vl["EPAS_sysStatus"]["EPAS_torsionBarTorque"]
    ret.steeringPressed = (self.hands_on_level > 0)
    ret.steerError = steer_status in ["EAC_FAULT", "EAC_INHIBITED"]
    ret.steerWarning = self.steer_warning in ["EAC_ERROR_MAX_SPEED", "EAC_ERROR_MIN_SPEED", "EAC_ERROR_TMP_FAULT", "SNA"]  # TODO: not sure if this list is complete

    # Cruise state
    cruise_state = self.can_define.dv["DI_state"]["DI_cruiseState"].get(int(cp.vl["DI_state"]["DI_cruiseState"]), None)
    speed_units = self.can_define.dv["DI_state"]["DI_speedUnits"].get(int(cp.vl["DI_state"]["DI_speedUnits"]), None)

    acc_enabled = (cruise_state in ["ENABLED", "STANDSTILL", "OVERRIDE", "PRE_FAULT", "PRE_CANCEL"])

    ret.cruiseState.enabled = acc_enabled
    if speed_units == "KPH":
      ret.cruiseState.speed = cp.vl["DI_state"]["DI_digitalSpeed"] * CV.KPH_TO_MS
    elif speed_units == "MPH":
      ret.cruiseState.speed = cp.vl["DI_state"]["DI_digitalSpeed"] * CV.MPH_TO_MS
    ret.cruiseState.available = ((cruise_state == "STANDBY") or ret.cruiseState.enabled)
    ret.cruiseState.standstill = (cruise_state == "STANDSTILL")

    # Gear
    ret.gearShifter = GEAR_MAP[self.can_define.dv["DI_torque2"]["DI_gear"].get(int(cp.vl["DI_torque2"]["DI_gear"]), "DI_GEAR_INVALID")]

    # Buttons
    buttonEvents = []
    for button in BUTTONS:
      state = (cp.vl[button.can_addr][button.can_msg] in button.values)
      if self.button_states[button.event_type] != state:
        event = car.CarState.ButtonEvent.new_message()
        event.type = button.event_type
        event.pressed = state
        buttonEvents.append(event)
      self.button_states[button.event_type] = state
    ret.buttonEvents = buttonEvents

    # Doors
    ret.doorOpen = any([(self.can_define.dv["GTW_carState"][door].get(int(cp.vl["GTW_carState"][door]), "OPEN") == "OPEN") for door in DOORS])

    # Blinkers
    ret.leftBlinker = (cp.vl["GTW_carState"]["BC_indicatorLStatus"] == 1)
    ret.rightBlinker = (cp.vl["GTW_carState"]["BC_indicatorRStatus"] == 1)

    # Seatbelt
    ret.seatbeltUnlatched = (cp.vl["SDM1"]["SDM_bcklDrivStatus"] != 1)

    # TODO: blindspot

    # Messages needed by carcontroller
    self.msg_stw_actn_req = copy.copy(cp.vl["STW_ACTN_RQ"])

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
      ("ESP_vehicleSpeed", "ESP_B", 0),
      ("DI_pedalPos", "DI_torque1", 0),
      ("DI_brakePedal", "DI_torque2", 0),
      ("StW_AnglHP", "STW_ANGLHP_STAT", 0),
      ("StW_AnglHP_Spd", "STW_ANGLHP_STAT", 0),
      ("EPAS_handsOnLevel", "EPAS_sysStatus", 0),
      ("EPAS_torsionBarTorque", "EPAS_sysStatus", 0),
      ("EPAS_internalSAS", "EPAS_sysStatus", 0),
      ("EPAS_eacStatus", "EPAS_sysStatus", 1),
      ("EPAS_eacErrorCode", "EPAS_sysStatus", 0),
      ("DI_cruiseState", "DI_state", 0),
      ("DI_digitalSpeed", "DI_state", 0),
      ("DI_speedUnits", "DI_state", 0),
      ("DI_gear", "DI_torque2", 0),
      ("DOOR_STATE_FL", "GTW_carState", 1),
      ("DOOR_STATE_FR", "GTW_carState", 1),
      ("DOOR_STATE_RL", "GTW_carState", 1),
      ("DOOR_STATE_RR", "GTW_carState", 1),
      ("DOOR_STATE_FrontTrunk", "GTW_carState", 1),
      ("BOOT_STATE", "GTW_carState", 1),
      ("BC_indicatorLStatus", "GTW_carState", 1),
      ("BC_indicatorRStatus", "GTW_carState", 1),
      ("SDM_bcklDrivStatus", "SDM1", 0),
      ("driverBrakeStatus", "BrakeMessage", 0),

      # We copy this whole message when spamming cancel
      ("SpdCtrlLvr_Stat", "STW_ACTN_RQ", 0),
      ("VSL_Enbl_Rq", "STW_ACTN_RQ", 0),
      ("SpdCtrlLvrStat_Inv", "STW_ACTN_RQ", 0),
      ("DTR_Dist_Rq", "STW_ACTN_RQ", 0),
      ("TurnIndLvr_Stat", "STW_ACTN_RQ", 0),
      ("HiBmLvr_Stat", "STW_ACTN_RQ", 0),
      ("WprWashSw_Psd", "STW_ACTN_RQ", 0),
      ("WprWash_R_Sw_Posn_V2", "STW_ACTN_RQ", 0),
      ("StW_Lvr_Stat", "STW_ACTN_RQ", 0),
      ("StW_Cond_Flt", "STW_ACTN_RQ", 0),
      ("StW_Cond_Psd", "STW_ACTN_RQ", 0),
      ("HrnSw_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw00_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw01_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw02_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw03_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw04_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw05_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw06_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw07_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw08_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw09_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw10_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw11_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw12_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw13_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw14_Psd", "STW_ACTN_RQ", 0),
      ("StW_Sw15_Psd", "STW_ACTN_RQ", 0),
      ("WprSw6Posn", "STW_ACTN_RQ", 0),
      ("MC_STW_ACTN_RQ", "STW_ACTN_RQ", 0),
      ("CRC_STW_ACTN_RQ", "STW_ACTN_RQ", 0),
    ]

    checks = [
      # sig_address, frequency
      ("ESP_B", 50),
      ("DI_torque1", 100),
      ("DI_torque2", 100),
      ("STW_ANGLHP_STAT", 100),
      ("EPAS_sysStatus", 25),
      ("DI_state", 10),
      ("STW_ACTN_RQ", 10),
      ("GTW_carState", 10),
      ("SDM1", 10),
      ("BrakeMessage", 50),
    ]

    return CANParser(DBC[CP.carFingerprint]['chassis'], signals, checks, CANBUS.chassis)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
    ]
    checks = [
      # sig_address, frequency
    ]
    return CANParser(DBC[CP.carFingerprint]['chassis'], signals, checks, CANBUS.autopilot)
