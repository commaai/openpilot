import numpy as np
from cereal import car
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.config import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.stellantis.values import DBC, CarControllerParams as P
from selfdrive.controls.lib.vehicle_model import VehicleModel  # temporary for calculating yawRate


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["SHIFTER_ASSM"]["SHIFTER_POSITION"]
    self.VM = VehicleModel(CP)  # temporary for calculating yawRate

  def update(self, cp, cp_cam):
    ret = car.CarState.new_message()

    # Update vehicle speed and acceleration from ABS wheel speeds
    # TODO: Verify wheel speeds match up to labeled positions
    # TODO: Find out why these values go bonkers-high sometimes
    ret.wheelSpeeds.fl = cp.vl["ABS_4"]["WHEEL_SPEED_FL"]
    ret.wheelSpeeds.fr = cp.vl["ABS_4"]["WHEEL_SPEED_FR"]
    ret.wheelSpeeds.rl = cp.vl["ABS_4"]["WHEEL_SPEED_RL"]
    ret.wheelSpeeds.rr = cp.vl["ABS_4"]["WHEEL_SPEED_RR"]
    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.01

    # Update steering angle and driver input torque
    ret.steeringAngleDeg = cp.vl["EPS_1"]["STEER_ANGLE"]
    # TODO: see if there's a signal to populate steeringRateDeg
    ret.steeringTorque = cp.vl["EPS_2"]["TORQUE_DRIVER"]
    # TODO: populate ret.steeringTorqueEps from one of the other EPS_2 signals, TBD
    ret.steeringPressed = abs(ret.steeringTorque) > P.STEER_THRESHOLD
    # FIXME: we have a candidate yawRate signal, but temporarily drive with back-calculated yawRate to check scaling
    self.VM.yaw_rate(ret.steeringAngleDeg * CV.DEG_TO_RAD, ret.vEgo)

    # Verify EPS readiness to accept steering commands
    # TODO: plot out enum/can_define for EPS error/warning states, taking original Tunder conditions for now
    ret.steerError = cp.vl["EPS_2"]["EPS_STATUS"] == 0 and ret.vEgo > self.CP.minSteerSpeed

    # Update gas, brakes, and gearshift
    ret.gas = cp.vl["TPS_1"]["THROTTLE_POSITION"]
    ret.gasPressed = ret.gas > 20
    ret.brake = cp.vl["ABS_1"]["BRAKE_PEDAL"]  # TODO: verify this is driver input only
    ret.brakePressed = ret.brake > 0  # FIXME: varies between vehicles maybe?

    # Update gear position
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(cp.vl["SHIFTER_ASSM"]["SHIFTER_POSITION"], None))

    # Update door and seatbelt status
    ret.doorOpen = any([cp.vl["BCM"]["DOOR_OPEN_FL"], cp.vl["BCM"]["DOOR_OPEN_FR"],
                        cp.vl["BCM"]["DOOR_OPEN_RL"], cp.vl["BCM"]["DOOR_OPEN_RR"]])
    ret.seatbeltUnlatched = bool(cp.vl["ORM"]["DRIVER_SEATBELT_STATUS"])

    # TODO: Can we find blindspot radar data?

    # Update cruise control states
    # TODO: there exists a CC-only, non-ACC mode, locate and populate cruiseState.nonAdaptive
    acc_state = cp_cam.vl["DASM_ACC_CMD_1"]["ACC_STATUS"]
    if acc_state == 1:
      # Ready but not engaged
      ret.cruiseState.available = True
      ret.cruiseState.enabled = False
    elif acc_state == 3:
      # Engaged
      ret.cruiseState.available = True
      ret.cruiseState.enabled = True
    else:
      # Unknown, or fault
      ret.cruiseState.available =  False
      ret.cruiseState.enabled = False

    ret.cruiseState.speed = cp_cam.vl["DASM_ACC_HUD"]["ACC_SET_SPEED_KPH"] * CV.KPH_TO_MS

    self.stock_cswc_values = cp_cam.vl["CSWC"]
    self.stock_lkas_hud_values = cp_cam.vl["DASM_LKAS_HUD"]

    # Update control button states for turn signals and ACC controls
    # TODO: read in ACC button states for DM reset and future long control
    ret.leftBlinker = bool(cp.vl["SCCM"]["BLINKER_LEFT"])
    ret.rightBlinker = bool(cp.vl["SCCM"]["BLINKER_RIGHT"])

    ret.espDisabled = bool(cp.vl["ICS"]["TRAC_OFF"])

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
      ("SHIFTER_POSITION", "SHIFTER_ASSM", 0),
      ("DOOR_OPEN_FL", "BCM", 0),
      ("DOOR_OPEN_FR", "BCM", 0),
      ("DOOR_OPEN_RL", "BCM", 0),
      ("DOOR_OPEN_RR", "BCM", 0),
      ("BRAKE_PEDAL", "ABS_1", 0),
      ("THROTTLE_POSITION", "TPS_1", 0),
      ("WHEEL_SPEED_FL", "ABS_4", 0),
      ("WHEEL_SPEED_FR", "ABS_4", 0),
      ("WHEEL_SPEED_RL", "ABS_4", 0),
      ("WHEEL_SPEED_RR", "ABS_4", 0),
      ("STEER_ANGLE", "EPS_1", 0),
      ("BLINKER_LEFT", "SCCM", 0),
      ("BLINKER_RIGHT", "SCCM", 0),
      ("TORQUE_DRIVER", "EPS_2", 0),
      ("EPS_STATUS", "EPS_2", 0),
      ("TRAC_OFF", "ICS", 0),
      ("DRIVER_SEATBELT_STATUS", "ORM", 0),
      ("COUNTER", "CSWC", 0),
      ("MAIN_SWITCH", "CSWC", 0),
      ("RESUME", "CSWC", 0),
      ("SET_PLUS", "CSWC", 0),
      ("SET_MINUS", "CSWC", 0),
      ("CANCEL", "CSWC", 0),
    ]

    checks = [
      # sig_address, frequency
      ("EPS_1", 100),
      ("EPS_2", 100),
      ("ABS_1", 50),
      ("ABS_4", 50),
      ("CSWC", 50),
      ("SHIFTER_ASSM", 50),
      ("TPS_1", 50),
      ("ICS", 20),
      ("SCCM", 10),
      ("ORM", 1),
      ("BCM", 1),
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address, default
      ("ACC_STATUS", "DASM_ACC_CMD_1", 0),
      ("ACC_SET_SPEED_KPH", "DASM_ACC_HUD", 0),
      ("UNKNOWN", "DASM_LKAS_HUD", 0),
      ("HIGH_BEAM_CONTROL", "DASM_LKAS_HUD", 0),
    ]
    checks = [
      ("DASM_ACC_CMD_1", 50),
      ("DASM_ACC_HUD", 50),
      ("DASM_LKAS_HUD", 15),
    ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)
