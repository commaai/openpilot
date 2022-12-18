from cereal import car
from common.conversions import Conversions as CV
from opendbc.can.can_define import CANDefine
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.mazda.values import DBC, LKAS_LIMITS, GEN1, GEN2, CarControllerParams
import copy

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)

    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["GEAR"]["GEAR"]

    self.crz_btns_counter = 0
    self.acc_active_last = False
    self.low_speed_alert = False
    self.lkas_allowed_speed = False
    self.lkas_disabled = False
    
    if CP.carFingerprint in GEN1:
      self.update = self.update_gen1
    if CP.carFingerprint in GEN2:
      self.update = self.update_gen2
      
    
    self.params = CarControllerParams(CP)
    
  def update_gen2(self, cp, cp_cam, cp_body):
    ret = car.CarState.new_message()
    ret.wheelSpeeds = self.get_wheel_speeds(
        cp_cam.vl["WHEEL_SPEEDS"]["FL"],
        cp_cam.vl["WHEEL_SPEEDS"]["FR"],
        cp_cam.vl["WHEEL_SPEEDS"]["RL"],
        cp_cam.vl["WHEEL_SPEEDS"]["RR"],
    )

    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(100, cp.vl["BLINK_INFO"]["LEFT_BLINK"] == 1,
                                                                      cp.vl["BLINK_INFO"]["RIGHT_BLINK"] == 1)

    ret.steeringAngleDeg = cp_cam.vl["STEER"]["STEER_ANGLE"]
    ret.steeringTorque = cp_body.vl["EPS_FEEDBACK"]["STEER_TORQUE_SENSOR"]
    can_gear = int(cp_cam.vl["GEAR"]["GEAR"])
    ret.gas = cp_cam.vl["ENGINE_DATA"]["PEDAL_GAS"]

    ret.steeringPressed = abs(ret.steeringTorque) > self.params.STEER_DRIVER_ALLOWANCE
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))
    # TODO: this should be from 0 - 1.
    ret.gasPressed = ret.gas > 0
    ret.seatbeltUnlatched = False
    ret.doorOpen = False
    ret.brakePressed = cp.vl["BRAKE_PEDAL"]["BRAKE_PEDAL_PRESSED"] == 1
    ret.brake = .1
    ret.steerFaultPermanent = False
    ret.steerFaultTemporary = False
    ret.cruiseState.available = True
    ret.cruiseState.speed = cp.vl["CRUZE_STATE"]["CRZ_SPEED"] * CV.KPH_TO_MS
    self.cam_lkas = 1
    self.cam_laneinfo = 1
    ret.cruiseState.enabled = ( (cp.vl["CRUZE_STATE"]["CRZ_ENABLED"] == 1) or (cp.vl["CRUZE_STATE"]["PRE_ENABLE"] == 1) )
    self.acc_active_last = ret.cruiseState.enabled
    self.crz_btns_counter = cp.vl["CRZ_BTNS"]["CTR"]

    speed_kph = cp_cam.vl["SPEED"]["SPEED"] * CV.KPH_TO_MS
    ret.standstill = speed_kph < .1
    ret.cruiseState.standstill = ret.standstill
    self.cp = cp
    self.cp_cam = cp_cam
    self.acc = cp.vl["ACC"]
    

    # end GEN2

    return ret
    

  def update_gen1(self, cp, cp_cam, _):

    ret = car.CarState.new_message()
    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["WHEEL_SPEEDS"]["FL"],
      cp.vl["WHEEL_SPEEDS"]["FR"],
      cp.vl["WHEEL_SPEEDS"]["RL"],
      cp.vl["WHEEL_SPEEDS"]["RR"],
    )
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    # Match panda speed reading
    speed_kph = cp.vl["ENGINE_DATA"]["SPEED"]
    ret.standstill = speed_kph < .1

    can_gear = int(cp.vl["GEAR"]["GEAR"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.genericToggle = bool(cp.vl["BLINK_INFO"]["HIGH_BEAMS"])
    ret.leftBlindspot = cp.vl["BSM"]["LEFT_BS1"] == 1
    ret.rightBlindspot = cp.vl["BSM"]["RIGHT_BS1"] == 1
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(40, cp.vl["BLINK_INFO"]["LEFT_BLINK"] == 1,
                                                                      cp.vl["BLINK_INFO"]["RIGHT_BLINK"] == 1)

    ret.steeringAngleDeg = cp.vl["STEER"]["STEER_ANGLE"]
    ret.steeringTorque = cp.vl["STEER_TORQUE"]["STEER_TORQUE_SENSOR"]
    ret.steeringPressed = abs(ret.steeringTorque) > LKAS_LIMITS.STEER_THRESHOLD

    ret.steeringTorqueEps = cp.vl["STEER_TORQUE"]["STEER_TORQUE_MOTOR"]
    ret.steeringRateDeg = cp.vl["STEER_RATE"]["STEER_ANGLE_RATE"]

    # TODO: this should be from 0 - 1.
    ret.brakePressed = cp.vl["PEDALS"]["BRAKE_ON"] == 1
    ret.brake = cp.vl["BRAKE"]["BRAKE_PRESSURE"]

    ret.seatbeltUnlatched = cp.vl["SEATBELT"]["DRIVER_SEATBELT"] == 0
    ret.doorOpen = any([cp.vl["DOORS"]["FL"], cp.vl["DOORS"]["FR"],
                        cp.vl["DOORS"]["BL"], cp.vl["DOORS"]["BR"]])

    # TODO: this should be from 0 - 1.
    ret.gas = cp.vl["ENGINE_DATA"]["PEDAL_GAS"]
    ret.gasPressed = ret.gas > 0

    # Either due to low speed or hands off
    lkas_blocked = cp.vl["STEER_RATE"]["LKAS_BLOCK"] == 1

    if self.CP.minSteerSpeed > 0:
      # LKAS is enabled at 52kph going up and disabled at 45kph going down
      # wait for LKAS_BLOCK signal to clear when going up since it lags behind the speed sometimes
      if speed_kph > LKAS_LIMITS.ENABLE_SPEED and not lkas_blocked:
        self.lkas_allowed_speed = True
      elif speed_kph < LKAS_LIMITS.DISABLE_SPEED:
        self.lkas_allowed_speed = False
    else:
      self.lkas_allowed_speed = True

    # TODO: the signal used for available seems to be the adaptive cruise signal, instead of the main on
    #       it should be used for carState.cruiseState.nonAdaptive instead
    ret.cruiseState.available = cp.vl["CRZ_CTRL"]["CRZ_AVAILABLE"] == 1
    ret.cruiseState.enabled = cp.vl["CRZ_CTRL"]["CRZ_ACTIVE"] == 1
    ret.cruiseState.standstill = cp.vl["PEDALS"]["STANDSTILL"] == 1
    ret.cruiseState.speed = cp.vl["CRZ_EVENTS"]["CRZ_SPEED"] * CV.KPH_TO_MS

    if ret.cruiseState.enabled:
      if not self.lkas_allowed_speed and self.acc_active_last:
        self.low_speed_alert = True
      else:
        self.low_speed_alert = False

    # Check if LKAS is disabled due to lack of driver torque when all other states indicate
    # it should be enabled (steer lockout). Don't warn until we actually get lkas active
    # and lose it again, i.e, after initial lkas activation
    ret.steerFaultTemporary = self.lkas_allowed_speed and lkas_blocked

    self.acc_active_last = ret.cruiseState.enabled

    self.crz_btns_counter = cp.vl["CRZ_BTNS"]["CTR"]

    # camera signals
    self.lkas_disabled = cp_cam.vl["CAM_LANEINFO"]["LANE_LINES"] == 0
    self.cam_lkas = cp_cam.vl["CAM_LKAS"]
    self.cam_laneinfo = cp_cam.vl["CAM_LANEINFO"]
    ret.steerFaultPermanent = cp_cam.vl["CAM_LKAS"]["ERR_BIT_1"] == 1

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = []
    checks = []
    if CP.carFingerprint not in GEN2:
      signals = [
        # sig_name, sig_address
        ("LEFT_BLINK", "BLINK_INFO"),
        ("RIGHT_BLINK", "BLINK_INFO"),
        ("HIGH_BEAMS", "BLINK_INFO"),
        ("STEER_ANGLE", "STEER"),
        ("STEER_ANGLE_RATE", "STEER_RATE"),
        ("STEER_TORQUE_SENSOR", "STEER_TORQUE"),
        ("STEER_TORQUE_MOTOR", "STEER_TORQUE"),
        ("FL", "WHEEL_SPEEDS"),
        ("FR", "WHEEL_SPEEDS"),
        ("RL", "WHEEL_SPEEDS"),
        ("RR", "WHEEL_SPEEDS"),
      ]

      checks = [
        # sig_address, frequency
        ("BLINK_INFO", 10),
        ("STEER", 67),
        ("STEER_RATE", 83),
        ("STEER_TORQUE", 83),
        ("WHEEL_SPEEDS", 100),
      ]

    if CP.carFingerprint in GEN1:
      signals += [
        ("LKAS_BLOCK", "STEER_RATE"),
        ("LKAS_TRACK_STATE", "STEER_RATE"),
        ("HANDS_OFF_5_SECONDS", "STEER_RATE"),
        ("CRZ_ACTIVE", "CRZ_CTRL"),
        ("CRZ_AVAILABLE", "CRZ_CTRL"),
        ("CRZ_SPEED", "CRZ_EVENTS"),
        ("STANDSTILL", "PEDALS"),
        ("BRAKE_ON", "PEDALS"),
        ("BRAKE_PRESSURE", "BRAKE"),
        ("GEAR", "GEAR"),
        ("DRIVER_SEATBELT", "SEATBELT"),
        ("FL", "DOORS"),
        ("FR", "DOORS"),
        ("BL", "DOORS"),
        ("BR", "DOORS"),
        ("PEDAL_GAS", "ENGINE_DATA"),
        ("SPEED", "ENGINE_DATA"),
        ("CTR", "CRZ_BTNS"),
        ("LEFT_BS1", "BSM"),
        ("RIGHT_BS1", "BSM"),
      ]

      checks += [
        ("ENGINE_DATA", 100),
        ("CRZ_CTRL", 50),
        ("CRZ_EVENTS", 50),
        ("CRZ_BTNS", 10),
        ("PEDALS", 50),
        ("BRAKE", 50),
        ("SEATBELT", 10),
        ("DOORS", 10),
        ("GEAR", 20),
        ("BSM", 10),
      ]

    if CP.carFingerprint in GEN2:
      signals += [
        ("CRZ_SPEED", "CRUZE_STATE", 0),
        ("CRZ_ENABLED", "CRUZE_STATE", 0),
        ("PRE_ENABLE", "CRUZE_STATE", 0),
        ("BRAKE_PEDAL_PRESSED", "BRAKE_PEDAL", 0),
        ("SET_P", "CRZ_BTNS",0),
        ("SET_M", "CRZ_BTNS",0),
        ("RES", "CRZ_BTNS",0),
        ("CAN", "CRZ_BTNS",0),
        ("MODE_ACC", "CRZ_BTNS",0),
        ("MODE_LKAS", "CRZ_BTNS",0),
        ("DISTANCE_P", "CRZ_BTNS",0),
        ("DISTANCE_M", "CRZ_BTNS",0),
        ("STATIC_0", "CRZ_BTNS",0),
        ("STATIC_1", "CRZ_BTNS",0),
        ("STATIC_2", "CRZ_BTNS",0),
        ("CTR", "CRZ_BTNS", 0),
        ("LEFT_BLINK", "BLINK_INFO", 0),
        ("RIGHT_BLINK", "BLINK_INFO", 0),
        ("ACCEL_CMD", "ACC", 0),
        ("HOLD", "ACC", 0),
        ("RESUME", "ACC", 0),
        ("NEW_SIGNAL_1", "ACC", 0),
        ("NEW_SIGNAL_2", "ACC", 0),
        ("NEW_SIGNAL_3", "ACC", 0),
        ("NEW_SIGNAL_4", "ACC", 0),
        ("NEW_SIGNAL_5", "ACC", 0),
        ("NEW_SIGNAL_6", "ACC", 0),
        ("NEW_SIGNAL_7", "ACC", 0),
        ("NEW_SIGNAL_8", "ACC", 0),
        ("NEW_SIGNAL_9", "ACC", 0),
        ("NEW_SIGNAL_10", "ACC", 0),
        ("NEW_SIGNAL_11", "ACC", 0),
        ("NEW_SIGNAL_12", "ACC", 0),
        ("NEW_SIGNAL_13", "ACC", 0),
        ("ACC_ENABLED", "ACC", 0),
        ("ACC_ENABLED_2", "ACC", 0),
        ("CHECKSUM", "ACC", 0), 
      ]

      checks += [
        ("BRAKE_PEDAL", 20),
        ("CRUZE_STATE", 10),
        ("BLINK_INFO", 10),
        ("ACC", 50),
        ("CRZ_BTNS", 10),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 0)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = []
    checks = []

    if CP.carFingerprint in GEN1:
      signals += [
        # sig_name, sig_address
        ("LKAS_REQUEST", "CAM_LKAS"),
        ("CTR", "CAM_LKAS"),
        ("ERR_BIT_1", "CAM_LKAS"),
        ("LINE_NOT_VISIBLE", "CAM_LKAS"),
        ("BIT_1", "CAM_LKAS"),
        ("ERR_BIT_2", "CAM_LKAS"),
        ("STEERING_ANGLE", "CAM_LKAS"),
        ("ANGLE_ENABLED", "CAM_LKAS"),
        ("CHKSUM", "CAM_LKAS"),
        ("LINE_VISIBLE", "CAM_LANEINFO"),
        ("LINE_NOT_VISIBLE", "CAM_LANEINFO"),
        ("LANE_LINES", "CAM_LANEINFO"),
        ("BIT1", "CAM_LANEINFO"),
        ("BIT2", "CAM_LANEINFO"),
        ("BIT3", "CAM_LANEINFO"),
        ("NO_ERR_BIT", "CAM_LANEINFO"),
        ("S1", "CAM_LANEINFO"),
        ("S1_HBEAM", "CAM_LANEINFO"),
      ]

      checks += [
        # sig_address, frequency
        ("CAM_LANEINFO", 2),
        ("CAM_LKAS", 16),
      ]

    if CP.carFingerprint in GEN2:
      signals += [
        ("STEER_TORQUE_SENSOR", "STEER_TORQUE", 0),
        ("GEAR", "GEAR", 0),
        ("PEDAL_GAS", "ENGINE_DATA", 0),
        ("FL", "WHEEL_SPEEDS", 0),
        ("FR", "WHEEL_SPEEDS", 0),
        ("RL", "WHEEL_SPEEDS", 0),
        ("RR", "WHEEL_SPEEDS", 0),
        ("STEER_ANGLE", "STEER", 0),
        ("SPEED", "SPEED", 0),
      ]
      checks += [
        ("ENGINE_DATA", 100),
        ("STEER_TORQUE", 100),
        ("GEAR", 40),
        ("WHEEL_SPEEDS", 100),
        ("STEER", 100),
        ("SPEED", 50),
      ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)

  @staticmethod
  def get_body_can_parser(CP):
    signals = []
    checks = []
    if CP.carFingerprint in GEN2:
      signals += [
        ("STEER_TORQUE_SENSOR", "EPS_FEEDBACK", 0),
      ]
      checks += [
        ("EPS_FEEDBACK", 50),
      ]
      
    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 1)
