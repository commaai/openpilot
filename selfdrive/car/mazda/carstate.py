from cereal import car
from selfdrive.config import Conversions as CV
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.mazda.values import DBC, LKAS_LIMITS, CAR

GearShifter = car.CarState.GearShifter

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)

    self.cruise_speed = 0
    self.acc_active_last = False
    self.speed_kph = 0
    self.lkas_speed_lock = False
    self.low_speed_lockout = True
    self.low_speed_lockout_last = True
    self.acc_press_update = False

  def update(self, cp, cp_cam):

    ret = car.CarState.new_message()
    ret.wheelSpeeds.fl = cp.vl["WHEEL_SPEEDS"]['FL'] * CV.KPH_TO_MS
    ret.wheelSpeeds.fr = cp.vl["WHEEL_SPEEDS"]['FR'] * CV.KPH_TO_MS
    ret.wheelSpeeds.rl = cp.vl["WHEEL_SPEEDS"]['RL'] * CV.KPH_TO_MS
    ret.wheelSpeeds.rr = cp.vl["WHEEL_SPEEDS"]['RR'] * CV.KPH_TO_MS
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.01

    self.speed_kph =  ret.vEgoRaw  // CV.KPH_TO_MS

    gear = int(cp.vl["GEAR"]['GEAR']) >> 1

    if gear >= 1 & gear <= 5:
      ret.gearShifter = GearShifter.drive
    elif gear == 0:
      ret.gearShifter = GearShifter.park
    elif gear == 6:
      ret.gearShifter = GearShifter.reverse
    elif gear == 7:
      ret.gearShifter = GearShifter.neutral
    else:
      ret.gearShifter = GearShifter.unknown

    ret.leftBlinker = cp.vl["BLINK_INFO"]['LEFT_BLINK'] == 1
    ret.rightBlinker = cp.vl["BLINK_INFO"]['RIGHT_BLINK'] == 1

    ret.steeringAngle =  cp.vl["STEER"]['STEER_ANGLE']
    ret.steeringTorque = cp.vl["STEER_TORQUE"]['STEER_TORQUE_SENSOR']
    ret.steeringPressed = abs(ret.steeringTorque) > LKAS_LIMITS.STEER_THRESHOLD

    ret.steeringTorqueEps = cp.vl["STEER_TORQUE"]['STEER_TORQUE_MOTOR']
    ret.steeringRate = cp.vl["STEER_RATE"]['STEER_ANGLE_RATE']

    ret.brakePressed = cp.vl["PEDALS"]['BRAKE_ON'] == 1
    ret.brake = cp.vl["BRAKE"]['BRAKE_PRESSURE']
    ret.brakeLights = ret.brakePressed

    ret.seatbeltUnlatched = cp.vl["SEATBELT"]['DRIVER_SEATBELT'] == 0
    ret.doorOpen = any([cp.vl["DOORS"]['FL'], cp.vl["DOORS"]['FR'],
                        cp.vl["DOORS"]['BL'], cp.vl["DOORS"]['BR']])

    ret.gas = cp.vl["ENGINE_DATA"]['PEDAL_GAS']
    ret.gasPressed = ret.gas > 1e-3

    # No steer if block signal is on
    block = cp.vl["STEER_RATE"]['LKAS_BLOCK'] == 1

    # On if no driver torque the last 5 seconds
    handsoff = cp.vl["STEER_RATE"]['HANDS_OFF_5_SECONDS'] == 1

    # LKAS is enabled at 50kph going up and disabled at 45kph going down
    if self.speed_kph > LKAS_LIMITS.ENABLE_SPEED and self.low_speed_lockout:
      self.low_speed_lockout = False
    elif self.speed_kph < LKAS_LIMITS.DISABLE_SPEED and not self.low_speed_lockout:
      self.low_speed_lockout = True

    if (self.low_speed_lockout or block) and self.speed_kph < LKAS_LIMITS.DISABLE_SPEED:
        if not self.lkas_speed_lock:
          self.lkas_speed_lock = True
    elif self.lkas_speed_lock:
      self.lkas_speed_lock = False


    # if any of the cruize buttons is pressed force state update
    if any([cp.vl["CRZ_BTNS"]['RES'],
                cp.vl["CRZ_BTNS"]['SET_P'],
                cp.vl["CRZ_BTNS"]['SET_M']]):
      self.acc_active = True
      self.cruise_speed = self.speed_kph
      if self.low_speed_lockout_last:
        self.acc_press_update = True
    elif self.acc_press_update:
      self.acc_press_update = False

    ret.cruiseState.available = cp.vl["CRZ_CTRL"]['CRZ_ACTIVE'] == 1
    if not ret.cruiseState.available:
      self.acc_active = False

    if self.acc_active != self.acc_active_last:
      self.acc_active_last = self.acc_active

    ret.cruiseState.enabled = self.acc_active

    if ret.cruiseState.enabled:
      ret.cruiseState.speed = self.cruise_speed
      if self.lkas_speed_lock or handsoff:
        ret.steerWarning = True
    else:
      ret.cruiseState.speed = 0


    self.low_speed_lockout_last = self.low_speed_lockout

    self.cam_lkas = cp_cam.vl["CAM_LKAS"]
    self.steerError = cp_cam.vl["CAM_LKAS"]['ERR_BIT_1'] == 1

    return ret

  @staticmethod
  def get_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("LEFT_BLINK", "BLINK_INFO", 0),
      ("RIGHT_BLINK", "BLINK_INFO", 0),
      ("STEER_ANGLE", "STEER", 0),
      ("STEER_ANGLE_RATE", "STEER_RATE", 0),
      ("STEER_TORQUE_SENSOR", "STEER_TORQUE", 0),
      ("STEER_TORQUE_MOTOR", "STEER_TORQUE", 0),
      ("FL", "WHEEL_SPEEDS", 0),
      ("FR", "WHEEL_SPEEDS", 0),
      ("RL", "WHEEL_SPEEDS", 0),
      ("RR", "WHEEL_SPEEDS", 0),
    ]

    checks = [
      # sig_address, frequency
      ("BLINK_INFO", 10),
      ("STEER", 67),
      ("STEER_RATE", 83),
      ("STEER_TORQUE", 83),
      ("WHEEL_SPEEDS", 100),
    ]

    if CP.carFingerprint == CAR.CX5:
      signals += [
        ("LKAS_BLOCK", "STEER_RATE", 0),
        ("LKAS_TRACK_STATE", "STEER_RATE", 0),
        ("HANDS_OFF_5_SECONDS", "STEER_RATE", 0),
        ("CRZ_ACTIVE", "CRZ_CTRL", 0),
        ("STANDSTILL","PEDALS", 0),
        ("BRAKE_ON","PEDALS", 0),
        ("BRAKE_PRESSURE","BRAKE", 0),
        ("GEAR","GEAR", 0),
        ("DRIVER_SEATBELT", "SEATBELT", 0),
        ("FL", "DOORS", 0),
        ("FR", "DOORS", 0),
        ("BL", "DOORS", 0),
        ("BR", "DOORS", 0),
        ("PEDAL_GAS", "ENGINE_DATA", 0),
        ("RES", "CRZ_BTNS", 0),
        ("SET_P", "CRZ_BTNS", 0),
        ("SET_M", "CRZ_BTNS", 0),
      ]

      checks += [
        ("ENGINE_DATA", 100),
        ("CRZ_CTRL", 50),
        ("CRZ_BTNS", 10),
        ("PEDALS", 50),
        ("BRAKE", 50),
        ("SEATBELT", 10),
        ("DOORS", 10),
        ("GEAR", 20),
      ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 0)


  @staticmethod
  def get_cam_can_parser(CP):
    signals = [ ]
    checks = [ ]

    if CP.carFingerprint == CAR.CX5:
      signals += [
        # sig_name, sig_address, default
        ("LKAS_REQUEST",     "CAM_LKAS", 2048),
        ("CTR",              "CAM_LKAS", 0),
        ("ERR_BIT_1",        "CAM_LKAS", 0),
        ("LINE_NOT_VISIBLE", "CAM_LKAS", 0),
        ("LDW",              "CAM_LKAS", 0),
        ("BIT_1",            "CAM_LKAS", 1),
        ("ERR_BIT_2",        "CAM_LKAS", 0),
        ("LKAS_ANGLE",       "CAM_LKAS", 2048),
        ("BIT2",             "CAM_LKAS", 0),
        ("CHKSUM",           "CAM_LKAS", 0),
      ]

      checks += [
        # sig_address, frequency
        ("CAM_LKAS",      16),
      ]

    return CANParser(DBC[CP.carFingerprint]['pt'], signals, checks, 2)

