from cereal import car
from opendbc.can.can_define import CANDefine
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.car.body.values import DBC

TORQUE_SAMPLES = 12

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])

  def update(self, cp, cp_cam):
    ret = car.CarState.new_message()

    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["BODY_SENSOR"]["SPEED_L"],
      cp.vl["BODY_SENSOR"]["SPEED_R"],
      cp.vl["BODY_SENSOR"]["SPEED_L"],
      cp.vl["BODY_SENSOR"]["SPEED_R"],
    )
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.

    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw < 0.01

    ret.cruiseState.enabled = True
    ret.cruiseState.available = True
    ret.seatbeltUnlatched = False

    ret.cruiseState.speed = 0
    ret.steeringTorque = 0

    self.steeringTorqueSamples.append(ret.steeringTorque)

    ret.steeringPressed = False

    ret.steeringAngleDeg = 0

    ret.doorOpen = False

    ret.gearShifter = self.parse_gear_shifter("D")

    self.lkas_enabled = True

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("SPEED_L", "BODY_SENSOR"),
      ("SPEED_R", "BODY_SENSOR"),
      ("BAT_VOLTAGE", "BODY_SENSOR"),
    ]

    checks = []

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 1)


  @staticmethod
  def get_cam_can_parser(CP):
    signals = []
    checks = []

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 1)
