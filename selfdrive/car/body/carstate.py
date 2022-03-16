from cereal import car
from opendbc.can.can_define import CANDefine
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.car.body.values import DBC


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)

  def update(self, cp):
    ret = car.CarState.new_message()

    ret.wheelSpeeds.fl = cp.vl['BODY_SENSOR']['SPEED_L']
    ret.wheelSpeeds.fr = cp.vl['BODY_SENSOR']['SPEED_R']

    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr) / 2.

    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = abs(ret.vEgo) < 100

    ret.cruiseState.enabled = True
    ret.cruiseState.available = True
    ret.seatbeltUnlatched = False

    ret.cruiseState.speed = 0
    ret.steeringTorque = 0

    ret.steeringPressed = False

    ret.steeringAngleDeg = 0

    ret.doorOpen = False

    ret.gearShifter = self.parse_gear_shifter("D")

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("SPEED_L", "BODY_SENSOR"),
      ("SPEED_R", "BODY_SENSOR"),
      ("BAT_VOLTAGE", "BODY_SENSOR"),
    ]

    checks = [
      ("BODY_SENSOR", 20),
    ]

    return CANParser(DBC[CP.carFingerprint][], signals, checks, 1)
