from cereal import car
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from selfdrive.car.body.values import DBC


class CarState(CarStateBase):
  def update(self, cp):
    ret = car.CarState.new_message()

    ret.wheelSpeeds.fl = cp.vl['MOTORS_DATA']['SPEED_L']
    ret.wheelSpeeds.fr = cp.vl['MOTORS_DATA']['SPEED_R']

    ret.vEgoRaw = ((ret.wheelSpeeds.fl + ret.wheelSpeeds.fr) / 2.) * self.CP.wheelSpeedFactor

    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = abs(ret.vEgo) < 1

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
      ("SPEED_L", "MOTORS_DATA"),
      ("SPEED_R", "MOTORS_DATA"),
      ("ELEC_ANGLE_L", "MOTORS_DATA"),
      ("ELEC_ANGLE_R", "MOTORS_DATA"),
      ("MOTOR_ERR_L", "MOTORS_DATA"),
      ("MOTOR_ERR_R", "MOTORS_DATA"),
      ("IGNITION", "VAR_VALUES"),
      ("ENABLE_MOTORS", "VAR_VALUES"),
      ("MCU_TEMP", "BODY_DATA"),
      ("BATT_VOLTAGE", "BODY_DATA"),
    ]

    checks = [
      ("MOTORS_DATA", 100),
      ("VAR_VALUES", 10),
      ("BODY_DATA", 1),
    ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 0)
