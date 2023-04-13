from cereal import car
from common.conversions import Conversions as CV
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.simulator.values import DBC


GearShifter = car.CarState.GearShifter

class CarState(CarStateBase):
  def update(self, cp):

    ret = car.CarState.new_message()
    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["ENGINE_DATA"]["SPEED"],
      cp.vl["ENGINE_DATA"]["SPEED"],
      cp.vl["ENGINE_DATA"]["SPEED"],
      cp.vl["ENGINE_DATA"]["SPEED"]
    )
    ret.vEgoRaw = (ret.wheelSpeeds.fl + ret.wheelSpeeds.fr + ret.wheelSpeeds.rl + ret.wheelSpeeds.rr) / 4.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    # Match panda speed reading
    speed_kph = cp.vl["ENGINE_DATA"]["SPEED"]
    ret.standstill = speed_kph < .1

    ret.gearShifter = GearShifter.drive

    
    ret.leftBlindspot = False
    ret.rightBlindspot = False
    ret.leftBlinker, ret.rightBlinker = False, False

    ret.steeringAngleDeg = cp.vl["STEER"]["STEER_ANGLE"]
    ret.steeringTorque = 0
    ret.steeringPressed = False

    ret.steeringTorqueEps = 0
    ret.steeringRateDeg = 0

    ret.brakePressed = False
    ret.brake = 0

    ret.seatbeltUnlatched = False
    ret.doorOpen = False

    ret.gas = 0
    ret.gasPressed = False

    ret.cruiseState.available = True
    ret.cruiseState.enabled = cp.vl["CRZ_CTRL"]["CRZ_ACTIVE"] == 1
    ret.cruiseState.standstill = cp.vl["PEDALS"]["STANDSTILL"] == 1
    ret.cruiseState.speed = cp.vl["CRZ_EVENTS"]["CRZ_SPEED"] * CV.KPH_TO_MS
    ret.steerFaultTemporary = False
    ret.steerFaultPermanent = False

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("STEER_ANGLE", "STEER"),
      ("SPEED", "ENGINE_DATA"),
      ("CRZ_ACTIVE", "CRZ_CTRL"),
      ("CRZ_SPEED", "CRZ_EVENTS"),
      ("STANDSTILL", "PEDALS"),
    ]
    checks = [
      # sig_address, frequency

      ("STEER", 50),
      ("ENGINE_DATA", 50),
      ("CRZ_CTRL", 50),
      ("CRZ_EVENTS", 50),
      ("PEDALS", 50),
    ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 0)