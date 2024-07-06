import numpy as np
from cereal import car
from openpilot.selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from openpilot.selfdrive.car.gwm.values import DBC, CANBUS, CarControllerParams


GearShifter = car.CarState.GearShifter

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.frame = 0
    self.CPP = CarControllerParams(CP)

  def update(self, pt_cp, cam_cp):
    ret = car.CarState.new_message()

    # begin TODO clean-after-port
    ret.wheelSpeeds = self.get_wheel_speeds(10,10,10,10,unit=1.0)
    # end TODO clean-after-port

    # ret.wheelSpeeds = self.get_wheel_speeds(
    #   pt_cp.vl["WHEEL_SPEEDS"]["FRONT_LEFT_WHEEL_SPEED"],
    #   pt_cp.vl["WHEEL_SPEEDS"]["FRONT_RIGHT_WHEEL_SPEED"],
    #   pt_cp.vl["WHEEL_SPEEDS"]["REAR_LEFT_WHEEL_SPEED"],
    #   pt_cp.vl["WHEEL_SPEEDS"]["REAR_RIGHT_WHEEL_SPEED"],
    # )
    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = abs(ret.vEgoRaw) < 1e-3

    ret.steeringAngleDeg = 0 # TODO
    ret.steeringRateDeg = 0 # TODO
    # ret.steeringTorque = TODO
    # ret.steeringPressed = TODO
    # ret.yawRate = NOT ABSOLUTE NECESSARY
    # ret.steerFaultTemporary, ret.steerFaultPermanent = CRITICAL SAFETY TODO, CRITICAL SAFETY TODO

    ret.gas = 0 # TODO
    ret.gasPressed = ret.gas > 0
    ret.brake = 0
    ret.brakePressed = 0
    # ret.parkingBrake = TODO

    # begin toyota brakePressed TODO clean-after-port
    ret.brakePressed = cp.vl["BRAKE_MODULE"]["BRAKE_PRESSED"] != 0
    # end TODO clean-after-port

    ret.gearShifter = GearShifter.drive # TODO

    # ret.doorOpen = TODO
    # ret.seatbeltUnlatched = TODO

    # ret.cruiseState.available = CRITICAL SAFETY TODO
    # ret.cruiseState.enabled = TODO
    # ret.cruiseState.speed = TODO

    # ret.leftBlinker = TODO
    # ret.rightBlinker = TODO
    # ret.buttonEvents = TODO
    # ret.espDisabled = TODO

    self.frame += 1
    return ret


@staticmethod
def get_can_parser(CP):
  messages = [
    # seg_address, frequency

    # begin toyota TODO clean-after-port
    ("BRAKE_MODULE", 40),
    # end TODO clean-after-port

  ]

  return CANParser(DBC[CP.carFingerprint]["pt"], messages, CANBUS.pt)


@staticmethod
def get_cam_can_parser(CP):
  messages = []

  return CANParser(DBC[CP.carFingerprint]["pt"], messages, CANBUS.cam)
