import numpy as np
from cereal import car
from openpilot.selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from openpilot.selfdrive.car.fca_giorgio.values import DBC, CANBUS, CarControllerParams


GearShifter = car.CarState.GearShifter

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.frame = 0
    self.CCP = CarControllerParams(CP)


  def update(self, pt_cp, cam_cp):
    ret = car.CarState.new_message()
    # Update vehicle speed and acceleration from ABS wheel speeds.
    ret.wheelSpeeds = self.get_wheel_speeds(
      pt_cp.vl["ABS_1"]["WHEEL_SPEED_FL"],
      pt_cp.vl["ABS_1"]["WHEEL_SPEED_FR"],
      pt_cp.vl["ABS_1"]["WHEEL_SPEED_RL"],
      pt_cp.vl["ABS_1"]["WHEEL_SPEED_RR"],
      unit=1.0
    )

    ret.vEgoRaw = float(np.mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr]))
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = ret.vEgoRaw == 0

    ret.steeringAngleDeg = pt_cp.vl["EPS_1"]["STEERING_ANGLE"]
    ret.steeringRateDeg = pt_cp.vl["EPS_1"]["STEERING_RATE"]
    #ret.steeringTorque = TODO
    #ret.steeringPressed = TODO
    ret.yawRate = pt_cp.vl["ABS_2"]["YAW_RATE"]
    #ret.steerFaultTemporary, ret.steerFaultPermanent = TODO, TODO

    ret.gas = pt_cp.vl["ENGINE_1"]["ACCEL_PEDAL"]
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["ABS_4"]["BRAKE_PRESSURE"]
    ret.brakePressed = bool(pt_cp.vl["ABS_3"]["BRAKE_PEDAL_SWITCH"])
    #ret.parkingBrake = TODO

    ret.gearShifter = GearShifter.drive  # TODO

    # ret.doorOpen = TODO
    # ret.seatbeltUnlatched = TODO

    # ret.cruiseState.available = TODO
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
      # sig_address, frequency
      ("ABS_1", 100),
      ("ABS_2", 100),
      ("ABS_3", 100),
      ("ABS_4", 100),
      ("ENGINE_1", 100),
      ("EPS_1", 100),
    ]

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CANBUS.pt)


  @staticmethod
  def get_cam_can_parser(CP):
    messages = []

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CANBUS.cam)
