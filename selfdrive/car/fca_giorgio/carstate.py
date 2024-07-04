import numpy as np
from cereal import car
from openpilot.common.conversions import Conversions as CV
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
    ret.steeringTorque = pt_cp.vl["EPS_2"]["DRIVER_TORQUE"]
    ret.steeringTorqueEps = pt_cp.vl["EPS_3"]["EPS_TORQUE"]
    ret.steeringPressed = ret.steeringTorque > 80
    ret.yawRate = pt_cp.vl["ABS_2"]["YAW_RATE"]
    ret.steerFaultPermanent = bool(pt_cp.vl["EPS_2"]["LKA_FAULT"])

    # TODO: unsure if this is accel pedal or engine throttle
    #ret.gas = pt_cp.vl["ENGINE_1"]["ACCEL_PEDAL"]
    ret.gasPressed = ret.gas > 0
    ret.brake = pt_cp.vl["ABS_4"]["BRAKE_PRESSURE"]
    ret.brakePressed = bool(pt_cp.vl["ABS_3"]["BRAKE_PEDAL_SWITCH"])
    #ret.parkingBrake = TODO

    if bool(pt_cp.vl["ENGINE_1"]["REVERSE"]):
      ret.gearShifter = GearShifter.reverse
    else:
      ret.gearShifter = GearShifter.drive

    ret.cruiseState.available = pt_cp.vl["ACC_1"]["CRUISE_STATUS"] in (1, 2, 3)
    ret.cruiseState.enabled = pt_cp.vl["ACC_1"]["CRUISE_STATUS"] in (2, 3)
    ret.cruiseState.speed = pt_cp.vl["ACC_1"]["HUD_SPEED"] * CV.KPH_TO_MS

    ret.leftBlinker = bool(pt_cp.vl["BCM_1"]["LEFT_TURN_STALK"])
    ret.rightBlinker = bool(pt_cp.vl["BCM_1"]["RIGHT_TURN_STALK"])
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
      ("EPS_2", 100),
      ("EPS_3", 100),
      ("ACC_1", 12),  # 12hz inactive / 50hz active
      ("BCM_1", 4),  # 4Hz plus triggered updates
    ]

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CANBUS.pt)


  @staticmethod
  def get_cam_can_parser(CP):
    messages = []

    return CANParser(DBC[CP.carFingerprint]["pt"], messages, CANBUS.cam)
