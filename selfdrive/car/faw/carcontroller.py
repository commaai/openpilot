import numpy as np
from cereal import car
from selfdrive.config import Conversions as CV
from selfdrive.car.interfaces import CarStateBase
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.car.volkswagen.values import DBC_FILES, CANBUS, NetworkLocation, TransmissionType, GearShifter, BUTTON_STATES, CarControllerParams

class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)

  def update(self, pt_cp, cam_cp, ext_cp, trans_type):
    ret = car.CarState.new_message()
    ret.wheelSpeeds = self.get_wheel_speeds(0,0,0,0)

    ret.vEgoRaw = pt_cp.vl["NEW_MGS_1"]["Car_Speed"]
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)

    return ret

  @staticmethod
  def get_can_parser(CP):
    # this function generates lists for signal, messages and initial values
    signals = [
      # sig_name, sig_address, default
      ("Car_Speed", "NEW_MSG_1", 0),       # Absolute steering angle
    ]

    checks = [

    ]

    return CANParser(DBC_FILES.mqb, signals, checks, CANBUS.pt)

  @staticmethod
  def get_cam_can_parser(CP):

    signals = []
    checks = []
    return CANParser(DBC_FILES.mqb, signals, checks, CANBUS.cam)
 