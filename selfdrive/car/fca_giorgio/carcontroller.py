from opendbc.can.packer import CANPacker
from openpilot.selfdrive.car.interfaces import CarControllerBase
from openpilot.selfdrive.car.fca_giorgio.values import CarControllerParams


class CarController(CarControllerBase):
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.CCP = CarControllerParams(CP)
    self.packer_pt = CANPacker(dbc_name)

    self.apply_steer_last = 0
    self.frame = 0

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    can_sends = []

    # **** Steering Controls ************************************************ #

    # **** Acceleration Controls ******************************************** #

    # **** HUD Controls ***************************************************** #

    # **** Stock ACC Button Controls **************************************** #

    new_actuators = actuators.copy()

    self.frame += 1
    return new_actuators, can_sends
