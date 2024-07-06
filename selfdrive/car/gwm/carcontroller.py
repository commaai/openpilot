from opendbc.can.packer import CANPacker
from openpilot.selfdrive.car.interfaces import CarControllerBase
from openpilot.selfdrive.car.gwm.values import CarControllerParams

class CarController(CarControllerBase):
  def __init__(self, dbc_name: str, CP, VM):
    self.CP = CP
    self.CCP = CarControllerParams(CP)
    self.packer_pt = CANPacker(dbc_name)

    self.frame = 0

  def update(self, CC, CS, now_nanos):
    actuators = CC.actuators
    can_sends = []

    # **** Steering Controls ******************************************* #
    # TODO

    # **** Acceleration Controls *************************************** #
    # TODO

    # **** HUD Controls ************************************************ #
    # TODO

    # **** Stock ACC Button Controls *********************************** #
    # TODO

    new_actuators = actuators.copy()

    self.frame += 1
    return new_actuators, can_sends
