import copy
from openpilot.selfdrive.car.interfaces import CarControllerBase

class CarController(CarControllerBase):
  def update(self, CC, CS, now_nanos):
    return copy.deepcopy(CC.actuators), []
