from openpilot.selfdrive.car import structs
from openpilot.selfdrive.car.interfaces import CarStateBase


class CarState(CarStateBase):
  def update(self, *_) -> structs.CarState:
    return structs.CarState()
