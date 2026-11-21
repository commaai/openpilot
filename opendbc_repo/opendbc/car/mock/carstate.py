from opendbc.car import structs
from opendbc.car.interfaces import CarStateBase


class CarState(CarStateBase):
  def update(self, can_parsers) -> structs.CarState:
    return structs.CarState()
