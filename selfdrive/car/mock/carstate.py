from cereal import car
from openpilot.selfdrive.car.interfaces import CarStateBase


class CarState(CarStateBase):
  def update(self, *args) -> car.CarState:
    pass
