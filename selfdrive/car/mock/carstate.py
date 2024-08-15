from cereal import car
from openpilot.selfdrive.car.interfaces import CarStateBase


class CarState(CarStateBase):
  def update(self, *_) -> car.CarState:
    return car.CarState.new_message()
