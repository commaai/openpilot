from selfdrive.car.interfaces import CarStateBase

class CarState(CarStateBase):

  @staticmethod
  def get_can_parser(CP):
    return None
