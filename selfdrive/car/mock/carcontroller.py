from openpilot.selfdrive.car.interfaces import CarControllerBase

class CarController(CarControllerBase):
  def update(self, CC, CS, now_nanos):
    return CC.actuators.copy(), []
