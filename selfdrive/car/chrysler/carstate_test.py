from selfdrive.car.chrysler import carstate
from selfdrive.car.chrysler.interface import CarInterface
from selfdrive.car.chrysler.values import CAR


import unittest


class TestCarstate(unittest.TestCase):

  def test_get_can_parser(self):
    CP = CarInterface.get_params(CAR.PACIFICA_2017_HYBRID, {})
    cpNew = carstate.get_can_parser(CP)
    cpCamNew = carstate.get_camera_parser(CP)
    CS = carstate.CarState(CP)
    CS.update(cpNew, cpCamNew)


if __name__ == '__main__':
  unittest.main()
