import chryslercan
from values import CAR
from carcontroller import CarController
from selfdrive.can.packer import CANPacker

from cereal import car
VisualAlert = car.CarControl.HUDControl.VisualAlert
AudibleAlert = car.CarControl.HUDControl.AudibleAlert

import unittest


class TestChryslerCan(unittest.TestCase):

  def test_checksum(self):
    self.assertEqual(0x75, chryslercan.calc_checksum([0x01, 0x20]))
    self.assertEqual(0xcc, chryslercan.calc_checksum([0x14, 0, 0, 0, 0x20]))

  def test_heartbit(self):
    self.assertEqual(
        [0x2d9, 0, '0000000820'.decode('hex'), 0],
        chryslercan.create_lkas_heartbit(CAR.PACIFICA_2017_HYBRID))

  def test_hud(self):
    packer = CANPacker('chrysler_pacifica_2017_hybrid')
    self.assertEqual(
        [0x2a6, 0, '0000010100000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(packer,
            'park', False, False, CAR.PACIFICA_2017_HYBRID, 1))
    self.assertEqual(
        [0x2a6, 0, '0000010000000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(packer,
            'park', False, False, CAR.PACIFICA_2017_HYBRID, 5*4))
    self.assertEqual(
        [0x2a6, 0, '0000000000000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(packer,
            'park', False, False, CAR.PACIFICA_2017_HYBRID, 99999))
    self.assertEqual(
        [0x2a6, 0, '0200060000000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(packer,
            'drive', True, False, CAR.PACIFICA_2017_HYBRID, 99999))
    self.assertEqual(
        [0x2a6, 0, '0264060000000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(packer,
            'drive', True, False, CAR.PACIFICA_2018, 99999))

if __name__ == '__main__':
  unittest.main()
