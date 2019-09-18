from selfdrive.car.chrysler import chryslercan
from selfdrive.can.packer import CANPacker

from cereal import car
VisualAlert = car.CarControl.HUDControl.VisualAlert

import unittest


class TestChryslerCan(unittest.TestCase):

  def test_checksum(self):
    self.assertEqual(0x75, chryslercan.calc_checksum([0x01, 0x20]))
    self.assertEqual(0xcc, chryslercan.calc_checksum([0x14, 0, 0, 0, 0x20]))

  def test_hud(self):
    packer = CANPacker('chrysler_pacifica_2017_hybrid')
    self.assertEqual(
        [0x2a6, 0, '0100010100000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(
            packer,
            'park', False, False, 1, 0))
    self.assertEqual(
        [0x2a6, 0, '0100010000000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(
            packer,
            'park', False, False, 5*4, 0))
    self.assertEqual(
        [0x2a6, 0, '0100010000000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(
            packer,
            'park', False, False, 99999, 0))
    self.assertEqual(
        [0x2a6, 0, '0200060000000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(
            packer,
            'drive', True, False, 99999, 0))
    self.assertEqual(
        [0x2a6, 0, '0264060000000000'.decode('hex'), 0],
        chryslercan.create_lkas_hud(
            packer,
            'drive', True, False, 99999, 0x64))

  def test_command(self):
    packer = CANPacker('chrysler_pacifica_2017_hybrid')
    self.assertEqual(
        [0x292, 0, '140000001086'.decode('hex'), 0],
        chryslercan.create_lkas_command(
            packer,
            0, True, 1))
    self.assertEqual(
        [0x292, 0, '040000008083'.decode('hex'), 0],
        chryslercan.create_lkas_command(
            packer,
            0, False, 8))


if __name__ == '__main__':
  unittest.main()
