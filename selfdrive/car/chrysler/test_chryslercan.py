import unittest

from cereal import car
from selfdrive.can.packer import CANPacker
from selfdrive.car.chrysler import chryslercan

VisualAlert = car.CarControl.HUDControl.VisualAlert
GearShifter = car.CarState.GearShifter



class TestChryslerCan(unittest.TestCase):

  def test_checksum(self):
    self.assertEqual(0x75, chryslercan.calc_checksum(b"\x01\x20"))
    self.assertEqual(0xcc, chryslercan.calc_checksum(b"\x14\x00\x00\x00\x20"))

  def test_hud(self):
    packer = CANPacker('chrysler_pacifica_2017_hybrid')
    self.assertEqual(
        [0x2a6, 0, b'\x01\x00\x01\x01\x00\x00\x00\x00', 0],
        chryslercan.create_lkas_hud(
            packer,
            GearShifter.park, False, False, 1, 0))
    self.assertEqual(
        [0x2a6, 0, b'\x01\x00\x01\x00\x00\x00\x00\x00', 0],
        chryslercan.create_lkas_hud(
            packer,
            GearShifter.park, False, False, 5*4, 0))
    self.assertEqual(
        [0x2a6, 0, b'\x01\x00\x01\x00\x00\x00\x00\x00', 0],
        chryslercan.create_lkas_hud(
            packer,
            GearShifter.park, False, False, 99999, 0))
    self.assertEqual(
        [0x2a6, 0, b'\x02\x00\x06\x00\x00\x00\x00\x00', 0],
        chryslercan.create_lkas_hud(
            packer,
            GearShifter.drive, True, False, 99999, 0))
    self.assertEqual(
        [0x2a6, 0, b'\x02\x64\x06\x00\x00\x00\x00\x00', 0],
        chryslercan.create_lkas_hud(
            packer,
            GearShifter.drive, True, False, 99999, 0x64))

  def test_command(self):
    packer = CANPacker('chrysler_pacifica_2017_hybrid')
    self.assertEqual(
        [0x292, 0, b'\x14\x00\x00\x00\x10\x86', 0],
        chryslercan.create_lkas_command(
            packer,
            0, True, 1))
    self.assertEqual(
        [0x292, 0, b'\x04\x00\x00\x00\x80\x83', 0],
        chryslercan.create_lkas_command(
            packer,
            0, False, 8))


if __name__ == '__main__':
  unittest.main()
