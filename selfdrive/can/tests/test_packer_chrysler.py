import unittest
import random

from selfdrive.can.tests.packer_old import CANPacker as CANPackerOld
from selfdrive.can.packer import CANPacker
import selfdrive.car.chrysler.chryslercan as chryslercan


class TestPackerMethods(unittest.TestCase):
  def setUp(self):
    self.chrysler_cp_old = CANPackerOld("chrysler_pacifica_2017_hybrid")
    self.chrysler_cp = CANPacker("chrysler_pacifica_2017_hybrid")

  def test_correctness(self):
    # Test all commands, randomize the params.
    for _ in range(1000):
      gear = ('drive', 'reverse', 'low')[random.randint(0, 3) % 3]
      lkas_active = (random.randint(0, 2) % 2 == 0)
      hud_alert = random.randint(0, 6)
      hud_count = random.randint(0, 65536)
      lkas_car_model = random.randint(0, 65536)
      m_old = chryslercan.create_lkas_hud(self.chrysler_cp_old, gear, lkas_active, hud_alert, hud_count, lkas_car_model)
      m = chryslercan.create_lkas_hud(self.chrysler_cp, gear, lkas_active, hud_alert, hud_count, lkas_car_model)
      self.assertEqual(m_old, m)

      apply_steer = (random.randint(0, 2) % 2 == 0)
      moving_fast = (random.randint(0, 2) % 2 == 0)
      frame = random.randint(0, 65536)
      m_old = chryslercan.create_lkas_command(self.chrysler_cp_old, apply_steer, moving_fast, frame)
      m = chryslercan.create_lkas_command(self.chrysler_cp, apply_steer, moving_fast, frame)
      self.assertEqual(m_old, m)


if __name__ == "__main__":
  unittest.main()
