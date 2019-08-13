import unittest
import random

from selfdrive.can.tests.packer_old import CANPacker as CANPackerOld
from selfdrive.can.packer import CANPacker
import selfdrive.car.honda.hondacan as hondacan
from selfdrive.car.honda.values import HONDA_BOSCH
from selfdrive.car.honda.carcontroller import HUDData


class TestPackerMethods(unittest.TestCase):
  def setUp(self):
    self.honda_cp_old = CANPackerOld("honda_pilot_touring_2017_can_generated")
    self.honda_cp = CANPacker("honda_pilot_touring_2017_can_generated")

  def test_correctness(self):
    # Test all commands, randomize the params.
    for _ in xrange(1000):
      is_panda_black = False
      car_fingerprint = HONDA_BOSCH[0]

      apply_brake = (random.randint(0, 2) % 2 == 0)
      pump_on = (random.randint(0, 2) % 2 == 0)
      pcm_override = (random.randint(0, 2) % 2 == 0)
      pcm_cancel_cmd = (random.randint(0, 2) % 2 == 0)
      fcw = random.randint(0, 65536)
      idx = random.randint(0, 65536)
      m_old = hondacan.create_brake_command(self.honda_cp_old, apply_brake, pump_on, pcm_override, pcm_cancel_cmd, fcw, idx, car_fingerprint, is_panda_black)
      m = hondacan.create_brake_command(self.honda_cp, apply_brake, pump_on, pcm_override, pcm_cancel_cmd, fcw, idx, car_fingerprint, is_panda_black)
      self.assertEqual(m_old, m)

      apply_steer = (random.randint(0, 2) % 2 == 0)
      lkas_active = (random.randint(0, 2) % 2 == 0)
      idx = random.randint(0, 65536)
      m_old = hondacan.create_steering_control(self.honda_cp_old, apply_steer, lkas_active, car_fingerprint, idx, is_panda_black)
      m = hondacan.create_steering_control(self.honda_cp, apply_steer, lkas_active, car_fingerprint, idx, is_panda_black)
      self.assertEqual(m_old, m)

      pcm_speed = random.randint(0, 65536)
      hud = HUDData(random.randint(0, 65536), random.randint(0, 65536), 1, random.randint(0, 65536),
              0xc1, random.randint(0, 65536), random.randint(0, 65536), random.randint(0, 65536), random.randint(0, 65536))
      idx = random.randint(0, 65536)
      is_metric = (random.randint(0, 2) % 2 == 0)
      m_old = hondacan.create_ui_commands(self.honda_cp_old, pcm_speed, hud, car_fingerprint, is_metric, idx, is_panda_black)
      m = hondacan.create_ui_commands(self.honda_cp, pcm_speed, hud, car_fingerprint, is_metric, idx, is_panda_black)
      self.assertEqual(m_old, m)

      button_val = random.randint(0, 65536)
      idx = random.randint(0, 65536)
      m_old = hondacan.spam_buttons_command(self.honda_cp_old, button_val, idx, car_fingerprint, is_panda_black)
      m = hondacan.spam_buttons_command(self.honda_cp, button_val, idx, car_fingerprint, is_panda_black)
      self.assertEqual(m_old, m)


if __name__ == "__main__":
  unittest.main()
