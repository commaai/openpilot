#!/usr/bin/env python3
import unittest
from panda import Panda
from panda.tests.safety import libpandasafety_py
from panda.tests.safety.common import make_msg, test_spam_can_buses
from panda.tests.safety.test_honda import HONDA_BG_HW, HONDA_BH_HW

MAX_BRAKE = 255

H_TX_MSGS = [[0xE4, 0], [0x296, 1], [0x33D, 0]]  # Bosch Harness
G_TX_MSGS = [[0xE4, 2], [0x296, 0], [0x33D, 2]]  # Bosch Giraffe


class TestHondaSafety(unittest.TestCase):
  @classmethod
  def setUp(cls):
    cls.safety = libpandasafety_py.libpandasafety
    cls.safety.set_safety_hooks(Panda.SAFETY_HONDA_BOSCH_GIRAFFE, 0)
    cls.safety.init_tests_honda()

  def test_spam_can_buses(self):
    for hw in [HONDA_BG_HW, HONDA_BH_HW]:
      self.safety.set_honda_hw(hw)
      test_spam_can_buses(self, H_TX_MSGS if hw == HONDA_BH_HW else G_TX_MSGS)

  def test_fwd_hook(self):
    buss = range(0x0, 0x3)
    msgs = range(0x1, 0x800)
    for hw in [HONDA_BG_HW, HONDA_BH_HW]:
      self.safety.set_honda_hw(hw)
      bus_rdr_cam = 2 if hw == HONDA_BH_HW else 1
      bus_rdr_car = 0 if hw == HONDA_BH_HW else 2
      bus_pt = 1 if hw == HONDA_BH_HW else 0

      blocked_msgs = [0xE4, 0x33D]
      for b in buss:
        for m in msgs:
          if b == bus_pt:
            fwd_bus = -1
          elif b == bus_rdr_cam:
            fwd_bus = -1 if m in blocked_msgs else bus_rdr_car
          elif b == bus_rdr_car:
            fwd_bus = bus_rdr_cam

          # assume len 8
          self.assertEqual(fwd_bus, self.safety.safety_fwd_hook(b, make_msg(b, m, 8)))


if __name__ == "__main__":
  unittest.main()
