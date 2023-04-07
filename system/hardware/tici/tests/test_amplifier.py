#!/usr/bin/env python3
import time
import random
import unittest

from panda import Panda
from system.hardware.tici.hardware import Tici
from system.hardware.tici.amplifier import Amplifier


class TestAmplifier(unittest.TestCase):

  def test_init_while_siren_play(self):
    p = Panda()
    for _ in range(5):
      p.set_siren(False)
      time.sleep(0.1)

      p.set_siren(True)
      time.sleep(random.randint(0, 5))

      amp = Amplifier(debug=True)
      r = amp.initialize_configuration(Tici().model)
      assert r



if __name__ == "__main__":
  unittest.main()
