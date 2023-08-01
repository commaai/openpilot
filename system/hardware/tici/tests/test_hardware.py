#!/usr/bin/env python3
import time
import unittest
import numpy as np

from system.hardware import TICI
from system.hardware.tici.hardware import Tici

HARDWARE = Tici()

class TestHardware(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  def test_power_save_time(self):
    ts = []
    for _ in range(5):
      for on in (True, False):
        st = time.monotonic()
        HARDWARE.set_power_save(on)
        ts.append(time.monotonic() - st)

    assert 0.1 < np.mean(ts) < 0.2
    assert max(ts) < 0.3


if __name__ == "__main__":
  unittest.main()
