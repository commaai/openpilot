#!/usr/bin/env python3
import pytest
import time
import unittest
import numpy as np

from openpilot.system.hardware.tici.hardware import Tici

HARDWARE = Tici()


@pytest.mark.tici
class TestHardware(unittest.TestCase):

  def test_power_save_time(self):
    ts = []
    for _ in range(5):
      for on in (True, False):
        st = time.monotonic()
        HARDWARE.set_power_save(on)
        ts.append(time.monotonic() - st)

    assert 0.1 < np.mean(ts) < 0.25
    assert max(ts) < 0.3


if __name__ == "__main__":
  unittest.main()
