import pytest
import time
import numpy as np

from openpilot.system.hardware.tici.hardware import Tici

HARDWARE = Tici()


@pytest.mark.tici
class TestHardware:

  def test_power_save_time(self):
    ts = {True: [], False: []}
    for _ in range(5):
      for on in (True, False):
        st = time.monotonic()
        HARDWARE.set_power_save(on)
        ts[on].append(time.monotonic() - st)

    # disabling power save is the main time-critical one
    assert 0.1 < np.mean(ts[False]) < 0.15
    assert max(ts[False]) < 0.2

    assert 0.1 < np.mean(ts[True]) < 0.35
    assert max(ts[True]) < 0.4
