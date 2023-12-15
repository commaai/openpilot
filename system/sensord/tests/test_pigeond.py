#!/usr/bin/env python3
import pytest
import time
import unittest

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.gpio import gpio_read
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.system.hardware.tici.pins import GPIO


# TODO: test TTFF when we have good A-GNSS
@pytest.mark.tici
class TestPigeond(unittest.TestCase):

  def tearDown(self):
    managed_processes['pigeond'].stop()

  @with_processes(['pigeond'])
  def test_frequency(self):
    sm = messaging.SubMaster(['ubloxRaw'])

    # setup time
    for _ in range(int(5 * SERVICE_LIST['ubloxRaw'].frequency)):
      sm.update()

    for _ in range(int(10 * SERVICE_LIST['ubloxRaw'].frequency)):
      sm.update()
      assert sm.all_checks()

  def test_startup_time(self):
    for _ in range(5):
      sm = messaging.SubMaster(['ubloxRaw'])
      managed_processes['pigeond'].start()

      start_time = time.monotonic()
      for __ in range(10):
        sm.update(1 * 1000)
        if sm.updated['ubloxRaw']:
          break
      assert sm.rcv_frame['ubloxRaw'] > 0, "pigeond didn't start outputting messages in time"

      et = time.monotonic() - start_time
      assert et < 5, f"pigeond took {et:.1f}s to start"
      managed_processes['pigeond'].stop()

  def test_turns_off_ublox(self):
    for s in (0.1, 0.5, 1, 5):
      managed_processes['pigeond'].start()
      time.sleep(s)
      managed_processes['pigeond'].stop()

      assert gpio_read(GPIO.UBLOX_RST_N) == 0
      assert gpio_read(GPIO.GNSS_PWR_EN) == 0


if __name__ == "__main__":
  unittest.main()
