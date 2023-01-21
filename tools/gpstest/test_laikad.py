#!/usr/bin/env python3
import os
import time
import unittest

from common.params import Params
from system.hardware import TICI
import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes
from selfdrive.test.helpers import with_processes


def wait_for_location(sm, timeout, con=10):
  cons_meas = 0
  start_time = time.monotonic()
  while (time.monotonic() - start_time) < timeout:
    sm.update()
    if not sm.updated["gnssMeasurements"]:
      continue

    msg = sm["gnssMeasurements"]
    if 'positionECEF' in msg.to_dict():
      cons_meas += 1
    else:
      cons_meas = 0

    if cons_meas >= con:
      return True
  return False


class TestLaikad(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

    ublox_available = Params().get_bool("UbloxAvailable")
    if not ublox_available:
      raise unittest.SkipTest

  @with_processes(['pigeond', 'ubloxd'])
  def test_a_laikad_cold_start(self):
    time.sleep(5) # give ublox some time
    # delete cache (download folder is always deleted)
    Params().remove("LaikadEphemeris")

    # disable internet usage
    os.environ["LAIKAD_NO_INTERNET"] = "1"

    managed_processes['laikad'].start()

    start_time = time.monotonic()
    sm = messaging.SubMaster(["gnssMeasurements"])

    timeout = 60*3 # 3 min
    success = wait_for_location(sm, timeout, con=10)
    managed_processes['laikad'].stop()

    assert success, "Waiting for location timed out (3min)!"

    duration = time.monotonic() - start_time
    assert duration < 160, f"Received Location {duration}!"


if __name__ == "__main__":
  unittest.main()

