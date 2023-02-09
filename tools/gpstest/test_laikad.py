#!/usr/bin/env python3
import os
import time
import unittest

import cereal.messaging as messaging
import selfdrive.sensord.pigeond as pd

from common.params import Params
from system.hardware import TICI
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
    cons_meas = (cons_meas + 1) if 'positionECEF' in msg.to_dict() else 0
    if cons_meas >= con:
      return True
  return False


class TestLaikad(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    if not TICI:
      raise unittest.SkipTest

    ublox_available = Params().get_bool("UbloxAvailable")
    if not ublox_available:
      raise unittest.SkipTest

  def setUp(self):
    # ensure laikad cold start
    Params().remove("LaikadEphemeris")
    os.environ["LAIKAD_NO_INTERNET"] = "1"
    managed_processes['laikad'].start()

  def tearDown(self):
    managed_processes['laikad'].stop()


  @with_processes(['pigeond', 'ubloxd'])
  def test_laikad_cold_start(self):
    time.sleep(5)

    start_time = time.monotonic()
    sm = messaging.SubMaster(["gnssMeasurements"])

    success = wait_for_location(sm, 60*2, con=10)
    duration = time.monotonic() - start_time

    assert success, "Waiting for location timed out (2min)!"
    assert duration < 60, f"Received Location {duration}!"


  @with_processes(['ubloxd'])
  def test_laikad_ublox_reset_start(self):
    time.sleep(2)

    pigeon, pm = pd.create_pigeon()
    pd.init_baudrate(pigeon)
    assert pigeon.reset_device(), "Could not reset device!"

    laikad_sock = messaging.sub_sock("gnssMeasurements", timeout=0.1)
    ublox_gnss_sock = messaging.sub_sock("ubloxGnss", timeout=0.1)

    pd.init_baudrate(pigeon)
    pd.initialize_pigeon(pigeon)
    pd.run_receiving(pigeon, pm, 180)

    ublox_msgs = messaging.drain_sock(ublox_gnss_sock)
    laikad_msgs = messaging.drain_sock(laikad_sock)

    gps_ephem_cnt = 0
    glonass_ephem_cnt = 0
    for um in ublox_msgs:
      if um.ubloxGnss.which() == 'ephemeris':
        gps_ephem_cnt += 1
      elif um.ubloxGnss.which() == 'glonassEphemeris':
        glonass_ephem_cnt += 1

    assert gps_ephem_cnt > 0, "NO gps ephemeris collected!"
    assert glonass_ephem_cnt > 0, "NO glonass ephemeris collected!"

    pos_meas = 0
    duration = -1
    for lm in laikad_msgs:
      pos_meas = (pos_meas + 1) if 'positionECEF' in lm.gnssMeasurements.to_dict() else 0
      if pos_meas > 5:
        duration = (lm.logMonoTime - laikad_msgs[0].logMonoTime)*1e-9
        break

    assert pos_meas > 5, "NOT enough positions at end of read!"
    assert duration < 120, "Laikad took too long to get a Position!"

if __name__ == "__main__":
  unittest.main()
