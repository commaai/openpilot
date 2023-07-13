#!/usr/bin/env python3
import os
import json
import time
import datetime
import unittest
import subprocess
import numpy as np

import cereal.messaging as messaging
from system.hardware import TICI
from system.sensord.rawgps.rawgpsd import at_cmd
from selfdrive.manager.process_config import managed_processes
from common.transformations.coordinates import ecef_from_geodetic

GOOD_SIGNAL = bool(int(os.getenv("GOOD_SIGNAL", '0')))
UPDATE_MS = 100
UPDATES_PER_S = 1000//UPDATE_MS


class TestRawgpsd(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

    cls.sm_qcom_gnss = messaging.SubMaster(['qcomGnss'])
    cls.sm_gps_location = messaging.SubMaster(['gpsLocation'])
    cls.sm_gnss_measurements = messaging.SubMaster(['gnssMeasurements'])

  def tearDown(self):
    managed_processes['rawgpsd'].stop()

  def _wait_for_output(self, t=10):
    self.sm_qcom_gnss.update(0)
    for __ in range(t*UPDATES_PER_S):
      self.sm_qcom_gnss.update(UPDATE_MS)
      if self.sm_qcom_gnss.updated['qcomGnss']:
        return True

  def _wait_for_location(self, t=10):
    self.sm_gps_location.update(0)
    for __ in range(t*UPDATES_PER_S):
      self.sm_gps_location.update(UPDATE_MS)
      if self.sm_gps_location.updated['gpsLocation'] and self.sm_gps_location['gpsLocation'].flags:
        return True
    return False

  def _wait_for_laikad_location(self, t=10):
    self.sm_gnss_measurements.update(0)
    for __ in range(t*UPDATES_PER_S):
      self.sm_gnss_measurements.update(UPDATE_MS)
      if self.sm_gnss_measurements.updated['gnssMeasurements'] and self.sm_gnss_measurements['gnssMeasurements'].positionECEF.valid:
        return True
    return False

  def test_wait_for_modem(self):
    os.system("sudo systemctl stop ModemManager lte")
    managed_processes['rawgpsd'].start()
    assert not self._wait_for_output(10)

    os.system("sudo systemctl restart ModemManager lte")
    assert self._wait_for_output(30)

  def test_startup_time(self):
    for _ in range(5):
      managed_processes['rawgpsd'].start()

      start_time = time.monotonic()
      assert self._wait_for_output(), "rawgpsd didn't start outputting messages in time"

      et = time.monotonic() - start_time
      assert et < 7, f"rawgpsd took {et:.1f}s to start"
      managed_processes['rawgpsd'].stop()

  def test_turns_off_gnss(self):
    for s in (0.1, 0.5, 1, 5):
      managed_processes['rawgpsd'].start()
      time.sleep(s)
      managed_processes['rawgpsd'].stop()

      ls = subprocess.check_output("mmcli -m any --location-status --output-json", shell=True, encoding='utf-8')
      loc_status = json.loads(ls)
      assert set(loc_status['modem']['location']['enabled']) <= {'3gpp-lac-ci'}

  def test_assistance_loading(self):
    # clear assistance data
    at_cmd("AT+QGPSDEL=0")

    managed_processes['rawgpsd'].start()
    assert self._wait_for_output(10)
    managed_processes['rawgpsd'].stop()

    # after QGPSDEL: '+QGPSXTRADATA: 0,"1980/01/05,19:00:00"'
    # after loading: '+QGPSXTRADATA: 10080,"2023/06/24,19:00:00"'
    out = at_cmd("AT+QGPSXTRADATA?")
    out = out.split("+QGPSXTRADATA:")[1].split("'")[0].strip()
    valid_duration, injected_time_str = out.split(",", 1)
    assert valid_duration == "10080"  # should be max time
    injected_time = datetime.datetime.strptime(injected_time_str.replace("\"", ""), "%Y/%m/%d,%H:%M:%S")
    self.assertLess(abs((datetime.datetime.utcnow() - injected_time).total_seconds()), 60*60*12)

  @unittest.skipIf(not GOOD_SIGNAL, "No good GPS signal")
  def test_fix(self):
    # clear assistance data
    at_cmd("AT+QGPSDEL=0")

    managed_processes['rawgpsd'].start()
    managed_processes['laikad'].start()
    assert self._wait_for_location(120)
    assert self.sm_gps_location['gpsLocation'].flags == 1
    module_fix = ecef_from_geodetic([self.sm_gps_location['gpsLocation'].latitude,
                                     self.sm_gps_location['gpsLocation'].longitude,
                                     self.sm_gps_location['gpsLocation'].altitude])
    assert self._wait_for_laikad_location(90)
    total_diff = np.array(self.sm_gnss_measurements['gnssMeasurements'].positionECEF.value) - module_fix
    print(total_diff)
    self.assertLess(np.linalg.norm(total_diff), 100)
    managed_processes['laikad'].stop()
    managed_processes['rawgpsd'].stop()

if __name__ == "__main__":
  unittest.main(failfast=True)
