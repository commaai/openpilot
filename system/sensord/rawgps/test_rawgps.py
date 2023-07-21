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
from system.sensord.rawgps.rawgpsd import at_cmd, wait_for_modem
from selfdrive.manager.process_config import managed_processes
from common.transformations.coordinates import ecef_from_geodetic

GOOD_SIGNAL = bool(int(os.getenv("GOOD_SIGNAL", '0')))


class TestRawgpsd(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    os.system("sudo systemctl restart systemd-resolved")
    os.system("sudo systemctl restart ModemManager lte")
    wait_for_modem()
    if not TICI:
      raise unittest.SkipTest
    cls.sm = messaging.SubMaster(['qcomGnss', 'gpsLocation', 'gnssMeasurements'])

  @classmethod
  def tearDownClass(cls):
    managed_processes['rawgpsd'].stop()
    os.system("sudo systemctl restart systemd-resolved")
    os.system("sudo systemctl restart ModemManager lte")

  def setUp(self):
    at_cmd("AT+QGPSDEL=0")

  def tearDown(self):
    managed_processes['rawgpsd'].stop()
    os.system("sudo systemctl restart systemd-resolved")

  def _wait_for_output(self, t=10):
    time.sleep(t)
    self.sm.update()

  def test_no_crash_double_command(self):
    at_cmd("AT+QGPSDEL=0")
    at_cmd("AT+QGPSDEL=0")

  def test_wait_for_modem(self):
    os.system("sudo systemctl stop ModemManager lte")
    managed_processes['rawgpsd'].start()
    self._wait_for_output(10)
    assert not self.sm.updated['qcomGnss']

    os.system("sudo systemctl restart ModemManager lte")
    self._wait_for_output(30)
    assert self.sm.updated['qcomGnss']

  def test_startup_time(self):
    for i in range(2):
      if i == 1:
        os.system("sudo systemctl stop systemd-resolved")
      managed_processes['rawgpsd'].start()
      self._wait_for_output(10)
      assert self.sm.updated['qcomGnss']
      managed_processes['rawgpsd'].stop()

  def test_turns_off_gnss(self):
    for s in (0.1, 0.5, 1, 5):
      managed_processes['rawgpsd'].start()
      time.sleep(s)
      managed_processes['rawgpsd'].stop()

      ls = subprocess.check_output("mmcli -m any --location-status --output-json", shell=True, encoding='utf-8')
      loc_status = json.loads(ls)
      assert set(loc_status['modem']['location']['enabled']) <= {'3gpp-lac-ci'}


  def check_assistance(self, should_be_loaded):
    # after QGPSDEL: '+QGPSXTRADATA: 0,"1980/01/05,19:00:00"'
    # after loading: '+QGPSXTRADATA: 10080,"2023/06/24,19:00:00"'
    out = at_cmd("AT+QGPSXTRADATA?")
    out = out.split("+QGPSXTRADATA:")[1].split("'")[0].strip()
    valid_duration, injected_time_str = out.split(",", 1)
    if should_be_loaded:
      assert valid_duration == "10080"  # should be max time
      injected_time = datetime.datetime.strptime(injected_time_str.replace("\"", ""), "%Y/%m/%d,%H:%M:%S")
      self.assertLess(abs((datetime.datetime.utcnow() - injected_time).total_seconds()), 60*60*12)
    else:
      valid_duration, injected_time_str = out.split(",", 1)
      injected_time_str = injected_time_str.replace('\"', '').replace('\'', '')
      assert injected_time_str[:] == '1980/01/05,19:00:00'[:]
      assert valid_duration == '0'

  def test_assistance_loading(self):
    managed_processes['rawgpsd'].start()
    self._wait_for_output(10)
    assert self.sm.updated['qcomGnss']
    managed_processes['rawgpsd'].stop()
    self.check_assistance(True)

  def test_no_assistance_loading(self):
    os.system("sudo systemctl stop systemd-resolved")

    managed_processes['rawgpsd'].start()
    self._wait_for_output(10)
    assert self.sm.updated['qcomGnss']
    managed_processes['rawgpsd'].stop()
    self.check_assistance(False)

  def test_late_assistance_loading(self):
    os.system("sudo systemctl stop systemd-resolved")

    managed_processes['rawgpsd'].start()
    self._wait_for_output(17)
    assert self.sm.updated['qcomGnss']
    os.system("sudo systemctl restart systemd-resolved")
    self._wait_for_output(15)
    managed_processes['rawgpsd'].stop()
    self.check_assistance(True)

  @unittest.skipIf(not GOOD_SIGNAL, "No good GPS signal")
  def test_fix(self):
    managed_processes['rawgpsd'].start()
    managed_processes['laikad'].start()
    assert self._wait_for_output(60)
    assert self.sm.updated['qcomGnss']
    assert self.sm.updated['gpsLocation']
    assert self.sm['gpsLocation'].flags == 1
    module_fix = ecef_from_geodetic([self.sm['gpsLocation'].latitude,
                                     self.sm['gpsLocation'].longitude,
                                     self.sm['gpsLocation'].altitude])
    assert self.sm['gnssMeasurements'].positionECEF.valid
    total_diff = np.array(self.sm['gnssMeasurements'].positionECEF.value) - module_fix
    self.assertLess(np.linalg.norm(total_diff), 100)
    managed_processes['laikad'].stop()
    managed_processes['rawgpsd'].stop()

if __name__ == "__main__":
  unittest.main(failfast=True)
