#!/usr/bin/env python3
import os
import json
import time
import datetime
import unittest
import subprocess

import cereal.messaging as messaging
from system.hardware import TICI
from system.sensord.rawgps.rawgpsd import at_cmd
from selfdrive.manager.process_config import managed_processes


class TestRawgpsd(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

    cls.sm = messaging.SubMaster(['qcomGnss'])

  def tearDown(self):
    managed_processes['rawgpsd'].stop()

  def _wait_for_output(self, t=10):
    self.sm.update(0)
    for __ in range(t*10):
      self.sm.update(100)
      if self.sm.updated['qcomGnss']:
        break
    return self.sm.updated['qcomGnss']

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
    valid_duration, injected_date = out.split(",", 1)
    assert valid_duration == "10080"  # should be max time

    # TODO: time doesn't match up
    assert injected_date[1:].startswith(datetime.datetime.now().strftime("%Y/%m/%d"))


if __name__ == "__main__":
  unittest.main(failfast=True)
