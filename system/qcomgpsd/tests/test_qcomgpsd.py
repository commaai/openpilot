import os
import pytest
import json
import time
import datetime
import subprocess

import cereal.messaging as messaging
from openpilot.system.qcomgpsd.qcomgpsd import at_cmd, wait_for_modem
from openpilot.system.manager.process_config import managed_processes

GOOD_SIGNAL = bool(int(os.getenv("GOOD_SIGNAL", '0')))


@pytest.mark.tici
class TestRawgpsd:
  @classmethod
  def setup_class(cls):
    os.system("sudo systemctl start systemd-resolved")
    os.system("sudo systemctl restart ModemManager lte")
    wait_for_modem()

  @classmethod
  def teardown_class(cls):
    managed_processes['qcomgpsd'].stop()
    os.system("sudo systemctl restart systemd-resolved")
    os.system("sudo systemctl restart ModemManager lte")

  def setup_method(self):
    at_cmd("AT+QGPSDEL=0")
    self.sm = messaging.SubMaster(['qcomGnss', 'gpsLocation', 'gnssMeasurements'])

  def teardown_method(self):
    managed_processes['qcomgpsd'].stop()
    os.system("sudo systemctl restart systemd-resolved")

  def _wait_for_output(self, t):
    dt = 0.1
    for _ in range(t*int(1/dt)):
      self.sm.update(0)
      if self.sm.updated['qcomGnss']:
        break
      time.sleep(dt)
    return self.sm.updated['qcomGnss']

  def test_no_crash_double_command(self):
    at_cmd("AT+QGPSDEL=0")
    at_cmd("AT+QGPSDEL=0")

  def test_wait_for_modem(self):
    os.system("sudo systemctl stop ModemManager")
    managed_processes['qcomgpsd'].start()
    assert not self._wait_for_output(5)

    os.system("sudo systemctl restart ModemManager")
    assert self._wait_for_output(30)

  def test_startup_time(self, subtests):
    for internet in (True, False):
      if not internet:
        os.system("sudo systemctl stop systemd-resolved")
      with subtests.test(internet=internet):
        managed_processes['qcomgpsd'].start()
        assert self._wait_for_output(7)
        managed_processes['qcomgpsd'].stop()

  def test_turns_off_gnss(self, subtests):
    for s in (0.1, 1, 5):
      with subtests.test(runtime=s):
        managed_processes['qcomgpsd'].start()
        time.sleep(s)
        managed_processes['qcomgpsd'].stop()

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
      assert abs((datetime.datetime.utcnow() - injected_time).total_seconds()) < 60*60*12
    else:
      valid_duration, injected_time_str = out.split(",", 1)
      injected_time_str = injected_time_str.replace('\"', '').replace('\'', '')
      assert injected_time_str[:] == '1980/01/05,19:00:00'[:]
      assert valid_duration == '0'

  def test_assistance_loading(self):
    managed_processes['qcomgpsd'].start()
    assert self._wait_for_output(10)
    managed_processes['qcomgpsd'].stop()
    self.check_assistance(True)

  def test_no_assistance_loading(self):
    os.system("sudo systemctl stop systemd-resolved")

    managed_processes['qcomgpsd'].start()
    assert self._wait_for_output(10)
    managed_processes['qcomgpsd'].stop()
    self.check_assistance(False)

  def test_late_assistance_loading(self):
    os.system("sudo systemctl stop systemd-resolved")

    managed_processes['qcomgpsd'].start()
    self._wait_for_output(17)
    assert self.sm.updated['qcomGnss']

    os.system("sudo systemctl restart systemd-resolved")
    time.sleep(15)
    managed_processes['qcomgpsd'].stop()
    self.check_assistance(True)
