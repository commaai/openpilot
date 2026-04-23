import os
import pytest
import time

import cereal.messaging as messaging
from openpilot.system.qcomgpsd.qcomgpsd import at_cmd, wait_for_modem
from openpilot.system.manager.process_config import managed_processes


@pytest.mark.tici
class TestQcomgpsd:
  @classmethod
  def setup_class(cls):
    os.system("sudo systemctl restart ModemManager lte")
    wait_for_modem()

  @classmethod
  def teardown_class(cls):
    managed_processes['qcomgpsd'].stop()
    os.system("sudo systemctl restart ModemManager lte")

  def setup_method(self):
    self.sm = messaging.SubMaster(['qcomGnss', 'gpsLocation'])

  def teardown_method(self):
    managed_processes['qcomgpsd'].stop()

  def _wait_for_output(self, t):
    dt = 0.1
    for _ in range(t*int(1/dt)):
      self.sm.update(0)
      if self.sm.updated['qcomGnss']:
        break
      time.sleep(dt)
    return self.sm.updated['qcomGnss']

  def test_startup_time(self):
    managed_processes['qcomgpsd'].start()
    assert self._wait_for_output(30)
    managed_processes['qcomgpsd'].stop()

  def test_turns_off_gnss(self, subtests):
    for s in (0.1, 1, 5):
      with subtests.test(runtime=s):
        managed_processes['qcomgpsd'].start()
        time.sleep(s)
        managed_processes['qcomgpsd'].stop()

        wait_for_modem()
        resp = at_cmd("AT+QGPS?")
        assert "+QGPS: 0" in resp
