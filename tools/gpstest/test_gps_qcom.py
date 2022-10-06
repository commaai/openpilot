#!/usr/bin/env python3
import time
import unittest
import subprocess as sp

from system.hardware import TICI
import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes


class TestGPS(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  def test_quectel_cold_start(self):
    # delete assistance data to enforce cold start for GNSS
    # testing shows that this takes up to 20min

    cmd = "mmcli -m 0 --command='AT+QGPSDEL=0'"
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    _, err = p.communicate()
    assert len(err) == 0, f"GPSDEL failed: {err}"

    managed_processes['rawgpsd'].start()
    start_time = time.monotonic()
    glo = messaging.sub_sock("gpsLocation", timeout=0.1)

    while True:
      events = messaging.drain_sock(glo)
      if len(events) == 0:
        time.sleep(0.5)
        continue

      print(f"received GPS location: {time.monotonic() - start_time}")
      break

    managed_processes['rawgpsd'].stop()

if __name__ == "__main__":
  unittest.main()