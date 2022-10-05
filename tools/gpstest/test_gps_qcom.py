#!/usr/bin/env python3
import time
import unittest
import subprocess as sp

from system.hardware import TICI
import cereal.messaging as messaging
from selfdrive.manager.process_config import managed_processes

def exec_cmd(cmd):
  p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
  return p.communicate()

def gps_enabled() -> bool:
  # verify GPS is disabled
  out, _ = exec_cmd("mmcli -m 0 --command='AT+QGPS?'")
  if b"QGPS: 0" in out.strip():
    return False
  return True

class TestGPS(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  def test_quectel_reset(self):
    # reset quectel modem and wait for first gps locations event

    # TODO: test on case where gps is disabled
    # mmcli -m 0 --command="ATI"

    if gps_enabled():
      cmd = "mmcli -m 0 --location-disable-gps-raw --location-disable-gps-nmea"
      exec_cmd(cmd)

      _, err = exec_cmd("mmcli -m 0 --command='AT+QGPSEND'")
      if len(err) != 0:
        print(f"QGPSEND failed: {err}")
        return

      # verify GPS is disabled
      if gps_enabled():
        print(f"GPS not disabled: {err}")
        return

    # delete assitent data to enforce cold start for GNSS
    cmd = "mmcli -m 0 --command='AT+QGPSDEL=0'"
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    _, err = p.communicate()
    if len(err) != 0:
      print(f"GPSDEL failed: {err}")


    # start rawgpsd and wait for gpsLocation
    managed_processes['rawgpsd'].start()
    start_time = time.monotonic()
    #qgs = messaging.sub_sock("qcomGnss", timeout=0.1)
    glo = messaging.sub_sock("gpsLocation", timeout=0.1)

    print("run waiting loop...")
    while True:
      '''
      events = messaging.drain_sock(qgs)
      if len(events) == 0:
        continue

      has_drmeas = False
      for e in events:
        print(e.qcomGnss.which())
        if e.qcomGnss.which() == 'drMeasurementReport':
          print(f"received qcomGNSS measurement: {time.monotonic() - start_time}")

          if len(e.qcomGnss.drMeasurementReport.sv) != 0:
            has_drmeas = True
            break

      if not has_drmeas:
        continue
      '''

      events = messaging.drain_sock(glo)
      if len(events) == 0:
        time.sleep(0.5)
        continue

      print(f"received GPS location: {time.monotonic() - start_time}")
      print(events[0])
      break

    managed_processes['rawgpsd'].stop()

if __name__ == "__main__":
  unittest.main()