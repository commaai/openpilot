#!/usr/bin/env python3
import time
import unittest
import struct
import numpy as np

import cereal.messaging as messaging
import selfdrive.sensord.pigeond as pd
from system.hardware import TICI
from selfdrive.test.helpers import with_processes


def read_events(service, duration_sec):
  service_sock = messaging.sub_sock(service, timeout=0.1)
  start_time_sec = time.monotonic()
  events = []
  while time.monotonic() - start_time_sec < duration_sec:
    events += messaging.drain_sock(service_sock)
    time.sleep(0.1)

  assert len(events) != 0, f"No '{service}'events collected!"
  return events


def verify_ubloxgnss_data(socket: messaging.SubSocket):
  start_time = 0
  end_time = 0
  events = messaging.drain_sock(socket)
  assert len(events) != 0, "no ublxGnss measurements"

  for event in events:
    if event.ubloxGnss.which() != "measurementReport":
      continue

    if start_time == 0:
      start_time = event.logMonoTime

    if event.ubloxGnss.measurementReport.numMeas != 0:
      end_time = event.logMonoTime
      break

  assert end_time != 0, "no ublox measurements received!"

  ttfm = (end_time - start_time)/1e9
  assert ttfm < 40, f"Time to first measurement > 40s, {ttfm}"

  # check for satellite count in measurements
  sat_count = []
  end_id = events.index(event)# pylint:disable=undefined-loop-variable
  for event in events[end_id:]:
    if event.ubloxGnss.which() == "measurementReport":
      sat_count.append(event.ubloxGnss.measurementReport.numMeas)

  num_sat = int(sum(sat_count)/len(sat_count))
  assert num_sat < 9, f"Not enough satellites {num_sat} (TestBox setup!)"


def verify_gps_location(socket: messaging.SubSocket):
  buf_lon = [0]*10
  buf_lat = [0]*10
  buf_i = 0
  events = messaging.drain_sock(socket)
  assert len(events) != 0, "no gpsLocationExternal measurements"

  start_time = events[0].logMonoTime
  end_time = 0
  for event in events:
    buf_lon[buf_i % 10] = event.gpsLocationExternal.longitude
    buf_lat[buf_i % 10] = event.gpsLocationExternal.latitude
    buf_i += 1

    if buf_i < 9:
      continue

    if any([lat == 0 or lon == 0 for lat,lon in zip(buf_lat, buf_lon)]):
      continue

    if np.std(buf_lon) < 1e-5 and np.std(buf_lat) < 1e-5:
      end_time = event.logMonoTime
      break

  assert end_time != 0, "GPS location never converged!"

  ttfl = (end_time - start_time)/1e9
  assert ttfl < 40, f"Time to first location > 40s, {ttfl}"

  hacc = events[-1].gpsLocationExternal.accuracy
  vacc = events[-1].gpsLocationExternal.verticalAccuracy
  assert hacc < 15, f"Horizontal accuracy too high, {hacc}"
  assert vacc < 40,  f"Vertical accuracy too high, {vacc}"


def verify_time_to_first_fix(pigeon):
  # get time to first fix from nav status message
  nav_status = b""
  while True:
    pigeon.send(b"\xb5\x62\x01\x03\x00\x00\x04\x0d")
    nav_status = pigeon.receive()
    if nav_status[:4] == b"\xb5\x62\x01\x03":
      break

  values = struct.unpack("<HHHIBBBBIIH", nav_status[:24])
  ttff = values[8]/1000
  # srms = values[9]/1000 
  assert ttff < 40, f"Time to first fix > 40s, {ttff}"


class TestGPS(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest


  @with_processes(['ubloxd'])
  def test_ublox_reset(self):

    pigeon, pm = pd.create_pigeon()
    pd.init_baudrate(pigeon)

    # deleting the backup does not always work on first try
    # mostly works on second try
    got_deleted = False
    for _ in range(5):
      # device cold start
      pigeon.send(b"\xb5\x62\x06\x04\x04\x00\xff\xff\x00\x00\x0c\x5d")
      time.sleep(1) # wait for cold start
      pd.init_baudrate(pigeon)

      # clear configuration
      pigeon.send_with_ack(b"\xb5\x62\x06\x09\x0d\x00\x00\x00\x1f\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x17\x71\x5b")

      # clear flash memory (almanac backup)
      pigeon.send_with_ack(b"\xB5\x62\x09\x14\x04\x00\x01\x00\x00\x00\x22\xf0")

      # try restoring backup to verify it got deleted
      pigeon.send(b"\xB5\x62\x09\x14\x00\x00\x1D\x60")
      # 1: failed to restore, 2: could restore, 3: no backup
      status = pigeon.wait_for_backup_restore_status()
      if status == 1 or status == 3:
        got_deleted = True
        break

    assert got_deleted, "Could not delete almanac backup"

    pd.initialize_pigeon(pigeon)

    ugs = messaging.sub_sock("ubloxGnss", timeout=0.1)
    gle = messaging.sub_sock("gpsLocationExternal", timeout=0.1)

    # receive some messages (restart after cold start takes up to 30seconds)
    pd.run_receiving(pigeon, pm, 40)

    verify_ubloxgnss_data(ugs)
    verify_gps_location(gle)

    # skip for now, this might hang for a while
    #verify_time_to_first_fix(pigeon)

    pd.set_power(False)


if __name__ == "__main__":
  unittest.main()