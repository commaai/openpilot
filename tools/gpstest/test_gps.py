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
  assert ttfm < 35, f"Time to first measurement > 35s, {ttfm}"

  # check for satellite count in measurements
  sat_count = []
  end_id = events.index(event)# pylint:disable=undefined-loop-variable
  for event in events[end_id:]:
    if event.ubloxGnss.which() == "measurementReport":
      sat_count.append(event.ubloxGnss.measurementReport.numMeas)

  num_sat = int(sum(sat_count)/len(sat_count))
  assert num_sat > 8, f"Not enough satellites {num_sat} (TestBox setup!)"


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
  assert vacc < 43,  f"Vertical accuracy too high, {vacc}"


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

  def tearDown(self):
    pd.set_power(False)

  @with_processes(['ubloxd'])
  def test_ublox_reset(self):

    pigeon, pm = pd.create_pigeon()
    pd.init_baudrate(pigeon)
    assert pigeon.reset_device(), "Could not reset device!"

    pd.initialize_pigeon(pigeon)

    ugs = messaging.sub_sock("ubloxGnss", timeout=0.1)
    gle = messaging.sub_sock("gpsLocationExternal", timeout=0.1)

    # receive some messages (restart after cold start takes up to 30seconds)
    pd.run_receiving(pigeon, pm, 40)

    verify_ubloxgnss_data(ugs)
    verify_gps_location(gle)

    # skip for now, this might hang for a while
    #verify_time_to_first_fix(pigeon)


if __name__ == "__main__":
  unittest.main()