#!/usr/bin/env python3
import pytest
import time
import unittest
import struct

from openpilot.common.params import Params
import cereal.messaging as messaging
import openpilot.system.sensord.pigeond as pd
from openpilot.selfdrive.test.helpers import with_processes


def read_events(service, duration_sec):
  service_sock = messaging.sub_sock(service, timeout=0.1)
  start_time_sec = time.monotonic()
  events = []
  while time.monotonic() - start_time_sec < duration_sec:
    events += messaging.drain_sock(service_sock)
    time.sleep(0.1)

  assert len(events) != 0, f"No '{service}'events collected!"
  return events


def create_backup(pigeon):
  # controlled GNSS stop
  pigeon.send(b"\xB5\x62\x06\x04\x04\x00\x00\x00\x08\x00\x16\x74")

  # store almanac in flash
  pigeon.send(b"\xB5\x62\x09\x14\x04\x00\x00\x00\x00\x00\x21\xEC")
  try:
    if not pigeon.wait_for_ack(ack=pd.UBLOX_SOS_ACK, nack=pd.UBLOX_SOS_NACK):
      raise RuntimeError("Could not store almanac")
  except TimeoutError:
    pass


def verify_ubloxgnss_data(socket: messaging.SubSocket, max_time: int):
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
  assert ttfm < max_time, f"Time to first measurement > {max_time}s, {ttfm}"

  # check for satellite count in measurements
  sat_count = []
  end_id = events.index(event)# pylint:disable=undefined-loop-variable
  for event in events[end_id:]:
    if event.ubloxGnss.which() == "measurementReport":
      sat_count.append(event.ubloxGnss.measurementReport.numMeas)

  num_sat = int(sum(sat_count)/len(sat_count))
  assert num_sat >= 5, f"Not enough satellites {num_sat} (TestBox setup!)"


def verify_gps_location(socket: messaging.SubSocket, max_time: int):
  events = messaging.drain_sock(socket)
  assert len(events) != 0, "no gpsLocationExternal measurements"

  start_time = events[0].logMonoTime
  end_time = 0
  for event in events:
    gps_valid = event.gpsLocationExternal.flags % 2

    if gps_valid:
      end_time = event.logMonoTime
      break

  assert end_time != 0, "GPS location never converged!"

  ttfl = (end_time - start_time)/1e9
  assert ttfl < max_time, f"Time to first location > {max_time}s, {ttfl}"

  hacc = events[-1].gpsLocationExternal.accuracy
  vacc = events[-1].gpsLocationExternal.verticalAccuracy
  assert hacc < 20, f"Horizontal accuracy too high, {hacc}"
  assert vacc < 45,  f"Vertical accuracy too high, {vacc}"


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


@pytest.mark.tici
class TestGPS(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    ublox_available = Params().get_bool("UbloxAvailable")
    if not ublox_available:
      raise unittest.SkipTest


  def tearDown(self):
    pd.set_power(False)

  @with_processes(['ubloxd'])
  def test_a_ublox_reset(self):

    pigeon, pm = pd.create_pigeon()
    pd.init_baudrate(pigeon)
    assert pigeon.reset_device(), "Could not reset device!"

    pd.initialize_pigeon(pigeon)

    ugs = messaging.sub_sock("ubloxGnss", timeout=0.1)
    gle = messaging.sub_sock("gpsLocationExternal", timeout=0.1)

    # receive some messages (restart after cold start takes up to 30seconds)
    pd.run_receiving(pigeon, pm, 60)

    # store almanac for next test
    create_backup(pigeon)

    verify_ubloxgnss_data(ugs, 60)
    verify_gps_location(gle, 60)

    # skip for now, this might hang for a while
    #verify_time_to_first_fix(pigeon)


  @with_processes(['ubloxd'])
  def test_b_ublox_almanac(self):
    pigeon, pm = pd.create_pigeon()
    pd.init_baudrate(pigeon)

    # device cold start
    pigeon.send(b"\xb5\x62\x06\x04\x04\x00\xff\xff\x00\x00\x0c\x5d")
    time.sleep(1) # wait for cold start
    pd.init_baudrate(pigeon)

    # clear configuration
    pigeon.send_with_ack(b"\xb5\x62\x06\x09\x0d\x00\x00\x00\x1f\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x17\x71\x5b")

    # restoring almanac backup
    pigeon.send(b"\xB5\x62\x09\x14\x00\x00\x1D\x60")
    status = pigeon.wait_for_backup_restore_status()
    assert status == 2, "Could not restore almanac backup"

    pd.initialize_pigeon(pigeon)

    ugs = messaging.sub_sock("ubloxGnss", timeout=0.1)
    gle = messaging.sub_sock("gpsLocationExternal", timeout=0.1)

    pd.run_receiving(pigeon, pm, 15)
    verify_ubloxgnss_data(ugs, 15)
    verify_gps_location(gle, 20)


  @with_processes(['ubloxd'])
  def test_c_ublox_startup(self):
    pigeon, pm = pd.create_pigeon()
    pd.init_baudrate(pigeon)
    pd.initialize_pigeon(pigeon)

    ugs = messaging.sub_sock("ubloxGnss", timeout=0.1)
    gle = messaging.sub_sock("gpsLocationExternal", timeout=0.1)
    pd.run_receiving(pigeon, pm, 10)
    verify_ubloxgnss_data(ugs, 10)
    verify_gps_location(gle, 10)


if __name__ == "__main__":
  unittest.main()
