#!/usr/bin/env python3
import os
import unittest
from datetime import datetime

from laika.gps_time import GPSTime
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement
from selfdrive.locationd.laikad import Laikad, create_measurement_msg
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader

os.environ["FILEREADER_CACHE"] = "1"


def get_log(segs=range(0)):
  logs = []
  for i in segs:
    logs.extend(LogReader(get_url("4cf7a6ad03080c90|2021-09-29--13-46-36", i)))
  return [m for m in logs if m.which() == 'ubloxGnss']


def verify_messages(lr, laikad):
  good_msgs = []
  for m in lr:
    msg = laikad.process_ublox_msg(m.ubloxGnss, laikad.astro_dog, m.logMonoTime)
    if msg is not None and len(msg.gnssMeasurements.correctedMeasurements) > 0:
      good_msgs.append(msg)
  return good_msgs


class TestLaikad(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.logs = get_log(range(1))

  def test_create_msg_without_errors(self):
    gpstime = GPSTime.from_datetime(datetime.now())
    meas = GNSSMeasurement(ConstellationId.GPS, 1, gpstime.week, gpstime.tow, {'C1C': 0., 'D1C': 0.}, {'C1C': 0., 'D1C': 0.})
    # Fake observables_final to be correct
    meas.observables_final = meas.observables
    msg = create_measurement_msg(meas)

    self.assertEqual(msg.constellationId, 'gps')

  def test_laika_online(self):
    # Set to offline forces to use ephemeris messages
    laikad = Laikad(use_internet=True)
    correct_msgs = verify_messages(self.logs, laikad)

    correct_msgs_expected = 560
    self.assertEqual(correct_msgs_expected, len(correct_msgs))
    self.assertEqual(correct_msgs_expected, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_laika_offline(self):
    # Set to offline forces to use ephemeris messages
    laikad = Laikad(use_internet=False)
    correct_msgs = verify_messages(self.logs, laikad)

    self.assertEqual(256, len(correct_msgs))
    self.assertEqual(256, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_laika_offline_ephem_at_start(self):
    # Test offline but process ephemeris msgs of segment first
    laikad = Laikad(use_internet=False)
    ephemeris_logs = [m for m in self.logs if m.ubloxGnss.which() == 'ephemeris']
    correct_msgs = verify_messages(ephemeris_logs+self.logs, laikad)

    self.assertEqual(554, len(correct_msgs))
    self.assertGreaterEqual(554, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_laika_get_orbits(self):
    os.environ["FILEREADER_CACHE"] = "1"
    os.environ["NASA_USERNAME"] = "gkoning"
    os.environ["NASA_PASSWORD"] = "u&+9A3L+RA6K6z8"
    laikad = Laikad(use_internet=False)
    laikad.fetch_orbits()
    print(laikad.astro_dog.orbits.keys())
    print(laikad.latest_epoch_fetched.as_datetime())


if __name__ == "__main__":
  unittest.main()
