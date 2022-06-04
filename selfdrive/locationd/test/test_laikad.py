#!/usr/bin/env python3
import unittest
from datetime import datetime

from laika.gps_time import GPSTime
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement
from selfdrive.locationd.laikad import Laikad, create_measurement_msg
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader


def get_log(segs=range(0)):
  logs = []
  for i in segs:
    logs.extend(LogReader(get_url("4cf7a6ad03080c90|2021-09-29--13-46-36", i)))
  return [m for m in logs if m.which() == 'ubloxGnss']


def process_msgs(lr, laikad: Laikad):
  good_msgs = []
  for m in lr:
    msg = laikad.process_ublox_msg(m.ubloxGnss, m.logMonoTime)
    if msg is not None:
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
    msgs = process_msgs(self.logs, laikad)
    correct_msgs = [m for m in msgs if len(m.gnssMeasurements.correctedMeasurements) > 0]
    correct_msgs_expected = 560
    self.assertEqual(correct_msgs_expected, len(correct_msgs))
    self.assertEqual(correct_msgs_expected, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_laika_offline(self):
    # Set to offline forces to use ephemeris messages
    laikad = Laikad(use_internet=False)
    msgs = process_msgs(self.logs, laikad)
    correct_msgs = [m for m in msgs if len(m.gnssMeasurements.correctedMeasurements) > 0]

    self.assertEqual(256, len(correct_msgs))
    self.assertEqual(256, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_laika_offline_ephem_at_start(self):
    # Test offline but process ephemeris msgs of segment first
    laikad = Laikad(use_internet=False)
    ephemeris_logs = [m for m in self.logs if m.ubloxGnss.which() == 'ephemeris']
    msgs = process_msgs(ephemeris_logs+self.logs, laikad)
    correct_msgs = [m for m in msgs if len(m.gnssMeasurements.correctedMeasurements) > 0]
    self.assertEqual(554, len(correct_msgs))
    self.assertGreaterEqual(554, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))


if __name__ == "__main__":
  unittest.main()
