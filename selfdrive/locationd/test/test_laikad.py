#!/usr/bin/env python3
import unittest
from datetime import datetime
from unittest import mock
from unittest.mock import Mock

from laika.ephemeris import EphemerisType
from laika.gps_time import GPSTime
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement, read_raw_ublox
from selfdrive.locationd.laikad import Laikad, create_measurement_msg
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader


def get_log(segs=range(0)):
  logs = []
  for i in segs:
    logs.extend(LogReader(get_url("4cf7a6ad03080c90|2021-09-29--13-46-36", i)))
  return [m for m in logs if m.which() == 'ubloxGnss']


def verify_messages(lr, laikad):
  good_msgs = []
  for m in lr:
    msg = laikad.process_ublox_msg(m.ubloxGnss, m.logMonoTime, block=True)
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
    laikad = Laikad(auto_update=True, valid_ephem_types=EphemerisType.ULTRA_RAPID_ORBIT)
    correct_msgs = verify_messages(self.logs, laikad)

    correct_msgs_expected = 560
    self.assertEqual(correct_msgs_expected, len(correct_msgs))
    self.assertEqual(correct_msgs_expected, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_laika_online_nav_only(self):
    laikad = Laikad(auto_update=True, valid_ephem_types=EphemerisType.NAV)
    # Disable fetch_orbits to test NAV only
    laikad.fetch_orbits = Mock()
    correct_msgs = verify_messages(self.logs, laikad)
    correct_msgs_expected = 560
    self.assertEqual(correct_msgs_expected, len(correct_msgs))
    self.assertEqual(correct_msgs_expected, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  @mock.patch('laika.downloader.download_and_cache_file')
  def test_laika_offline(self, downloader_mock):
    downloader_mock.side_effect = IOError
    laikad = Laikad(auto_update=False)
    correct_msgs = verify_messages(self.logs, laikad)
    self.assertEqual(256, len(correct_msgs))
    self.assertEqual(256, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_laika_get_orbits(self):
    laikad = Laikad(auto_update=False)
    first_gps_time = None
    for m in self.logs:
      if m.ubloxGnss.which == 'measurementReport':
        new_meas = read_raw_ublox(m.ubloxGnss.measurementReport)
        if len(new_meas) != 0:
          first_gps_time = new_meas[0].recv_time
          break
    # Pretend process has loaded the orbits on startup by using the time of the first gps message.
    laikad.fetch_orbits(first_gps_time, block=True)
    self.assertEqual(29, len(laikad.astro_dog.orbits.keys()))

  @unittest.skip("Use to debug live data")
  def test_laika_get_orbits_now(self):
    laikad = Laikad(auto_update=False)
    laikad.fetch_orbits(GPSTime.from_datetime(datetime.utcnow()), block=True)
    prn = "G01"
    self.assertLess(0, len(laikad.astro_dog.orbits[prn]))
    prn = "R01"
    self.assertLess(0, len(laikad.astro_dog.orbits[prn]))
    print(min(laikad.astro_dog.orbits[prn], key=lambda e: e.epoch).epoch.as_datetime())


if __name__ == "__main__":
  unittest.main()
