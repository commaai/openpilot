#!/usr/bin/env python3
import time
import unittest
from cereal import log
from common.params import Params
from datetime import datetime
from unittest import mock


from laika.constants import SECS_IN_DAY
from laika.downloader import DownloadFailed
from laika.ephemeris import EphemerisType
from laika.gps_time import GPSTime
from laika.helpers import ConstellationId
from laika.raw_gnss import GNSSMeasurement, read_raw_ublox, read_raw_qcom
from selfdrive.locationd.laikad import EPHEMERIS_CACHE, Laikad
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader

from selfdrive.test.process_replay.process_replay import get_process_config, replay_process

GPS_TIME_PREDICTION_ORBITS_RUSSIAN_SRC = GPSTime.from_datetime(datetime(2022, month=1, day=29, hour=12))


def get_log_ublox(segs=range(0)):
  logs = []
  for i in segs:
    logs.extend(LogReader(get_url("4cf7a6ad03080c90|2021-09-29--13-46-36", i)))

  ublox_cfg = get_process_config("ubloxd")
  all_logs = replay_process(ublox_cfg, logs)
  low_gnss = []
  for m in all_logs:
    if m.which() != "ubloxGnss" or m.ubloxGnss.which() != 'measurementReport':
      continue

    MAX_MEAS = 7
    if m.ubloxGnss.measurementReport.numMeas > MAX_MEAS:
      mb = log.Event.new_message(ubloxGnss=m.ubloxGnss.to_dict())
      mb.logMonoTime = m.logMonoTime
      mb.ubloxGnss.measurementReport.numMeas = MAX_MEAS
      mb.ubloxGnss.measurementReport.measurements = list(m.ubloxGnss.measurementReport.measurements)[:MAX_MEAS]
      mb.ubloxGnss.measurementReport.measurements[0].pseudorange += 1000
      low_gnss.append(mb.as_reader())
    else:
      low_gnss.append(m)
  return all_logs, low_gnss


def get_log_qcom(segs=range(0)):
  logs = []
  for i in segs:
    logs.extend(LogReader(get_url("b0b3cba7abf862d1|2023-03-11--09-40-33", i)))
  all_logs = [m for m in logs if m.which() == 'qcomGnss']
  return all_logs


def verify_messages(lr, laikad, return_one_success=False):
  good_msgs = []
  for m in lr:
    if m.which() == 'ubloxGnss':
      gnss_msg = m.ubloxGnss
    elif m.which() == 'qcomGnss':
      gnss_msg = m.qcomGnss
    else:
      continue
    msg = laikad.process_gnss_msg(gnss_msg, m.logMonoTime, block=True)
    if msg is not None and len(msg.gnssMeasurements.correctedMeasurements) > 0:
      good_msgs.append(msg)
      if return_one_success:
        return msg
  return good_msgs


def get_first_gps_time(logs):
  for m in logs:
    if m.which() == 'ubloxGnss':
      if m.ubloxGnss.which == 'measurementReport':
        new_meas = read_raw_ublox(m.ubloxGnss.measurementReport)
        if len(new_meas) > 0:
          return new_meas[0].recv_time
    elif m.which() == "qcomGnss":
      if m.qcomGnss.which == 'measurementReport':
        new_meas = read_raw_qcom(m.qcomGnss.measurementReport)
        if len(new_meas) > 0:
          return new_meas[0].recv_time


def get_measurement_mock(gpstime, sat_ephemeris):
  meas = GNSSMeasurement(ConstellationId.GPS, 1, gpstime.week, gpstime.tow, {'C1C': 0., 'D1C': 0.}, {'C1C': 0., 'D1C': 0.})
  # Fake measurement being processed
  meas.observables_final = meas.observables
  meas.sat_ephemeris = sat_ephemeris
  return meas


class TestLaikad(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    logs, low_gnss = get_log_ublox(range(1))
    cls.logs = logs
    cls.low_gnss = low_gnss
    cls.logs_qcom = get_log_qcom(range(1))
    first_gps_time = get_first_gps_time(logs)
    cls.first_gps_time = first_gps_time

  def setUp(self):
    Params().remove(EPHEMERIS_CACHE)

  def test_fetch_navs_non_blocking(self):
    gpstime = GPSTime.from_datetime(datetime(2021, month=3, day=1))
    laikad = Laikad()
    laikad.fetch_navs(gpstime, block=False)
    laikad.orbit_fetch_future.result(30)
    
    # Get results and save orbits to laikad:
    laikad.fetch_navs(gpstime, block=False)
    ephem = laikad.astro_dog.navs['G01'][0]
    self.assertIsNotNone(ephem)

    laikad.fetch_navs(gpstime+2*SECS_IN_DAY, block=False)
    laikad.orbit_fetch_future.result(30)
    # Get results and save orbits to laikad:
    laikad.fetch_navs(gpstime + 2 * SECS_IN_DAY, block=False)

    ephem2 = laikad.astro_dog.navs['G01'][0]
    self.assertIsNotNone(ephem)
    self.assertNotEqual(ephem, ephem2)


  def test_fetch_navs_with_wrong_clocks(self):
    laikad = Laikad()

    def check_has_navs():
      self.assertGreater(len(laikad.astro_dog.navs), 0)
      ephem = laikad.astro_dog.navs['G01'][0]
      self.assertIsNotNone(ephem)
    real_current_time = GPSTime.from_datetime(datetime(2021, month=3, day=1))
    wrong_future_clock_time = real_current_time + SECS_IN_DAY

    laikad.fetch_navs(wrong_future_clock_time, block=True)
    check_has_navs()
    self.assertEqual(laikad.last_fetch_navs_t, wrong_future_clock_time)

    # Test fetching orbits with earlier time
    assert real_current_time < laikad.last_fetch_navs_t

    laikad.astro_dog.orbits = {}
    laikad.fetch_navs(real_current_time, block=True)
    check_has_navs()
    self.assertEqual(laikad.last_fetch_navs_t, real_current_time)

  def test_laika_online(self):
    laikad = Laikad(auto_update=True, valid_ephem_types=EphemerisType.ULTRA_RAPID_ORBIT)
    correct_msgs = verify_messages(self.logs, laikad)

    correct_msgs_expected = 560
    self.assertEqual(correct_msgs_expected, len(correct_msgs))
    self.assertEqual(correct_msgs_expected, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_kf_becomes_valid(self):
    laikad = Laikad(auto_update=False)
    m = self.logs[0]
    self.assertFalse(all(laikad.kf_valid(m.logMonoTime * 1e-9)))
    kf_valid = False
    for m in self.logs:
      if m.which() != 'ubloxGnss':
        continue

      laikad.process_gnss_msg(m.ubloxGnss, m.logMonoTime, block=True)
      kf_valid = all(laikad.kf_valid(m.logMonoTime * 1e-9))
      if kf_valid:
        break
    self.assertTrue(kf_valid)

  def test_laika_online_nav_only(self):
    for use_qcom, logs in zip([True, False], [self.logs_qcom, self.logs]):
      laikad = Laikad(auto_update=True, valid_ephem_types=EphemerisType.NAV, use_qcom=use_qcom)
      # Disable fetch_orbits to test NAV only
      correct_msgs = verify_messages(logs, laikad)
      correct_msgs_expected = 44 if use_qcom else 560
      valid_fix_expected = 43 if use_qcom else 560

      self.assertEqual(correct_msgs_expected, len(correct_msgs))
      self.assertEqual(valid_fix_expected, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  @mock.patch('laika.downloader.download_and_cache_file')
  def test_laika_offline(self, downloader_mock):
    downloader_mock.side_effect = DownloadFailed("Mock download failed")
    laikad = Laikad(auto_update=False)
    laikad.fetch_navs(GPS_TIME_PREDICTION_ORBITS_RUSSIAN_SRC, block=True)

  @mock.patch('laika.downloader.download_and_cache_file')
  def test_download_failed_russian_source(self, downloader_mock):
    downloader_mock.side_effect = DownloadFailed
    laikad = Laikad(auto_update=False)
    correct_msgs = verify_messages(self.logs, laikad)
    expected_msgs = 376
    self.assertEqual(expected_msgs, len(correct_msgs))
    self.assertEqual(expected_msgs, len([m for m in correct_msgs if m.gnssMeasurements.positionECEF.valid]))

  def test_laika_get_orbits(self):
    laikad = Laikad(auto_update=False)
    # Pretend process has loaded the orbits on startup by using the time of the first gps message.
    laikad.fetch_navs(self.first_gps_time, block=True)
    self.dict_has_values(laikad.astro_dog.navs)

  @unittest.skip("Use to debug live data")
  def test_laika_get_navs_now(self):
    laikad = Laikad(auto_update=False)
    laikad.fetch_navs(GPSTime.from_datetime(datetime.utcnow()), block=True)
    prn = "G01"
    self.assertGreater(len(laikad.astro_dog.navs[prn]), 0)
    prn = "R01"
    self.assertGreater(len(laikad.astro_dog.navs[prn]), 0)

  def test_get_navs_in_process(self):
    for auto_fetch_navs in [True, False]:
      for use_qcom, logs in zip([True, False], [self.logs_qcom, self.logs]):
        laikad = Laikad(auto_update=False, use_qcom=use_qcom, auto_fetch_navs=auto_fetch_navs)
        has_navs = False
        has_fix = False
        seen_chip_eph = False
        seen_internet_eph = False

        for m in logs:
          if m.which() != 'ubloxGnss' and m.which() != 'qcomGnss':
            continue

          gnss_msg = m.qcomGnss if use_qcom else m.ubloxGnss
          out_msg = laikad.process_gnss_msg(gnss_msg, m.logMonoTime, block=False)
          if laikad.orbit_fetch_future is not None:
            laikad.orbit_fetch_future.result()
          vals = laikad.astro_dog.navs.values()
          has_navs = len(vals) > 0 and max([len(v) for v in vals]) > 0
          vals = laikad.astro_dog.qcom_polys.values()
          has_polys = len(vals) > 0 and max([len(v) for v in vals]) > 0
          has_fix = has_fix or out_msg.gnssMeasurements.positionECEF.valid
          if len(out_msg.gnssMeasurements.ephemerisStatuses):
            seen_chip_eph = seen_chip_eph or any([x.source == 'gnssChip' for x in out_msg.gnssMeasurements.ephemerisStatuses])
            seen_internet_eph = seen_internet_eph or any([x.source == 'internet' for x in out_msg.gnssMeasurements.ephemerisStatuses])
            
        self.assertTrue(has_navs or has_polys)
        self.assertTrue(has_fix)
        self.assertTrue(seen_chip_eph or auto_fetch_navs)
        self.assertTrue(seen_internet_eph or not auto_fetch_navs)
        self.assertEqual(len(laikad.astro_dog.navs_fetched_times._ranges), 0)
        self.assertEqual(None, laikad.orbit_fetch_future)

  def test_cache(self):
    use_qcom = True
    for use_qcom, logs in zip([True, False], [self.logs_qcom, self.logs]):
      laikad = Laikad(auto_update=True, save_ephemeris=True, use_qcom=use_qcom)
      def wait_for_cache():
        max_time = 2
        while Params().get(EPHEMERIS_CACHE) is None:
          time.sleep(0.1)
          max_time -= 0.1
          if max_time < 0:
            self.fail("Cache has not been written after 2 seconds")

      # Test cache with no ephemeris
      laikad.last_report_time = GPSTime(1,0)
      laikad.cache_ephemeris()
      wait_for_cache()
      Params().remove(EPHEMERIS_CACHE)

      #laikad.astro_dog.get_navs(self.first_gps_time)
      msg = verify_messages(logs, laikad, return_one_success=True)
      laikad.cache_ephemeris()
      wait_for_cache()

      # Check both nav and orbits separate
      laikad = Laikad(auto_update=False, valid_ephem_types=EphemerisType.NAV,
                      save_ephemeris=True, use_qcom=use_qcom, auto_fetch_navs=False)
      # Verify navs are loaded from cache
      self.dict_has_values(laikad.astro_dog.navs)
      # Verify cache is working for only nav by running a segment
      msg = verify_messages(logs, laikad, return_one_success=True)
      self.assertTrue(len(msg.gnssMeasurements.ephemerisStatuses))
      self.assertTrue(any([x.source=='cache' for x in msg.gnssMeasurements.ephemerisStatuses]))
      self.assertIsNotNone(msg)

      #TODO test cache with only orbits 
      #with patch('selfdrive.locationd.laikad.get_orbit_data', return_value=None) as mock_method:
      #  # Verify no orbit downloads even if orbit fetch times is reset since the cache has recently been saved and we don't want to download high frequently
      #  laikad.astro_dog.orbit_fetched_times = TimeRangeHolder()
      #  laikad.fetch_navs(self.first_gps_time, block=False)
      #  mock_method.assert_not_called()

      #  # Verify cache is working for only orbits by running a segment
      #  laikad = Laikad(auto_update=False, valid_ephem_types=EphemerisType.ULTRA_RAPID_ORBIT, save_ephemeris=True)
      #  msg = verify_messages(self.logs, laikad, return_one_success=True)
      #  self.assertIsNotNone(msg)
      #  # Verify orbit data is not downloaded
      #  mock_method.assert_not_called()
      #break

  def test_low_gnss_meas(self):
    cnt = 0
    laikad = Laikad()
    for m in self.low_gnss:
      msg = laikad.process_gnss_msg(m.ubloxGnss, m.logMonoTime, block=True)
      if msg is None:
        continue
      gm = msg.gnssMeasurements
      if len(gm.correctedMeasurements) != 0 and gm.positionECEF.valid:
        cnt += 1
    self.assertEqual(cnt, 560)

  def dict_has_values(self, dct):
    self.assertGreater(len(dct), 0)
    self.assertGreater(min([len(v) for v in dct.values()]), 0)


if __name__ == "__main__":
  unittest.main()
