import unittest
import time
import numpy as np

from laika import AstroDog
from laika.helpers import ConstellationId
from laika.raw_gnss import correct_measurements, process_measurements, read_raw_ublox
from laika.opt import calc_pos_fix
from selfdrive.test.openpilotci import get_url
from tools.lib.logreader import LogReader
from selfdrive.test.helpers import with_processes
import cereal.messaging as messaging

def get_gnss_measurements(log_reader):
  gnss_measurements = []
  for msg in log_reader:
    if msg.which() == "ubloxGnss":
      ublox_msg = msg.ubloxGnss
      if ublox_msg.which == 'measurementReport':
        report = ublox_msg.measurementReport
        if len(report.measurements) > 0:
          gnss_measurements.append(read_raw_ublox(report))
  return gnss_measurements

def get_ublox_raw(log_reader):
  ublox_raw = []
  for msg in log_reader:
    if msg.which() == "ubloxRaw":
      ublox_raw.append(msg)
  return ublox_raw

class TestUbloxProcessing(unittest.TestCase):
  NUM_TEST_PROCESS_MEAS = 10

  @classmethod
  def setUpClass(cls):
    lr = LogReader(get_url("4cf7a6ad03080c90|2021-09-29--13-46-36", 0))
    cls.gnss_measurements = get_gnss_measurements(lr)

    # test gps ephemeris continuity check (drive has ephemeris issues with cutover data)
    lr = LogReader(get_url("37b6542f3211019a|2023-01-15--23-45-10", 14))
    cls.ublox_raw = get_ublox_raw(lr)

  def test_read_ublox_raw(self):
    count_gps = 0
    count_glonass = 0
    for measurements in self.gnss_measurements:
      for m in measurements:
        if m.constellation_id == ConstellationId.GPS:
          count_gps += 1
        elif m.constellation_id == ConstellationId.GLONASS:
          count_glonass += 1

    self.assertEqual(count_gps, 5036)
    self.assertEqual(count_glonass, 3651)

  def test_get_fix(self):
    dog = AstroDog()
    position_fix_found = 0
    count_processed_measurements = 0
    count_corrected_measurements = 0
    position_fix_found_after_correcting = 0

    pos_ests = []
    for measurements in self.gnss_measurements[:self.NUM_TEST_PROCESS_MEAS]:
      processed_meas = process_measurements(measurements, dog)
      count_processed_measurements += len(processed_meas)
      pos_fix = calc_pos_fix(processed_meas)
      if len(pos_fix) > 0 and all(p != 0 for p in pos_fix[0]):
        position_fix_found += 1

        corrected_meas = correct_measurements(processed_meas, pos_fix[0][:3], dog)
        count_corrected_measurements += len(corrected_meas)

        pos_fix = calc_pos_fix(corrected_meas)
        if len(pos_fix) > 0 and all(p != 0 for p in pos_fix[0]):
          pos_ests.append(pos_fix[0])
          position_fix_found_after_correcting += 1

    mean_fix = np.mean(np.array(pos_ests)[:, :3], axis=0)
    np.testing.assert_allclose(mean_fix, [-2452306.662377, -4778343.136806, 3428550.090557], rtol=0, atol=1)

    # Note that can happen that there are less corrected measurements compared to processed when they are invalid.
    # However, not for the current segment
    self.assertEqual(position_fix_found, self.NUM_TEST_PROCESS_MEAS)
    self.assertEqual(position_fix_found_after_correcting, self.NUM_TEST_PROCESS_MEAS)
    self.assertEqual(count_processed_measurements, 69)
    self.assertEqual(count_corrected_measurements, 69)

  @with_processes(['ubloxd'])
  def test_ublox_gps_cutover(self):
    time.sleep(2)
    ugs = messaging.sub_sock("ubloxGnss", timeout=0.1)
    ur_pm = messaging.PubMaster(['ubloxRaw'])

    def replay_segment():
      rcv_msgs = []
      for msg in self.ublox_raw:
        ur_pm.send(msg.which(), msg.as_builder())
        time.sleep(0.01)
        rcv_msgs += messaging.drain_sock(ugs)

      time.sleep(0.1)
      rcv_msgs += messaging.drain_sock(ugs)
      return rcv_msgs

    # replay twice to enforce cutover data on rewind
    rcv_msgs = replay_segment()
    rcv_msgs += replay_segment()

    ephems_cnt = sum(m.ubloxGnss.which() == 'ephemeris' for m in rcv_msgs)
    self.assertEqual(ephems_cnt, 15)

if __name__ == "__main__":
  unittest.main()
