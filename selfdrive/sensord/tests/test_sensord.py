#!/usr/bin/env python3

import time
import unittest

import cereal.messaging as messaging
from system.hardware import TICI
from selfdrive.test.helpers import with_processes

TEST_SENORS_PRESENT_TIMESPAN = 10

SENSOR_CONFIGURATIONS = (
  {
    ('bmx055', 'acceleration'),
    ('bmx055', 'gyroUncalibrated'),
    ('bmx055', 'magneticUncalibrated'),
    ('bmx055', 'temperature'),
    ('lsm6ds3', 'acceleration'),
    ('lsm6ds3', 'gyroUncalibrated'),
    ('lsm6ds3', 'temperature'),
    ('rpr0521', 'light'),
  },
  {
    ('lsm6ds3', 'acceleration'),
    ('lsm6ds3', 'gyroUncalibrated'),
    ('lsm6ds3', 'temperature'),
    ('mmc5603nj', 'magneticUncalibrated'),
    ('rpr0521', 'light'),
  },
  {
    ('bmx055', 'acceleration'),
    ('bmx055', 'gyroUncalibrated'),
    ('bmx055', 'magneticUncalibrated'),
    ('bmx055', 'temperature'),
    ('lsm6ds3trc', 'acceleration'),
    ('lsm6ds3trc', 'gyroUncalibrated'),
    ('lsm6ds3trc', 'temperature'),
    ('rpr0521', 'light'),
  },
  {
    ('lsm6ds3trc', 'acceleration'),
    ('lsm6ds3trc', 'gyroUncalibrated'),
    ('lsm6ds3trc', 'temperature'),
    ('mmc5603nj', 'magneticUncalibrated'),
    ('rpr0521', 'light'),
  },
)

SENSOR_TYPE_AMBIENT_TEMPERATURE = 13

def read_sensor_events(duration_sec):
  sensor_events = messaging.sub_sock("sensorEvents", timeout=0.1)
  start_time_sec = time.time()
  events = []
  while time.time() - start_time_sec < duration_sec:
    events += messaging.drain_sock(sensor_events)
    time.sleep(0.01)
  return events

class TestSensord(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  @with_processes(['sensord'])
  def test_sensors_present(self):
    # verify correct sensors configuration
    events = read_sensor_events(TEST_SENORS_PRESENT_TIMESPAN)

    seen = set()
    for event in events:
      for measurement in event.sensorEvents:
        # Filter out unset events
        if measurement.version == 0:
          continue
        seen.add((str(measurement.source), measurement.which()))

    self.assertIn(seen, SENSOR_CONFIGURATIONS)

  @with_processes(['sensord'])
  def test_lsm6ds3_100Hz(self):
    # verify samples arrive in a 100Hz rate
    events = read_sensor_events(3) # 3sec (about 300 measurements)

    data_points = set()
    for event in events:
      for measurement in event.sensorEvents:
        # Filter out unset events
        if measurement.version == 0:
          continue

        # skip lsm6ds3 temperature measurements
        if measurement.type == SENSOR_TYPE_AMBIENT_TEMPERATURE:
          continue

        if str(measurement.source).startswith("lsm6ds3"):
          data_points.add(measurement.timestamp)

    # 3s/10ms = 300 samples, check for that (may vary slightly)
    print("lsm6ds3 measurements: {}".format(len(data_points)))
    self.assertTrue(len(data_points) > 290 and len(data_points) < 310)

    data_list = list(data_points)
    data_list.sort()

    # Calc differences between measurements
    tdiffs = list()
    lt = data_list[-1]
    for t in data_list[:-1][::-1]:
      tdiffs.append(lt - t)
      lt = t

    # get out Frequency
    avg_freq = 0
    for td in tdiffs:
      avg_freq += 1/td * 10**9
    avg_freq /= len(tdiffs)

    print("avg Freq: {}".format(round(avg_freq, 4)))

    # lsm6ds3 sensor is set to trigger at 104Hz rate so it cant get higher,
    # it also shouldnt be lower than 100 Hz, delay comes from the reading
    self.assertTrue(avg_freq > 100 and avg_freq < 104)

if __name__ == "__main__":
  unittest.main()
