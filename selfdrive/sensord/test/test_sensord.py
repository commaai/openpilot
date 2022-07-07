#!/usr/bin/env python3

import time
import unittest

import cereal.messaging as messaging
from system.hardware import TICI
from selfdrive.test.helpers import with_processes

TEST_TIMESPAN = 10

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


class TestSensord(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  @with_processes(['sensord'])
  def test_sensors_present(self):
    sensor_events = messaging.sub_sock("sensorEvents", timeout=0.1)

    start_time_sec = time.time()
    events = []
    while time.time() - start_time_sec < TEST_TIMESPAN:
      events += messaging.drain_sock(sensor_events)
      time.sleep(0.01)

    seen = set()
    for event in events:
      for measurement in event.sensorEvents:
        # Filter out unset events
        if measurement.version == 0:
          continue
        seen.add((str(measurement.source), measurement.which()))

    self.assertIn(seen, SENSOR_CONFIGURATIONS)


if __name__ == "__main__":
  unittest.main()
