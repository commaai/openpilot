#!/usr/bin/env python3

import time
import unittest
import numpy as np
from collections import namedtuple

import cereal.messaging as messaging
from cereal import log
from system.hardware import TICI
from selfdrive.test.helpers import with_processes

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

Sensor = log.SensorEventData.SensorSource
SensorConfig = namedtuple('SensorConfig', ['type', 'min_samples', 'sanity_min', 'sanity_max'])
ALL_SENSORS = {
  Sensor.rpr0521: {
    SensorConfig("light", 100, 0, 150),
  },

  Sensor.lsm6ds3: {
    SensorConfig("acceleration", 100, 5, 15),
    SensorConfig("gyroUncalibrated", 100, 0, .2),
    SensorConfig("temperature", 100, 0, 60),
  },

  Sensor.lsm6ds3trc: {
    SensorConfig("acceleration", 100, 5, 15),
    SensorConfig("gyroUncalibrated", 100, 0, .2),
    SensorConfig("temperature", 100, 0, 60),
  },

  Sensor.bmx055: {
    SensorConfig("acceleration", 100, 5, 15),
    SensorConfig("gyroUncalibrated", 100, 0, .2),
    SensorConfig("magneticUncalibrated", 100, 0, 300),
    SensorConfig("temperature", 100, 0, 60),
  },

  Sensor.mmc5603nj: {
    SensorConfig("magneticUncalibrated", 100, 0, 300),
  }
}

def read_sensor_events(duration_sec):
  sensor_events = messaging.sub_sock("sensorEvents", timeout=0.1)
  start_time_sec = time.monotonic()
  events = []
  while time.monotonic() - start_time_sec < duration_sec:
    events += messaging.drain_sock(sensor_events)
    time.sleep(0.01)

  assert len(events) != 0, "No sensor events collected"
  return events

class TestSensord(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

  @with_processes(['sensord'])
  def test_sensors_present(self):
    # verify correct sensors configuration
    events = read_sensor_events(10)

    seen = set()
    for event in events:
      for measurement in event.sensorEvents:
        # filter unset events (bmx magn)
        if measurement.version == 0:
          continue
        seen.add((str(measurement.source), measurement.which()))

    self.assertIn(seen, SENSOR_CONFIGURATIONS)

  @with_processes(['sensord'])
  def test_lsm6ds3_100Hz(self):
    # verify measurements are sampled and published at a 100Hz rate
    events = read_sensor_events(3) # 3sec (about 300 measurements)

    data_points = set()
    for event in events:
      for measurement in event.sensorEvents:

        # skip lsm6ds3 temperature measurements
        if measurement.which() == 'temperature':
          continue

        if str(measurement.source).startswith("lsm6ds3"):
          data_points.add(measurement.timestamp)

    assert len(data_points) != 0, "No lsm6ds3 sensor events"

    data_list = list(data_points)
    data_list.sort()
    tdiffs = np.diff(data_list)

    high_delay_diffs = list(filter(lambda d: d >= 10*10**6, tdiffs))
    assert len(high_delay_diffs) < 10, f"Too many high delay packages: {high_delay_diffs}"

    avg_diff = sum(tdiffs)/len(tdiffs)
    assert avg_diff > 9.6*10**6, f"avg difference {avg_diff}, below threshold"

    stddev = np.std(tdiffs)
    assert stddev < 600*10**3, f"Standard-dev to big {stddev}"

  @with_processes(['sensord'])
  def test_events_check(self):
    # verify if all sensors produce events
    events = read_sensor_events(3)

    sensor_events = dict()
    for event in events:
      for measurement in event.sensorEvents:

        # filter unset events (bmx magn)
        if measurement.version == 0:
          continue

        if measurement.type in sensor_events:
          sensor_events[measurement.type] += 1
        else:
          sensor_events[measurement.type] = 1

    for s in sensor_events:
      assert sensor_events[s] > 200, f"Sensor {s}: {sensor_events[s]} < 200 events"

  @with_processes(['sensord'])
  def test_logmonottime_timestamp_diff(self):
    # ensure diff between the message logMonotime and sample timestamp is small
    events = read_sensor_events(3)

    tdiffs = list()
    for event in events:
      for measurement in event.sensorEvents:

        # filter unset events (bmx magn)
        if measurement.version == 0:
          continue

        # negative values might occur, as non interrupt packages created
        tdiffs.append(abs(event.logMonoTime - measurement.timestamp))
        # before the sensor is read

    high_delay_diffs = list(filter(lambda d: d >= 10*10**6, tdiffs))
    assert len(high_delay_diffs) < 10, f"Too many high delay packages: {high_delay_diffs}"

    avg_diff = round(sum(tdiffs)/len(tdiffs), 4)
    assert avg_diff < 4*10**6, f"Avg packet diff: {avg_diff:.1f}ns"

    stddev = np.std(tdiffs)
    assert stddev < 1.5*10**6, f"Timing diffs have to high stddev: {stddev}"

  @with_processes(['sensord'])
  def test_sensor_values_sanity_check(self):

    events = read_sensor_events(2)

    sensor_values = dict()
    for event in events:
      for m in event.sensorEvents:

        # filter unset events (bmx magn)
        if m.version == 0:
          continue

        key = (m.source.raw, m.which())
        values = getattr(m, m.which())
        if hasattr(values, 'v'):
          values = values.v
        values = np.atleast_1d(values)

        if key in sensor_values:
          sensor_values[key].append(values)
        else:
          sensor_values[key] = [values]

    # Sanity check sensor values and counts
    for sensor, stype in sensor_values:

      for s in ALL_SENSORS[sensor]:
        if s.type != stype:
          continue

        key = (sensor, s.type)
        val_cnt = len(sensor_values[key])
        err_msg = f"Sensor {sensor} {s.type} got {val_cnt} measurements, expected {s.min_samples}"
        assert val_cnt > s.min_samples, err_msg

        mean_norm = np.mean(np.linalg.norm(sensor_values[key], axis=1))
        err_msg = f"Sensor '{sensor} {s.type}' failed sanity checks {mean_norm} is not between {s.sanity_min} and {s.sanity_max}"
        assert s.sanity_min <= mean_norm <= s.sanity_max, err_msg

if __name__ == "__main__":
  unittest.main()
