#!/usr/bin/env python3

import time
import unittest
import numpy

import cereal.messaging as messaging
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

SENSOR_TYPE_AMBIENT_TEMPERATURE = 13


def read_sensor_events(duration_sec):
  sensor_events = messaging.sub_sock("sensorEvents", timeout=0.1)
  start_time_sec = time.time()
  events = []
  while time.time() - start_time_sec < duration_sec:
    events += messaging.drain_sock(sensor_events)
    time.sleep(0.01)
  return events


def get_filter_bounds(values, percent):
  values.sort()
  median = int(len(values)/2)
  lb = median - int(len(values)*percent/2)
  ub = median + int(len(values)*percent/2)
  return (lb, ub)


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

    data_list = list(data_points)
    data_list.sort()

    # Calc differences between measurements
    tdiffs = list()
    lt = data_list[0]
    for t in data_list[1:]:
      tdiffs.append(t - lt)
      lt = t

    # filter 10% of the data to remove outliers
    lb, ub = get_filter_bounds(tdiffs, 0.9)
    diffs = tdiffs[lb:ub]
    avg_diff = sum(diffs)/len(diffs)
    assert avg_diff > 9.6*10**6, f"avg difference {avg_diff}, below threshold"

    # standard deviation
    stddev = numpy.std(diffs)
    assert stddev < 50000, f"Standard-dev to big {stddev}"

    # calculate average frequency
    avg_freq = 0
    for td in diffs:
      avg_freq += 1/td * 10**9
    avg_freq /= len(diffs)

    # lsm6ds3 sensor is set to trigger at 104Hz rate so it can't get higher,
    # it also shouldn't be lower than 100 Hz, delay comes from the reading
    self.assertTrue(avg_freq > 100 and avg_freq < 104)
    assert avg_freq > 100 and avg_freq < 104, f"Avg_freq out of bounds {avg_freq}"

  @with_processes(['sensord'])
  def test_events_check(self):
    # verify if all sensors produce events
    events = read_sensor_events(3)

    sensor_events = dict()
    for event in events:
      for measurement in event.sensorEvents:
        # Filter out unset events
        if measurement.version == 0:
          continue

        if measurement.type in sensor_events:
          sensor_events[measurement.type] += 1
        else:
          sensor_events[measurement.type] = 1

    for s in sensor_events:
      assert sensor_events[s] > 200, f"Sensor {s}: {sensor_events[s]} < 200 events"

  @with_processes(['sensord'])
  def test_logmonottime_timestamp(self):
    # ensure diff logMonotime and timestamp is rather small
    # -> published when created
    events = read_sensor_events(3)

    tdiffs = list()
    for event in events:
      for measurement in event.sensorEvents:
        # Filter out unset events
        if measurement.version == 0:
          continue

        tdiffs.append(abs(event.logMonoTime - measurement.timestamp))
        # negative values might occur, as non interrupt packages created
        # before the sensor is read

    # filter 10% of the data to remove outliers
    lb, ub = get_filter_bounds(tdiffs, 0.9)
    diffs = tdiffs[lb:ub]
    avg_diff = round(sum(diffs)/len(diffs), 4)

    assert max(diffs) < 10*10**6, f"packet took { max(diffs):.1f}ns for publishing"
    assert avg_diff < 4*10**6, f"Avg packet diff: {avg_diff:.1f}ns"


if __name__ == "__main__":
  unittest.main()
