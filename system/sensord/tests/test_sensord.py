#!/usr/bin/env python3
import os
import time
import unittest
import numpy as np
from collections import namedtuple, defaultdict

import cereal.messaging as messaging
from cereal import log
from common.gpio import get_irq_for_action
from system.hardware import TICI
from selfdrive.manager.process_config import managed_processes

BMX = {
  ('bmx055', 'acceleration'),
  ('bmx055', 'gyroUncalibrated'),
  ('bmx055', 'magneticUncalibrated'),
  ('bmx055', 'temperature'),
}

LSM = {
  ('lsm6ds3', 'acceleration'),
  ('lsm6ds3', 'gyroUncalibrated'),
  ('lsm6ds3', 'temperature'),
}
LSM_C = {(x[0]+'trc', x[1]) for x in LSM}

MMC = {
  ('mmc5603nj', 'magneticUncalibrated'),
}

RPR = {
  ('rpr0521', 'light'),
}

SENSOR_CONFIGURATIONS = (
  (BMX | LSM | RPR),
  (MMC | LSM | RPR),
  (BMX | LSM_C | RPR),
  (MMC| LSM_C | RPR),
)

Sensor = log.SensorEventData.SensorSource
SensorConfig = namedtuple('SensorConfig', ['type', 'sanity_min', 'sanity_max', 'expected_freq'])
ALL_SENSORS = {
  Sensor.rpr0521: {
    SensorConfig("light", 0, 1023, 100),
  },

  Sensor.lsm6ds3: {
    SensorConfig("acceleration", 5, 15, 100),
    SensorConfig("gyroUncalibrated", 0, .2, 100),
    SensorConfig("temperature", 0, 60, 100),
  },

  Sensor.lsm6ds3trc: {
    SensorConfig("acceleration", 5, 15, 104),
    SensorConfig("gyroUncalibrated", 0, .2, 104),
    SensorConfig("temperature", 0, 60, 100),
  },

  Sensor.bmx055: {
    SensorConfig("acceleration", 5, 15, 100),
    SensorConfig("gyroUncalibrated", 0, .2, 100),
    SensorConfig("magneticUncalibrated", 0, 300, 100),
    SensorConfig("temperature", 0, 60, 100),
  },

  Sensor.mmc5603nj: {
    SensorConfig("magneticUncalibrated", 0, 300, 100),
  }
}


def get_irq_count(irq: int):
  with open(f"/sys/kernel/irq/{irq}/per_cpu_count") as f:
    per_cpu = map(int, f.read().split(","))
    return sum(per_cpu)

def read_sensor_events(duration_sec):
  sensor_types = ['accelerometer', 'gyroscope', 'magnetometer', 'accelerometer2',
                  'gyroscope2', 'lightSensor', 'temperatureSensor']
  esocks = {}
  events = defaultdict(list)
  for stype in sensor_types:
    esocks[stype] = messaging.sub_sock(stype, timeout=0.1)

  start_time_sec = time.monotonic()
  while time.monotonic() - start_time_sec < duration_sec:
    for esock in esocks:
      events[esock] += messaging.drain_sock(esocks[esock])
    time.sleep(0.1)

  assert sum(map(len, events.values())) != 0, "No sensor events collected!"

  return events

class TestSensord(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not TICI:
      raise unittest.SkipTest

    # enable LSM self test
    os.environ["LSM_SELF_TEST"] = "1"

    # read initial sensor values every test case can use
    os.system("pkill -f ./_sensord")
    try:
      managed_processes["sensord"].start()
      time.sleep(3)
      cls.sample_secs = 10
      cls.events = read_sensor_events(cls.sample_secs)

      # determine sensord's irq
      cls.sensord_irq = get_irq_for_action("sensord")[0]
    finally:
      # teardown won't run if this doesn't succeed
      managed_processes["sensord"].stop()

  @classmethod
  def tearDownClass(cls):
    managed_processes["sensord"].stop()

  def tearDown(self):
    managed_processes["sensord"].stop()

  def test_sensors_present(self):
    # verify correct sensors configuration

    seen = set()
    for etype in self.events:
      for measurement in self.events[etype]:
        m = getattr(measurement, measurement.which())
        seen.add((str(m.source), m.which()))

    self.assertIn(seen, SENSOR_CONFIGURATIONS)

  def test_lsm6ds3_timing(self):
    # verify measurements are sampled and published at 104Hz

    sensor_t = {
      1: [], # accel
      5: [], # gyro
    }

    for measurement in self.events['accelerometer']:
      m = getattr(measurement, measurement.which())
      sensor_t[m.sensor].append(m.timestamp)

    for measurement in self.events['gyroscope']:
      m = getattr(measurement, measurement.which())
      sensor_t[m.sensor].append(m.timestamp)

    for s, vals in sensor_t.items():
      with self.subTest(sensor=s):
        assert len(vals) > 0
        tdiffs = np.diff(vals) / 1e6 # millis

        high_delay_diffs = list(filter(lambda d: d >= 20., tdiffs))
        assert len(high_delay_diffs) < 15, f"Too many large diffs: {high_delay_diffs}"

        avg_diff = sum(tdiffs)/len(tdiffs)
        avg_freq = 1. / (avg_diff * 1e-3)
        assert 92. < avg_freq < 114., f"avg freq {avg_freq}Hz wrong, expected 104Hz"

        stddev = np.std(tdiffs)
        assert stddev < 2.0, f"Standard-dev to big {stddev}"

  def test_events_check(self):
    # verify if all sensors produce events

    sensor_events = dict()
    for etype in self.events:
      for measurement in self.events[etype]:
        m = getattr(measurement, measurement.which())

        if m.type in sensor_events:
          sensor_events[m.type] += 1
        else:
          sensor_events[m.type] = 1

    for s in sensor_events:
      err_msg = f"Sensor {s}: 200 < {sensor_events[s]}"
      assert sensor_events[s] > 200, err_msg

  def test_logmonottime_timestamp_diff(self):
    # ensure diff between the message logMonotime and sample timestamp is small

    tdiffs = list()
    for etype in self.events:
      for measurement in self.events[etype]:
        m = getattr(measurement, measurement.which())

        # check if gyro and accel timestamps are before logMonoTime
        if str(m.source).startswith("lsm6ds3") and m.which() != 'temperature':
          err_msg = f"Timestamp after logMonoTime: {m.timestamp} > {measurement.logMonoTime}"
          assert m.timestamp < measurement.logMonoTime, err_msg

        # negative values might occur, as non interrupt packages created
        # before the sensor is read
        tdiffs.append(abs(measurement.logMonoTime - m.timestamp) / 1e6)

    high_delay_diffs = set(filter(lambda d: d >= 15., tdiffs))
    assert len(high_delay_diffs) < 20, f"Too many measurements published : {high_delay_diffs}"

    avg_diff = round(sum(tdiffs)/len(tdiffs), 4)
    assert avg_diff < 4, f"Avg packet diff: {avg_diff:.1f}ms"

    stddev = np.std(tdiffs)
    assert stddev < 2, f"Timing diffs have too high stddev: {stddev}"

  def test_sensor_values_sanity_check(self):
    sensor_values = dict()
    for etype in self.events:
      for measurement in self.events[etype]:
        m = getattr(measurement, measurement.which())
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
        min_samples = self.sample_secs * s.expected_freq
        err_msg = f"Sensor {sensor} {s.type} got {val_cnt} measurements, expected {min_samples}"
        assert min_samples*0.9 < val_cnt < min_samples*1.1, err_msg

        mean_norm = np.mean(np.linalg.norm(sensor_values[key], axis=1))
        err_msg = f"Sensor '{sensor} {s.type}' failed sanity checks {mean_norm} is not between {s.sanity_min} and {s.sanity_max}"
        assert s.sanity_min <= mean_norm <= s.sanity_max, err_msg

  def test_sensor_verify_no_interrupts_after_stop(self):
    managed_processes["sensord"].start()
    time.sleep(3)

    # read /proc/interrupts to verify interrupts are received
    state_one = get_irq_count(self.sensord_irq)
    time.sleep(1)
    state_two = get_irq_count(self.sensord_irq)

    error_msg = f"no interrupts received after sensord start!\n{state_one} {state_two}"
    assert state_one != state_two, error_msg

    managed_processes["sensord"].stop()
    time.sleep(1)

    # read /proc/interrupts to verify no more interrupts are received
    state_one = get_irq_count(self.sensord_irq)
    time.sleep(1)
    state_two = get_irq_count(self.sensord_irq)
    assert state_one == state_two, "Interrupts received after sensord stop!"


if __name__ == "__main__":
  unittest.main()
