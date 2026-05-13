import os
import pytest
import time
import numpy as np
from collections import namedtuple, defaultdict

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.gpio import get_irqs_for_action
from openpilot.common.timeout import Timeout
from openpilot.system.manager.process_config import managed_processes

SensorConfig = namedtuple('SensorConfig', ['service', 'measurement', 'sanity_min', 'sanity_max', 'std_max'])

SENSOR_CONFIGS = (
  SensorConfig("accelerometer", "acceleration", 5, 15, 5),
  SensorConfig("gyroscope", "gyroUncalibrated", 0, .15, 0.5),
  SensorConfig("temperatureSensor", "temperature", 10, 40, 0.5),  # set for max range of our office
)
SENSOR_CONFIGS_BY_MEASUREMENT = {config.measurement: config for config in SENSOR_CONFIGS}

def get_irq_count(irq: int):
  with open(f"/sys/kernel/irq/{irq}/per_cpu_count") as f:
    per_cpu = map(int, f.read().split(","))
    return sum(per_cpu)

def read_sensor_events(duration_sec):
  socks = {}
  poller = messaging.Poller()
  events = defaultdict(list)
  for config in SENSOR_CONFIGS:
    socks[config.service] = messaging.sub_sock(config.service, poller=poller, timeout=100)

  # wait for sensors to come up
  with Timeout(int(os.environ.get("SENSOR_WAIT", "5")), "sensors didn't come up"):
    while len(poller.poll(250)) == 0:
      pass
  time.sleep(1)
  for s in socks.values():
    messaging.drain_sock_raw(s)

  st = time.monotonic()
  while time.monotonic() - st < duration_sec:
    for s in socks:
      events[s] += messaging.drain_sock(socks[s])
    time.sleep(0.1)
  assert sum(map(len, events.values())) != 0, "No sensor events collected!"

  return {k: v for k, v in events.items() if len(v) > 0}

def iter_measurements(events):
  for msgs in events.values():
    for measurement in msgs:
      yield measurement, getattr(measurement, measurement.which())

@pytest.mark.tici
class TestSensord:
  @classmethod
  def setup_class(cls):
    # enable LSM self test
    os.environ["LSM_SELF_TEST"] = "1"

    # read initial sensor values every test case can use
    os.system("pkill -f \\\\./sensord")
    try:
      managed_processes["sensord"].start()
      cls.sample_secs = int(os.getenv("SAMPLE_SECS", "10"))
      cls.events = read_sensor_events(cls.sample_secs)

      # determine sensord's irq
      cls.sensord_irq = get_irqs_for_action("sensord")[0]
    finally:
      # teardown won't run if this doesn't succeed
      managed_processes["sensord"].stop()

  @classmethod
  def teardown_class(cls):
    managed_processes["sensord"].stop()

  def teardown_method(self):
    managed_processes["sensord"].stop()

  def test_all_sensors_present(self):
    missing = [config.service for config in SENSOR_CONFIGS if config.service not in self.events]
    assert len(missing) == 0, f"missing sensors: {missing}"

  def test_lsm6ds3_timing(self, subtests):
    # verify measurements are sampled and published at 104Hz

    sensor_t = {service: [] for service in ('accelerometer', 'gyroscope')}

    for service in sensor_t:
      for measurement in self.events.get(service, []):
        m = getattr(measurement, measurement.which())
        sensor_t[service].append(m.timestamp)

    for s, vals in sensor_t.items():
      with subtests.test(sensor=s):
        assert len(vals) > 0
        tdiffs = np.diff(vals) / 1e6 # millis

        high_delay_diffs = list(filter(lambda d: d >= 20., tdiffs))
        assert len(high_delay_diffs) < 15, f"Too many large diffs: {high_delay_diffs}"

        avg_diff = sum(tdiffs)/len(tdiffs)
        avg_freq = 1. / (avg_diff * 1e-3)
        assert 92. < avg_freq < 114., f"avg freq {avg_freq}Hz wrong, expected 104Hz"

        stddev = np.std(tdiffs)
        assert stddev < 2.0, f"Standard-dev to big {stddev}"

  def test_sensor_frequency(self, subtests):
    for s, msgs in self.events.items():
      with subtests.test(sensor=s):
        freq = len(msgs) / self.sample_secs
        ef = SERVICE_LIST[s].frequency
        assert ef*0.85 <= freq <= ef*1.15

  def test_logmonottime_timestamp_diff(self):
    # ensure diff between the message logMonotime and sample timestamp is small

    tdiffs = []
    for measurement, m in iter_measurements(self.events):
      # check if gyro and accel timestamps are before logMonoTime
      if str(m.source).startswith("lsm6ds3") and m.which() != 'temperature':
        err_msg = f"Timestamp after logMonoTime: {m.timestamp} > {measurement.logMonoTime}"
        assert m.timestamp < measurement.logMonoTime, err_msg

      # negative values might occur, as non interrupt packages created
      # before the sensor is read
      tdiffs.append(abs(measurement.logMonoTime - m.timestamp) / 1e6)

    # some sensors have a read procedure that will introduce an expected diff on the order of 20ms
    high_delay_diffs = set(filter(lambda d: d >= 25., tdiffs))
    assert len(high_delay_diffs) < 20, f"Too many measurements published: {high_delay_diffs}"

    avg_diff = round(sum(tdiffs)/len(tdiffs), 4)
    assert avg_diff < 4, f"Avg packet diff: {avg_diff:.1f}ms"

  def test_sensor_values(self):
    sensor_values = defaultdict(list)
    for _, m in iter_measurements(self.events):
      key = (m.source.raw, m.which())
      values = getattr(m, m.which())

      if hasattr(values, 'v'):
        values = values.v
      sensor_values[key].append(np.atleast_1d(values))

    # Sanity check sensor values
    for (sensor, stype), values in sensor_values.items():
      config = SENSOR_CONFIGS_BY_MEASUREMENT[stype]

      if config.measurement == 'temperature':
        measurement_stat = np.mean(values)
      else:
        measurement_stat = np.mean(np.linalg.norm(values, axis=1))
      err_msg = f"Sensor '{sensor} {config.measurement}' failed sanity checks {measurement_stat} is not between {config.sanity_min} and {config.sanity_max}"
      assert config.sanity_min <= measurement_stat <= config.sanity_max, err_msg

      std_dev = np.std(values, axis=0)
      err_msg = f"Sensor '{sensor} {config.measurement}' failed std dev test {std_dev} is not under {config.std_max}"
      assert np.all(std_dev <= config.std_max), err_msg

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
