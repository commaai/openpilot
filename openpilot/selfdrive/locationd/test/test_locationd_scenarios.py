import fcntl
import numpy as np
import os
import tempfile
from collections import defaultdict
from enum import Enum

from openpilot.common.test import OpenpilotTestCase
from openpilot.tools.lib.logreader import LogReader, save_log
from openpilot.selfdrive.locationd.lagd import masked_symmetric_moving_average
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.selfdrive.test.process_replay.process_replay import replay_process_with_name

# TODO find a new segment to test
TEST_ROUTE = "4019fff6e54cf1c7|00000123--4bc0d95ef6/5"
GPS_MESSAGES = ['gpsLocationExternal', 'gpsLocation']
SELECT_COMPARE_FIELDS = {
  'yaw_rate': ['angularVelocityDevice', 'z'],
  'roll': ['orientationNED', 'x'],
  'inputs_flag': ['inputsOK'],
  'sensors_flag': ['sensorsOK'],
}
SMOOTH_FIELDS = ['yaw_rate', 'roll']
JUNK_IDX = 100
CONSISTENT_SPIKES_COUNT = 10
SCENARIO_DURATION_SEC = 25


class Scenario(Enum):
  BASE = 'base'
  GYRO_OFF = 'gyro_off'
  GYRO_SPIKE_MIDWAY = 'gyro_spike_midway'
  GYRO_CONSISTENT_SPIKES = 'gyro_consistent_spikes'
  ACCEL_OFF = 'accel_off'
  ACCEL_SPIKE_MIDWAY = 'accel_spike_midway'
  ACCEL_CONSISTENT_SPIKES = 'accel_consistent_spikes'
  SENSOR_TIMING_SPIKE_MIDWAY = 'timing_spikes'
  SENSOR_TIMING_CONSISTENT_SPIKES = 'timing_consistent_spikes'


def get_select_fields_data(logs):
  def sig_smooth(signal):
    return masked_symmetric_moving_average(signal, np.ones_like(signal), 5, 1.0)
  def get_nested_keys(msg, keys):
    val = msg
    for key in keys:
      val = getattr(val, key) if isinstance(key, str) else val[key]
    return val
  lp = [x.livePose for x in logs if x.which() == 'livePose']
  data = defaultdict(list)
  for msg in lp:
    for key, fields in SELECT_COMPARE_FIELDS.items():
      data[key].append(get_nested_keys(msg, fields))
  for key in data:
    data[key] = np.array(data[key][JUNK_IDX:], dtype=float)
    if key in SMOOTH_FIELDS:
      data[key] = sig_smooth(data[key])
  return data


def modify_logs_midway(logs, which, count, fn):
  non_which = [x for x in logs if x.which() != which]
  which = [x for x in logs if x.which() == which]
  temps = which[len(which) // 2:len(which) // 2 + count]
  for i, temp in enumerate(temps):
    temp = temp.as_builder()
    fn(temp)
    which[len(which) // 2 + i] = temp.as_reader()
  return sorted(non_which + which, key=lambda x: x.logMonoTime)


def assert_inputs_flag_cycle(data):
  transitions = np.diff(data['inputs_flag'])
  falling = np.where(transitions == -1.0)[0]
  rising = np.where(transitions == 1.0)[0]
  assert len(falling) == 1
  assert len(rising) == 1
  assert falling[0] < rising[0]


def run_scenarios(scenario, logs):
  if scenario == Scenario.BASE:
    pass

  elif scenario == Scenario.GYRO_OFF:
    logs = sorted([x for x in logs if x.which() != 'gyroscope'], key=lambda x: x.logMonoTime)

  elif scenario == Scenario.GYRO_SPIKE_MIDWAY or scenario == Scenario.GYRO_CONSISTENT_SPIKES:
    def gyro_spike(msg):
      msg.gyroscope.gyroUncalibrated.v[0] += 3.0
    count = 1 if scenario == Scenario.GYRO_SPIKE_MIDWAY else CONSISTENT_SPIKES_COUNT
    logs = modify_logs_midway(logs, 'gyroscope', count, gyro_spike)

  elif scenario == Scenario.ACCEL_OFF:
    logs = sorted([x for x in logs if x.which() != 'accelerometer'], key=lambda x: x.logMonoTime)

  elif scenario == Scenario.ACCEL_SPIKE_MIDWAY or scenario == Scenario.ACCEL_CONSISTENT_SPIKES:
    def acc_spike(msg):
      msg.accelerometer.acceleration.v[0] += 100.0
    count = 1 if scenario == Scenario.ACCEL_SPIKE_MIDWAY else CONSISTENT_SPIKES_COUNT
    logs = modify_logs_midway(logs, 'accelerometer', count, acc_spike)

  elif scenario == Scenario.SENSOR_TIMING_SPIKE_MIDWAY or scenario == Scenario.SENSOR_TIMING_CONSISTENT_SPIKES:
    def timing_spike(msg):
      msg.accelerometer.timestamp -= int(0.150 * 1e9)
    count = 1 if scenario == Scenario.SENSOR_TIMING_SPIKE_MIDWAY else CONSISTENT_SPIKES_COUNT
    logs = modify_logs_midway(logs, 'accelerometer', count, timing_spike)

  replayed_logs = replay_process_with_name(name='locationd', lr=logs)
  orig_data, replayed_data = get_select_fields_data(logs), get_select_fields_data(replayed_logs)
  common_length = min(len(orig_data['yaw_rate']), len(replayed_data['yaw_rate']))
  for data in (orig_data, replayed_data):
    for key in data:
      data[key] = data[key][:common_length]
  return orig_data, replayed_data


class TestLocationdScenarios(OpenpilotTestCase):
  """
  Test locationd with different scenarios. In all these scenarios, we expect the following:
    - locationd kalman filter should never go unstable (we care mostly about yaw_rate, roll, gpsOK, inputsOK, sensorsOK)
    - faulty values should be ignored, with appropriate flags set
  """

  @classmethod
  def setup_class(cls):
    # Scenario tests run in separate workers. Migrate the remote route once,
    # then let every worker read the same uncompressed per-run fixture.
    cache_root = os.environ.get("OPENPILOT_TEST_CACHE", tempfile.gettempdir())
    cache_path = os.path.join(cache_root, "locationd-scenarios")
    lock_path = f"{cache_path}.lock"
    logs = None
    with open(lock_path, "w") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      if not os.path.exists(cache_path):
        logs = migrate_all(list(LogReader(TEST_ROUTE)))
        sensor_start = min(msg.logMonoTime for msg in logs if msg.which() == 'accelerometer')
        logs = [msg for msg in logs if msg.logMonoTime <= sensor_start + int(SCENARIO_DURATION_SEC * 1e9)]
        save_log(cache_path, logs, compress=False)
    if logs is None:
      logs = list(LogReader(cache_path))
    cls.logs = logs

  def test_base(self):
    """
    Test: unchanged log
    Expected Result:
      - yaw_rate: unchanged
      - roll: unchanged
    """
    orig_data, replayed_data = run_scenarios(Scenario.BASE, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.35))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.35))

  def test_gyro_off(self):
    """
    Test: no gyroscope message for the entire segment
    Expected Result:
      - yaw_rate: 0
      - roll: 0
      - sensorsOK: False
    """
    _, replayed_data = run_scenarios(Scenario.GYRO_OFF, self.logs)
    assert np.allclose(replayed_data['yaw_rate'], 0.0)
    assert np.allclose(replayed_data['roll'], 0.0)
    assert np.all(replayed_data['sensors_flag'] == 0.0)

  def test_gyro_spike(self):
    """
    Test: a gyroscope spike in the middle of the segment
    Expected Result:
      - yaw_rate: unchanged
      - roll: unchanged
      - inputsOK: False for some time after the spike, True for the rest
    """
    orig_data, replayed_data = run_scenarios(Scenario.GYRO_SPIKE_MIDWAY, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.35))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.35))
    assert np.all(replayed_data['inputs_flag'] == orig_data['inputs_flag'])
    assert np.all(replayed_data['sensors_flag'] == orig_data['sensors_flag'])

  def test_consistent_gyro_spikes(self):
    """
    Test: consistent timing spikes for N gyroscope messages in the middle of the segment
    Expected Result: inputsOK becomes False after N of bad measurements
    """
    _, replayed_data = run_scenarios(Scenario.GYRO_CONSISTENT_SPIKES, self.logs)
    assert_inputs_flag_cycle(replayed_data)

  def test_accel_off(self):
    """
    Test: no accelerometer message for the entire segment
    Expected Result:
      - yaw_rate: 0
      - roll: 0
      - sensorsOK: False
    """
    _, replayed_data = run_scenarios(Scenario.ACCEL_OFF, self.logs)
    assert np.allclose(replayed_data['yaw_rate'], 0.0)
    assert np.allclose(replayed_data['roll'], 0.0)
    assert np.all(replayed_data['sensors_flag'] == 0.0)

  def test_accel_spike(self):
    """
    ToDo:
    Test: an accelerometer spike in the middle of the segment
    Expected Result: Right now, the kalman filter is not robust to small spikes like it is to gyroscope spikes.
    """
    orig_data, replayed_data = run_scenarios(Scenario.ACCEL_SPIKE_MIDWAY, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.35))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.35))

  def test_single_timing_spike(self):
    """
    Test: timing of 150ms off for the single accelerometer message in the middle of the segment
    Expected Result: the message is ignored, and inputsOK is False for that time
    """
    orig_data, replayed_data = run_scenarios(Scenario.SENSOR_TIMING_SPIKE_MIDWAY, self.logs)
    assert np.all(replayed_data['inputs_flag'] == orig_data['inputs_flag'])
    assert np.all(replayed_data['sensors_flag'] == orig_data['sensors_flag'])

  def test_consistent_timing_spikes(self):
    """
    Test: consistent timing spikes for N accelerometer messages in the middle of the segment
    Expected Result: inputsOK becomes False after N of bad measurements
    """
    _, replayed_data = run_scenarios(Scenario.SENSOR_TIMING_CONSISTENT_SPIKES, self.logs)
    assert_inputs_flag_cycle(replayed_data)
