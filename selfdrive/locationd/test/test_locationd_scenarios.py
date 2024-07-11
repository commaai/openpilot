import pytest
import numpy as np
from collections import defaultdict
from enum import Enum

from openpilot.tools.lib.logreader import LogReader
from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.selfdrive.test.process_replay.process_replay import replay_process_with_name

TEST_ROUTE = "ff2bd20623fcaeaa|2023-09-05--10-14-54/4"
GPS_MESSAGES = ['gpsLocationExternal', 'gpsLocation']
SELECT_COMPARE_FIELDS = {
  'yaw_rate': ['angularVelocityCalibrated', 'value', 2],
  'roll': ['orientationNED', 'value', 0],
  'gps_flag': ['gpsOK'],
  'inputs_flag': ['inputsOK'],
  'sensors_flag': ['sensorsOK'],
}
JUNK_IDX = 100


class Scenario(Enum):
  BASE = 'base'
  GPS_OFF = 'gps_off'
  GPS_OFF_MIDWAY = 'gps_off_midway'
  GPS_ON_MIDWAY = 'gps_on_midway'
  GPS_TUNNEL = 'gps_tunnel'
  GYRO_OFF = 'gyro_off'
  GYRO_SPIKE_MIDWAY = 'gyro_spike_midway'
  ACCEL_OFF = 'accel_off'
  ACCEL_SPIKE_MIDWAY = 'accel_spike_midway'


def get_select_fields_data(logs):
  def get_nested_keys(msg, keys):
    val = None
    for key in keys:
      val = getattr(msg if val is None else val, key) if isinstance(key, str) else val[key]
    return val
  llk = [x.liveLocationKalman for x in logs if x.which() == 'liveLocationKalman']
  data = defaultdict(list)
  for msg in llk:
    for key, fields in SELECT_COMPARE_FIELDS.items():
      data[key].append(get_nested_keys(msg, fields))
  for key in data:
    data[key] = np.array(data[key][JUNK_IDX:], dtype=float)
  return data


def run_scenarios(scenario, logs):
  if scenario == Scenario.BASE:
    pass

  elif scenario == Scenario.GPS_OFF:
    logs = sorted([x for x in logs if x.which() not in GPS_MESSAGES], key=lambda x: x.logMonoTime)

  elif scenario == Scenario.GPS_OFF_MIDWAY:
    non_gps = [x for x in logs if x.which() not in GPS_MESSAGES]
    gps = [x for x in logs if x.which() in GPS_MESSAGES]
    logs = sorted(non_gps + gps[: len(gps) // 2], key=lambda x: x.logMonoTime)

  elif scenario == Scenario.GPS_ON_MIDWAY:
    non_gps = [x for x in logs if x.which() not in GPS_MESSAGES]
    gps = [x for x in logs if x.which() in GPS_MESSAGES]
    logs = sorted(non_gps + gps[len(gps) // 2:], key=lambda x: x.logMonoTime)

  elif scenario == Scenario.GPS_TUNNEL:
    non_gps = [x for x in logs if x.which() not in GPS_MESSAGES]
    gps = [x for x in logs if x.which() in GPS_MESSAGES]
    logs = sorted(non_gps + gps[:len(gps) // 4] + gps[-len(gps) // 4:], key=lambda x: x.logMonoTime)

  elif scenario == Scenario.GYRO_OFF:
    logs = sorted([x for x in logs if x.which() != 'gyroscope'], key=lambda x: x.logMonoTime)

  elif scenario == Scenario.GYRO_SPIKE_MIDWAY:
    non_gyro = [x for x in logs if x.which() not in 'gyroscope']
    gyro = [x for x in logs if x.which() in 'gyroscope']
    temp = gyro[len(gyro) // 2].as_builder()
    temp.gyroscope.gyroUncalibrated.v[0] += 3.0
    gyro[len(gyro) // 2] = temp.as_reader()
    logs = sorted(non_gyro + gyro, key=lambda x: x.logMonoTime)

  elif scenario == Scenario.ACCEL_OFF:
    logs = sorted([x for x in logs if x.which() != 'accelerometer'], key=lambda x: x.logMonoTime)

  elif scenario == Scenario.ACCEL_SPIKE_MIDWAY:
    non_accel = [x for x in logs if x.which() not in 'accelerometer']
    accel = [x for x in logs if x.which() in 'accelerometer']
    temp = accel[len(accel) // 2].as_builder()
    temp.accelerometer.acceleration.v[0] += 10.0
    accel[len(accel) // 2] = temp.as_reader()
    logs = sorted(non_accel + accel, key=lambda x: x.logMonoTime)

  replayed_logs = replay_process_with_name(name='locationd', lr=logs)
  return get_select_fields_data(logs), get_select_fields_data(replayed_logs)


@pytest.mark.xdist_group("test_locationd_scenarios")
@pytest.mark.shared_download_cache
class TestLocationdScenarios:
  """
  Test locationd with different scenarios. In all these scenarios, we expect the following:
    - locationd kalman filter should never go unstable (we care mostly about yaw_rate, roll, gpsOK, inputsOK, sensorsOK)
    - faulty values should be ignored, with appropriate flags set
  """

  @classmethod
  def setup_class(cls):
    cls.logs = migrate_all(LogReader(TEST_ROUTE))

  def test_base(self):
    """
    Test: unchanged log
    Expected Result:
      - yaw_rate: unchanged
      - roll: unchanged
    """
    orig_data, replayed_data = run_scenarios(Scenario.BASE, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.2))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.5))

  def test_gps_off(self):
    """
    Test: no GPS message for the entire segment
    Expected Result:
      - yaw_rate: unchanged
      - roll:
      - gpsOK: False
    """
    orig_data, replayed_data = run_scenarios(Scenario.GPS_OFF, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.2))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.5))
    assert np.all(replayed_data['gps_flag'] == 0.0)

  def test_gps_off_midway(self):
    """
    Test: no GPS message for the second half of the segment
    Expected Result:
      - yaw_rate: unchanged
      - roll:
      - gpsOK: True for the first half, False for the second half
    """
    orig_data, replayed_data = run_scenarios(Scenario.GPS_OFF_MIDWAY, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.2))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.5))
    assert np.diff(replayed_data['gps_flag'])[512] == -1.0

  def test_gps_on_midway(self):
    """
    Test: no GPS message for the first half of the segment
    Expected Result:
      - yaw_rate: unchanged
      - roll:
      - gpsOK: False for the first half, True for the second half
    """
    orig_data, replayed_data = run_scenarios(Scenario.GPS_ON_MIDWAY, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.2))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(1.5))
    assert np.diff(replayed_data['gps_flag'])[505] == 1.0

  def test_gps_tunnel(self):
    """
    Test: no GPS message for the middle section of the segment
    Expected Result:
      - yaw_rate: unchanged
      - roll:
      - gpsOK: False for the middle section, True for the rest
    """
    orig_data, replayed_data = run_scenarios(Scenario.GPS_TUNNEL, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.2))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.5))
    assert np.diff(replayed_data['gps_flag'])[213] == -1.0
    assert np.diff(replayed_data['gps_flag'])[805] == 1.0

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

  def test_gyro_spikes(self):
    """
    Test: a gyroscope spike in the middle of the segment
    Expected Result:
      - yaw_rate: unchanged
      - roll: unchanged
      - inputsOK: False for some time after the spike, True for the rest
    """
    orig_data, replayed_data = run_scenarios(Scenario.GYRO_SPIKE_MIDWAY, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.2))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.5))
    assert np.diff(replayed_data['inputs_flag'])[500] == -1.0
    assert np.diff(replayed_data['inputs_flag'])[694] == 1.0

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

  def test_accel_spikes(self):
    """
    ToDo:
    Test: an accelerometer spike in the middle of the segment
    Expected Result: Right now, the kalman filter is not robust to small spikes like it is to gyroscope spikes.
    """
    orig_data, replayed_data = run_scenarios(Scenario.ACCEL_SPIKE_MIDWAY, self.logs)
    assert np.allclose(orig_data['yaw_rate'], replayed_data['yaw_rate'], atol=np.radians(0.2))
    assert np.allclose(orig_data['roll'], replayed_data['roll'], atol=np.radians(0.5))
