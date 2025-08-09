import os
import time
import pytest
import numpy as np

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.system.manager.process_config import managed_processes
from openpilot.tools.lib.log_time_series import msgs_to_time_series

TEST_TIMESPAN = 10
CAMERAS = ('roadCameraState', 'driverCameraState', 'wideRoadCameraState')


def run_and_log(procs, services, duration):
  logs = []

  try:
    for p in procs:
      managed_processes[p].start()
    socks = [messaging.sub_sock(s, conflate=False, timeout=100) for s in services]

    start_time = time.monotonic()
    while time.monotonic() - start_time < duration:
      for s in socks:
        logs.extend(messaging.drain_sock(s))
    for p in procs:
      assert managed_processes[p].proc.is_alive()
  finally:
    for p in procs:
      managed_processes[p].stop()

  return logs

@pytest.fixture(scope="module")
def logs():
  logs = run_and_log(["camerad", ], CAMERAS, TEST_TIMESPAN)
  ts = msgs_to_time_series(logs)

  for cam in CAMERAS:
    expected_frames = SERVICE_LIST[cam].frequency * TEST_TIMESPAN
    cnt = len(ts[cam]['t'])
    assert expected_frames*0.8 < cnt < expected_frames*1.2, f"unexpected frame count {cam}: {expected_frames=}, got {cnt}"

    dts = np.abs(np.diff([ts[cam]['timestampSof']/1e6]) - 1000/SERVICE_LIST[cam].frequency)
    assert (dts < 1.0).all(), f"{cam} dts(ms) out of spec: max diff {dts.max()}, 99 percentile {np.percentile(dts, 99)}"
  return ts

@pytest.mark.tici
class TestCamerad:
  def test_frame_skips(self, logs):
    for c in CAMERAS:
      assert set(np.diff(logs[c]['frameId'])) == {1, }, f"{c} has frame skips"

  def test_frame_sync(self, logs):
    n = range(len(logs['roadCameraState']['t'][:-10]))

    frame_ids = {i: [logs[cam]['frameId'][i] for cam in CAMERAS] for i in n}
    assert all(len(set(v)) == 1 for v in frame_ids.values()), "frame IDs not aligned"

    frame_times = {i: [logs[cam]['timestampSof'][i] for cam in CAMERAS] for i in n}
    diffs = {i: (max(ts) - min(ts))/1e6 for i, ts in frame_times.items()}

    laggy_frames = {k: v for k, v in diffs.items() if v > 1.1}
    assert len(laggy_frames) == 0, f"Frames not synced properly: {laggy_frames=}"

  def test_sanity_checks(self, logs):
    self._sanity_checks(logs)

  def _sanity_checks(self, ts):
    for c in CAMERAS:
      assert c in ts
      assert len(ts[c]['t']) > 20

      # not a valid request id
      assert 0 not in ts[c]['requestId']

      # should monotonically increase
      assert np.all(np.diff(ts[c]['frameId']) >= 1)
      assert np.all(np.diff(ts[c]['requestId']) >= 1)

      # EOF > SOF
      assert np.all((ts[c]['timestampEof'] - ts[c]['timestampSof']) > 0)

      # logMonoTime > SOF
      assert np.all((ts[c]['t'] - ts[c]['timestampSof']/1e9) > 1e-7)

      # logMonoTime > EOF, needs some tolerance since EOF is (SOF + readout time) but there is noise in the SOF timestamping (done via IRQ)
      assert np.mean((ts[c]['t'] - ts[c]['timestampEof']/1e9) > 1e-7) > 0.7  # should be mostly logMonoTime > EOF
      assert np.all((ts[c]['t'] - ts[c]['timestampEof']/1e9) > -0.10)        # when EOF > logMonoTime, it should never be more than two frames

  def test_stress_test(self):
    os.environ['SPECTRA_ERROR_PROB'] = '0.008'
    logs = run_and_log(["camerad", ], CAMERAS, 10)
    ts = msgs_to_time_series(logs)

    # we should see some jumps from introduced errors
    assert np.max([ np.max(np.diff(ts[c]['frameId'])) for c in CAMERAS ]) > 1
    assert np.max([ np.max(np.diff(ts[c]['requestId'])) for c in CAMERAS ]) > 1

    self._sanity_checks(ts)
