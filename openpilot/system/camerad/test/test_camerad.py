#!/usr/bin/env python3

import os
import time
import unittest
from unittest.mock import patch
import numpy as np

from msgq.visionipc import VisionIpcClient
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal.services import SERVICE_LIST
from openpilot.tools.lib.log_time_series import msgs_to_time_series
from openpilot.system.camerad.snapshot import VISION_STREAMS, get_snapshots
from openpilot.selfdrive.test.helpers import collect_logs, log_collector, processes_context

TEST_TIMESPAN = 10
CAMERAS = ('narrowRoadCameraState', 'cabinCameraState', 'wideRoadCameraState')
EXPOSURE_STABLE_COUNT = 3
EXPOSURE_RANGE = (0.15, 0.35)
MAX_TEST_TIME = 25
TEST_PATTERN_FRAMES = 200
TEST_PATTERN_MIN_CONFIDENCE = 10
# Full rolling-pattern cycle in frames and maximum detector position error in pixels.
TEST_PATTERN_CONFIGS = {
  'ox03c10': (41, 4),
  'os04c10': (97, 4),
}


def _numpy_rgb2gray(im):
  return np.clip(im[:,:,2] * 0.114 + im[:,:,1] * 0.587 + im[:,:,0] * 0.299, 0, 255).astype(np.uint8)

def _exposure_stats(im):
  h, w = im.shape[:2]
  gray = _numpy_rgb2gray(im[h//10:9*h//10, w//10:9*w//10])
  return float(np.median(gray) / 255.), float(np.mean(gray) / 255.)

def _in_range(median, mean):
  lo, hi = EXPOSURE_RANGE
  return lo < median < hi and lo < mean < hi

def _exposure_stable(results):
  return all(
    len(v) >= EXPOSURE_STABLE_COUNT and all(_in_range(*s) for s in v[-EXPOSURE_STABLE_COUNT:])
    for v in results.values()
  )


def _pattern_sample(client):
  buf = client.recv(1000)
  if buf is None:
    return None

  y = np.asarray(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height, :buf.width]
  profile = y[:, ::8].mean(axis=1)
  padded = np.pad(profile, (4, 4), mode='edge')
  neighbors = [padded[i:i + len(profile)] for i in range(9) if i != 4]
  residual = profile - np.median(neighbors, axis=0)
  position = int(np.argmax(residual))
  return client.frame_id, client.timestamp_sof, position, residual[position], buf.height


def _test_pattern_session():
  samples = {camera: [] for camera in CAMERAS}
  env = {'SPECTRA_TEST_PATTERN': '1', 'SPECTRA_ERROR_PROB': '-1'}
  with patch.dict(os.environ, env), processes_context(['camerad']), log_collector(CAMERAS) as (raw_logs, lock):
    clients = {camera: VisionIpcClient('camerad', VISION_STREAMS[camera], False) for camera in CAMERAS}
    for client in clients.values():
      assert client.connect(True)

    for _ in range(TEST_PATTERN_FRAMES):
      for camera, client in clients.items():
        sample = _pattern_sample(client)
        if sample is not None:
          samples[camera].append(sample)

  with lock:
    logs = msgs_to_time_series(raw_logs)
  return logs, samples


def run_and_log(procs, services, duration):
  with processes_context(procs):
    return collect_logs(services, duration)

def _camera_session():
  """Single camerad session that collects logs and exposure data.
     Runs until exposure stabilizes (min TEST_TIMESPAN seconds for enough log data)."""
  with processes_context(["camerad"]), log_collector(CAMERAS) as (raw_logs, lock):
    exposure = {cam: [] for cam in CAMERAS}
    start = time.monotonic()
    while time.monotonic() - start < MAX_TEST_TIME:
      rpic, dpic = get_snapshots(frame="narrowRoadCameraState", front_frame="cabinCameraState")
      wpic, _ = get_snapshots(frame="wideRoadCameraState")
      for cam, img in zip(CAMERAS, [rpic, dpic, wpic], strict=True):
        exposure[cam].append(_exposure_stats(img))

      if time.monotonic() - start >= TEST_TIMESPAN and _exposure_stable(exposure):
        break

    elapsed = time.monotonic() - start

  with lock:
    ts = msgs_to_time_series(raw_logs)

  for cam in CAMERAS:
    expected_frames = SERVICE_LIST[cam].frequency * elapsed
    cnt = len(ts[cam]['t'])
    assert expected_frames*0.8 < cnt < expected_frames*1.2, f"unexpected frame count {cam}: {expected_frames=}, got {cnt}"

    dts = np.abs(np.diff([ts[cam]['timestampSof']/1e6]) - 1000/SERVICE_LIST[cam].frequency)
    assert (dts < 1.0).all(), f"{cam} dts(ms) out of spec: max diff {dts.max()}, 99 percentile {np.percentile(dts, 99)}"

  return ts, exposure

class TestCamerad(OpenpilotTestCase):
  COMMA_HARDWARE_TEST = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.logs, cls.exposure_data = _camera_session()

  @parameterized.expand(CAMERAS, names=("cam",))
  def test_camera_exposure(self, cam):
    lo, hi = EXPOSURE_RANGE
    checks = self.exposure_data[cam]
    assert len(checks) >= EXPOSURE_STABLE_COUNT, f"{cam}: only got {len(checks)} samples"

    # check that exposure converges into the valid range
    passed = sum(_in_range(med, mean) for med, mean in checks)
    assert passed >= EXPOSURE_STABLE_COUNT, \
      f"{cam}: only {passed}/{len(checks)} checks in range. " + \
      " | ".join(f"#{i+1}: med={m:.4f} mean={u:.4f}" for i, (m, u) in enumerate(checks))

    # check that exposure is stable once converged (no regressions)
    in_range = False
    for i, (median, mean) in enumerate(checks):
      ok = _in_range(median, mean)
      if in_range and not ok:
        self.fail(f"{cam}: exposure regressed on sample {i+1} " +
                    f"(median={median:.4f}, mean={mean:.4f}, expected: ({lo}, {hi}))")
      in_range = ok

  def test_frame_skips(self):
    for c in CAMERAS:
      assert set(np.diff(self.logs[c]['frameId'])) == {1, }, f"{c} has frame skips"

  def test_frame_sync(self):
    SYNCED_CAMS = ('narrowRoadCameraState', 'wideRoadCameraState')
    n = range(len(self.logs['narrowRoadCameraState']['t'][:-10]))

    frame_ids = {i: [self.logs[cam]['frameId'][i] for cam in CAMERAS] for i in n}
    assert all(len(set(v)) == 1 for v in frame_ids.values()), "frame IDs not aligned"

    # road and wide cameras should be synced within 1.1ms
    synced_times = {i: [self.logs[cam]['timestampSof'][i] for cam in SYNCED_CAMS] for i in n}
    diffs = {i: (max(ts) - min(ts))/1e6 for i, ts in synced_times.items()}
    laggy_frames = {k: v for k, v in diffs.items() if v > 1.1}
    assert len(laggy_frames) == 0, f"Frames not synced properly: {laggy_frames=}"

    # cabin camera should be staggered ~25ms from road camera
    for i in n:
      offset_ms = abs(self.logs['cabinCameraState']['timestampSof'][i] - self.logs['narrowRoadCameraState']['timestampSof'][i]) / 1e6
      assert 20 < offset_ms < 30, f"cabin camera stagger out of range at frame {i}: {offset_ms:.1f}ms (expected ~25ms)"

  def test_sanity_checks(self):
    self._sanity_checks(self.logs)

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
    try:
      logs = run_and_log(["camerad", ], CAMERAS, 10)
    finally:
      del os.environ['SPECTRA_ERROR_PROB']
    ts = msgs_to_time_series(logs)

    # we should see some jumps from introduced errors
    assert np.max([ np.max(np.diff(ts[c]['frameId'])) for c in CAMERAS ]) > 1
    assert np.max([ np.max(np.diff(ts[c]['requestId'])) for c in CAMERAS ]) > 1

    self._sanity_checks(ts)


class TestCameradTestPattern(OpenpilotTestCase):
  COMMA_HARDWARE_TEST = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.logs, cls.samples = _test_pattern_session()

  def test_frame_delivery(self):
    for camera in CAMERAS:
      assert camera in self.logs
      samples = self.samples[camera]
      assert len(samples) > TEST_PATTERN_FRAMES * 0.9

      state_frame_ids = self.logs[camera]['frameId']
      state_request_ids = self.logs[camera]['requestId']
      vipc_frame_ids = np.array([sample[0] for sample in samples])
      for source, frame_ids in (('camera state', state_frame_ids), ('VisionIPC', vipc_frame_ids)):
        frame_steps = np.diff(frame_ids)
        skipped = frame_ids[1:][frame_steps != 1]
        assert len(skipped) == 0, f'{camera} {source} skipped frames before {skipped}'

      expected_sof_step = 1e9 / SERVICE_LIST[camera].frequency
      sof_step_errors = np.diff(self.logs[camera]['timestampSof']) - expected_sof_step
      assert np.all(np.abs(sof_step_errors) < 1e6), f'{camera} SOF cadence errors: {sof_step_errors[np.abs(sof_step_errors) >= 1e6]}'

      request_steps = np.diff(state_request_ids)
      skipped_requests = state_request_ids[1:][request_steps != 1]
      assert len(skipped_requests) == 0, f'{camera} skipped requests before {skipped_requests}'

      state_sofs = dict(zip(state_frame_ids, self.logs[camera]['timestampSof'], strict=True))
      matched_samples = [sample for sample in samples if sample[0] in state_sofs]
      assert len(matched_samples) > len(samples) * 0.8
      mismatched_sofs = {
        frame_id: (timestamp_sof, state_sofs[frame_id]) for frame_id, timestamp_sof, *_ in matched_samples if timestamp_sof != state_sofs[frame_id]
      }
      assert not mismatched_sofs, f'{camera} VisionIPC/camera state SOFs disagree: {mismatched_sofs}'

  def test_pattern(self):
    for camera in CAMERAS:
      sensors = set(self.logs[camera]['sensor'])
      assert len(sensors) == 1
      sensor = sensors.pop()
      assert sensor in TEST_PATTERN_CONFIGS, f'unsupported test pattern sensor: {sensor}'
      cycle_frames, position_tolerance = TEST_PATTERN_CONFIGS[sensor]

      samples = self.samples[camera]
      confident = [sample for sample in samples if sample[3] > TEST_PATTERN_MIN_CONFIDENCE]
      positions = np.array([sample[2] for sample in confident])
      assert len(confident) > len(samples) * 0.7, f'{camera} test pattern confidence too low'
      assert len(np.unique(positions)) > 20, f'{camera} test pattern is not moving'
      assert np.ptp(positions) > confident[0][4] * 0.75, f'{camera} test pattern does not span the frame'

      samples_by_frame = {sample[0]: sample for sample in confident}
      repeating_pairs = [(sample, samples_by_frame[sample[0] + cycle_frames]) for sample in confident if sample[0] + cycle_frames in samples_by_frame]
      assert len(repeating_pairs) > 20
      unexpected = [(first[0], first[2], second[2]) for first, second in repeating_pairs if abs(second[2] - first[2]) > position_tolerance]
      assert len(unexpected) < len(repeating_pairs) * 0.3, f'{camera} test pattern cycle mismatches: {unexpected}'


if __name__ == "__main__":
  unittest.main()
