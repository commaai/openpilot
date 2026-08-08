#!/usr/bin/env python3

import os
import time
import unittest
from unittest.mock import patch
import numpy as np

from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal.services import SERVICE_LIST
from openpilot.tools.lib.log_time_series import msgs_to_time_series
from openpilot.system.camerad.snapshot import get_snapshots
from openpilot.selfdrive.test.helpers import collect_logs, log_collector, processes_context

TEST_TIMESPAN = 10
CAMERAS = ('roadCameraState', 'driverCameraState', 'wideRoadCameraState')
EXPOSURE_STABLE_COUNT = 3
EXPOSURE_RANGE = (0.15, 0.35)
MAX_TEST_TIME = 25
STRESS_ERRORS = (
  ("skip_sof", "skipping SOF event"),
  ("processing_delay", "sync sleep time"),
  ("ife_timeout", "IFE sync"),
  ("bps_timeout", "BPS sync"),
)


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
  y = np.asarray(buf.data[:buf.uv_offset], dtype=np.uint8).reshape((-1, buf.stride))[:buf.height:4, :buf.width:4]
  profile = y.mean(1)
  padded = np.pad(profile, (4, 4), mode="edge")
  neighbors = [padded[i:i + len(profile)] for i in range(9) if i != 4]
  residual = profile - np.median(neighbors, axis=0)
  position = int(np.argmax(residual))
  if residual[position] <= 10:
    return None
  return client.timestamp_sof, client.frame_id, position / len(profile)

def _sanity_checks(ts):
  for camera in CAMERAS:
    assert camera in ts
    assert len(ts[camera]['t']) > 20
    assert 0 not in ts[camera]['requestId']

    frame_steps = np.diff(ts[camera]['frameId'])
    assert np.all(frame_steps > 0)
    assert np.all(np.diff(ts[camera]['requestId']) > 0)
    # Skipped frame IDs must account for the same number of frame periods in SOF time.
    expected_sof_steps = frame_steps * 1e9 / SERVICE_LIST[camera].frequency
    sof_step_errors = np.diff(ts[camera]['timestampSof']) - expected_sof_steps
    assert np.all(np.abs(sof_step_errors) < 2e6), f"{camera} frame/SOF steps disagree: {sof_step_errors[np.abs(sof_step_errors) >= 2e6]}"

    assert np.all((ts[camera]['timestampEof'] - ts[camera]['timestampSof']) > 0)
    assert np.all((ts[camera]['t'] - ts[camera]['timestampSof']/1e9) > 1e-7)
    assert np.mean((ts[camera]['t'] - ts[camera]['timestampEof']/1e9) > 1e-7) > 0.7
    assert np.all((ts[camera]['t'] - ts[camera]['timestampEof']/1e9) > -0.10)


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
      rpic, dpic = get_snapshots(frame="roadCameraState", front_frame="driverCameraState")
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
    SYNCED_CAMS = ('roadCameraState', 'wideRoadCameraState')
    n = range(len(self.logs['roadCameraState']['t'][:-10]))

    frame_ids = {i: [self.logs[cam]['frameId'][i] for cam in CAMERAS] for i in n}
    assert all(len(set(v)) == 1 for v in frame_ids.values()), "frame IDs not aligned"

    # road and wide cameras should be synced within 1.1ms
    synced_times = {i: [self.logs[cam]['timestampSof'][i] for cam in SYNCED_CAMS] for i in n}
    diffs = {i: (max(ts) - min(ts))/1e6 for i, ts in synced_times.items()}
    laggy_frames = {k: v for k, v in diffs.items() if v > 1.1}
    assert len(laggy_frames) == 0, f"Frames not synced properly: {laggy_frames=}"

    # driver camera should be staggered ~25ms from road camera
    for i in n:
      offset_ms = abs(self.logs['driverCameraState']['timestampSof'][i] - self.logs['roadCameraState']['timestampSof'][i]) / 1e6
      assert 20 < offset_ms < 30, f"driver camera stagger out of range at frame {i}: {offset_ms:.1f}ms (expected ~25ms)"

  def test_sanity_checks(self):
    _sanity_checks(self.logs)


class TestCameradStress(OpenpilotTestCase):
  TICI_TEST = True

  @parameterized.expand(STRESS_ERRORS, ids=lambda name, _: name)
  def test_stress_test(self, _, error):
    env = {'SPECTRA_ERROR_FILTER': error, 'SPECTRA_ERROR_PROB': '1', 'SPECTRA_ERROR_DT': '2000'}
    with patch.dict(os.environ, env):
      logs = run_and_log(["camerad"], CAMERAS, 6)
    ts = msgs_to_time_series(logs)

    assert max(np.max(np.diff(ts[c]['frameId'])) for c in CAMERAS) > 1
    assert max(np.max(np.diff(ts[c]['requestId'])) for c in CAMERAS) > 1
    _sanity_checks(ts)

  def test_frame_data_alignment(self):
    env = {'SPECTRA_TEST_PATTERN': '1', 'SPECTRA_ERROR_FILTER': 'publish delay', 'SPECTRA_ERROR_CAMERA': '1',
           'SPECTRA_ERROR_PROB': '1', 'SPECTRA_ERROR_DT': '6000'}
    samples = {'road': [], 'wide': []}
    with patch.dict(os.environ, env), processes_context(["camerad"]), log_collector(CAMERAS) as (raw_logs, lock):
      clients = {
        'road': VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True),
        'wide': VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True),
      }
      for client in clients.values():
        client.connect(True)
      end = time.monotonic() + 10
      while time.monotonic() < end:
        for camera, client in clients.items():
          sample = _pattern_sample(client)
          if sample is not None:
            samples[camera].append(sample)

    with lock:
      ts = msgs_to_time_series(raw_logs)
    assert all(samples.values())
    offsets = []
    for timestamp, frame_id, phase in samples['road']:
      wide = min(samples['wide'], key=lambda sample: abs(sample[0] - timestamp))
      if abs(wide[0] - timestamp) < 1.1e6:
        offsets.append((frame_id, (phase - wide[2]) % 1))
    assert len(offsets) > 20
    expected_offsets = {offset for _, offset in offsets[:len(offsets) // 2]}
    # Allow small sensor-specific phase jitter while rejecting a substituted frame.
    tolerance = 1 / 32
    unexpected = {frame_id: offset for frame_id, offset in offsets[len(offsets) // 2:]
                  if all(abs((offset - expected + .5) % 1 - .5) > tolerance for expected in expected_offsets)}
    assert not unexpected, f"road/wide pixels disagree for synchronized frames: {unexpected}"

    assert max(np.max(np.diff(ts[c]['frameId'])) for c in CAMERAS) > 1
    _sanity_checks(ts)


if __name__ == "__main__":
  unittest.main()
