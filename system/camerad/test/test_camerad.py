import os
import time
import pytest
import numpy as np

from cereal.services import SERVICE_LIST
from openpilot.tools.lib.log_time_series import msgs_to_time_series
from openpilot.system.camerad.snapshot import get_snapshots
from openpilot.selfdrive.test.helpers import collect_logs, log_collector, processes_context

TEST_TIMESPAN = 10
CAMERAS = ('roadCameraState', 'driverCameraState', 'wideRoadCameraState')
EXPOSURE_STABLE_COUNT = 3
EXPOSURE_RANGE = (0.15, 0.35)
MAX_TEST_TIME = 25

# IFE-shared cameras (wide + driver share one IFE) run at half the nominal rate
IFE_SHARED_CAMERAS = ('wideRoadCameraState', 'driverCameraState')

def _cam_freq(cam):
  freq = SERVICE_LIST[cam].frequency
  if cam in IFE_SHARED_CAMERAS:
    freq /= 2
  return freq


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


def run_and_log(procs, services, duration):
  with processes_context(procs):
    return collect_logs(services, duration)

@pytest.fixture(scope="module")
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
    freq = _cam_freq(cam)
    expected_frames = freq * elapsed
    cnt = len(ts[cam]['t'])
    assert expected_frames*0.8 < cnt < expected_frames*1.2, f"unexpected frame count {cam}: {expected_frames=:.1f} ({freq}Hz), got {cnt}"

    dts = np.abs(np.diff([ts[cam]['timestampSof']/1e6]) - 1000/freq)
    if cam in IFE_SHARED_CAMERAS:
      # IFE-shared cameras have SOF timestamp jitter from PHY switching
      good_pct = np.mean(dts < 1.0) * 100
      assert good_pct > 90, f"{cam} dts(ms): only {good_pct:.0f}% within 1ms (need >90%)"
    else:
      assert (dts < 1.0).all(), f"{cam} dts(ms) out of spec: max diff {dts.max()}, 99 percentile {np.percentile(dts, 99)}"

  return ts, exposure

@pytest.fixture(scope="module")
def logs(_camera_session):
  return _camera_session[0]

@pytest.fixture(scope="module")
def exposure_data(_camera_session):
  return _camera_session[1]

@pytest.mark.tici
class TestCamerad:
  @pytest.mark.parametrize("cam", CAMERAS)
  def test_camera_exposure(self, exposure_data, cam):
    lo, hi = EXPOSURE_RANGE
    checks = exposure_data[cam]
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
        pytest.fail(f"{cam}: exposure regressed on sample {i+1} " +
                    f"(median={median:.4f}, mean={mean:.4f}, expected: ({lo}, {hi}))")
      in_range = ok

  def test_frame_skips(self, logs):
    for c in CAMERAS:
      assert set(np.diff(logs[c]['frameId'])) == {1, }, f"{c} has frame skips"

  def test_frame_sync(self, logs):
    # IFE sharing: wide and driver share one IFE at 10fps each, road has its own IFE at 20fps.
    # Compare wide and driver directly (both at 10fps), and match road to wide by nearest timestamp.

    # Align wide and driver by common frame_id range (both at 10fps from shared IFE)
    ife_cams = ('wideRoadCameraState', 'driverCameraState')
    start_id = max(int(logs[cam]['frameId'][0]) for cam in ife_cams)
    end_id = min(int(logs[cam]['frameId'][-1]) for cam in ife_cams)
    n = end_id - start_id - 10
    assert n > 20, f"not enough overlapping IFE-shared frames: {n}"

    def get_slice(cam):
      offset = start_id - int(logs[cam]['frameId'][0])
      return slice(offset, offset + n)

    # Wide and driver frame IDs must match (both use contiguous counters from shared IFE)
    wide_ids = logs['wideRoadCameraState']['frameId'][get_slice('wideRoadCameraState')]
    driver_ids = logs['driverCameraState']['frameId'][get_slice('driverCameraState')]
    assert np.all(wide_ids == driver_ids), "wide/driver frame IDs not aligned"

    # Skip first few frames after sync where timing hasn't stabilized
    settle = 5
    wide_sofs = logs['wideRoadCameraState']['timestampSof'][get_slice('wideRoadCameraState')][settle:]
    driver_sofs = logs['driverCameraState']['timestampSof'][get_slice('driverCameraState')][settle:]
    road_sofs = logs['roadCameraState']['timestampSof']

    # Road and wide must be synced in timestamp. Road runs at 2x wide's rate,
    # so match each wide frame to the closest road frame.
    road_wide_diff = np.array([np.min(np.abs(road_sofs - ws)) / 1e6 for ws in wide_sofs])
    laggy = np.where(road_wide_diff > 1.1)[0]
    assert len(laggy) == 0, f"Road/wide not synced at indices {laggy[:10]}, max={road_wide_diff.max():.2f}ms"

    # Driver camera has FSIN stagger offset from wide. The offset magnitude varies
    # by sensor type and panda timer config (e.g., ~41ms on OX03C10, ~50ms on OS04C10).
    # Verify the offset is consistent (low jitter) rather than checking absolute value.
    driver_offset = (driver_sofs - wide_sofs) / 1e6
    median_offset = np.median(driver_offset)
    jitter = np.abs(driver_offset - median_offset)
    laggy = np.where(jitter > 5.0)[0]
    assert len(laggy) == 0, f"Driver FSIN jitter too high (median={median_offset:.1f}ms) at indices {laggy[:10]}"

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
    try:
      logs = run_and_log(["camerad", ], CAMERAS, 10)
    finally:
      del os.environ['SPECTRA_ERROR_PROB']
    ts = msgs_to_time_series(logs)

    # we should see some jumps from introduced errors
    assert np.max([ np.max(np.diff(ts[c]['frameId'])) for c in CAMERAS ]) > 1
    assert np.max([ np.max(np.diff(ts[c]['requestId'])) for c in CAMERAS ]) > 1

    self._sanity_checks(ts)
