import pytest
import time
import numpy as np
from flaky import flaky
from collections import defaultdict

import cereal.messaging as messaging
from cereal import log
from cereal.services import SERVICE_LIST
from openpilot.system.manager.process_config import managed_processes

TEST_TIMESPAN = 30
LAG_FRAME_TOLERANCE = {log.FrameData.ImageSensor.ar0231: 0.5,  # ARs use synced pulses for frame starts
                       log.FrameData.ImageSensor.ox03c10: 1.1} # OXs react to out-of-sync at next frame
FRAME_DELTA_TOLERANCE = {log.FrameData.ImageSensor.ar0231: 1.0,
                       log.FrameData.ImageSensor.ox03c10: 1.0}

CAMERAS = ('roadCameraState', 'driverCameraState', 'wideRoadCameraState')

# TODO: this shouldn't be needed
@flaky(max_runs=3)
@pytest.mark.tici
class TestCamerad:
  def setup_method(self):
    # run camerad and record logs
    managed_processes['camerad'].start()
    time.sleep(3)
    socks = {c: messaging.sub_sock(c, conflate=False, timeout=100) for c in CAMERAS}

    self.logs = defaultdict(list)
    start_time = time.monotonic()
    while time.monotonic()- start_time < TEST_TIMESPAN:
      for cam, s in socks.items():
        self.logs[cam] += messaging.drain_sock(s)
      time.sleep(0.2)
    managed_processes['camerad'].stop()

    self.log_by_frame_id = defaultdict(list)
    self.sensor_type = None
    for cam, msgs in self.logs.items():
      if self.sensor_type is None:
        self.sensor_type = getattr(msgs[0], msgs[0].which()).sensor.raw
      expected_frames = SERVICE_LIST[cam].frequency * TEST_TIMESPAN
      assert expected_frames*0.95 < len(msgs) < expected_frames*1.05, f"unexpected frame count {cam}: {expected_frames=}, got {len(msgs)}"

      dts = np.abs(np.diff([getattr(m, m.which()).timestampSof/1e6 for m in msgs]) - 1000/SERVICE_LIST[cam].frequency)
      assert (dts < FRAME_DELTA_TOLERANCE[self.sensor_type]).all(), f"{cam} dts(ms) out of spec: max diff {dts.max()}, 99 percentile {np.percentile(dts, 99)}"

      for m in msgs:
        self.log_by_frame_id[getattr(m, m.which()).frameId].append(m)

    # strip beginning and end
    for _ in range(3):
      mn, mx = min(self.log_by_frame_id.keys()), max(self.log_by_frame_id.keys())
      del self.log_by_frame_id[mn]
      del self.log_by_frame_id[mx]

  def test_frame_skips(self):
    skips = {}
    frame_ids = self.log_by_frame_id.keys()
    for frame_id in range(min(frame_ids), max(frame_ids)):
      seen_cams = [msg.which() for msg in self.log_by_frame_id[frame_id]]
      skip_cams = set(CAMERAS) - set(seen_cams)
      if len(skip_cams):
        skips[frame_id] = skip_cams
    assert len(skips) == 0, f"Found frame skips, missing cameras for the following frames: {skips}"

  def test_frame_sync(self):
    frame_times = {frame_id: [getattr(m, m.which()).timestampSof for m in msgs] for frame_id, msgs in self.log_by_frame_id.items()}
    diffs = {frame_id: (max(ts) - min(ts))/1e6 for frame_id, ts in frame_times.items()}

    def get_desc(fid, diff):
      cam_times = [(m.which(), getattr(m, m.which()).timestampSof/1e6) for m in self.log_by_frame_id[fid]]
      return (diff, cam_times)
    laggy_frames = {k: get_desc(k, v) for k, v in diffs.items() if v > LAG_FRAME_TOLERANCE[self.sensor_type]}

    def in_tol(diff):
      return 50 - LAG_FRAME_TOLERANCE[self.sensor_type] < diff and diff < 50 + LAG_FRAME_TOLERANCE[self.sensor_type]
    if len(laggy_frames) != 0 and all( in_tol(laggy_frames[lf][0]) for lf in laggy_frames):
      print("TODO: handle camera out of sync")
    else:
      assert len(laggy_frames) == 0, f"Frames not synced properly: {laggy_frames=}"
