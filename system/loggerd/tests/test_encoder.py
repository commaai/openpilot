import math
import os
import pytest
import random
import shutil
import subprocess
import time
from pathlib import Path

from parameterized import parameterized
from tqdm import trange

from openpilot.common.params import Params
from openpilot.common.timeout import Timeout
from openpilot.system.hardware import TICI
from openpilot.system.manager.process_config import managed_processes
from openpilot.tools.lib.logreader import LogReader
from openpilot.system.hardware.hw import Paths

SEGMENT_LENGTH = 2
FULL_SIZE = 2507572
def hevc_size(w): return FULL_SIZE // 2 if w <= 1344 else FULL_SIZE
CAMERAS = [
  ("fcamera.hevc", 20, hevc_size, "roadEncodeIdx"),
  ("dcamera.hevc", 20, hevc_size, "driverEncodeIdx"),
  ("ecamera.hevc", 20, hevc_size, "wideRoadEncodeIdx"),
  ("qcamera.ts", 20, lambda x: 130000, None),
]

# we check frame count, so we don't have to be too strict on size
FILE_SIZE_TOLERANCE = 0.7


@pytest.mark.tici # TODO: all of loggerd should work on PC
class TestEncoder:

  def setup_method(self):
    self._clear_logs()
    os.environ["LOGGERD_TEST"] = "1"
    os.environ["LOGGERD_SEGMENT_LENGTH"] = str(SEGMENT_LENGTH)

  def teardown_method(self):
    self._clear_logs()

  def _clear_logs(self):
    if os.path.exists(Paths.log_root()):
      shutil.rmtree(Paths.log_root())

  def _get_latest_segment_path(self):
    last_route = sorted(Path(Paths.log_root()).iterdir())[-1]
    return os.path.join(Paths.log_root(), last_route)

  # TODO: this should run faster than real time
  @parameterized.expand([(True, ), (False, )])
  def test_log_rotation(self, record_front):
    Params().put_bool("RecordFront", record_front)

    managed_processes['sensord'].start()
    managed_processes['loggerd'].start()
    managed_processes['encoderd'].start()

    time.sleep(1.0)
    managed_processes['camerad'].start()

    num_segments = int(os.getenv("SEGMENTS", random.randint(2, 8)))

    # wait for loggerd to make the dir for first segment
    route_prefix_path = None
    with Timeout(int(SEGMENT_LENGTH*3)):
      while route_prefix_path is None:
        try:
          route_prefix_path = self._get_latest_segment_path().rsplit("--", 1)[0]
        except Exception:
          time.sleep(0.1)

    def check_seg(i):
      # check each camera file size
      counts = []
      first_frames = []
      for camera, fps, size_lambda, encode_idx_name in CAMERAS:
        if not record_front and "dcamera" in camera:
          continue

        file_path = f"{route_prefix_path}--{i}/{camera}"

        # check file exists
        assert os.path.exists(file_path), f"segment #{i}: '{file_path}' missing"

        # TODO: this ffprobe call is really slow
        # get width and check frame count
        cmd = f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets,width -of csv=p=0 {file_path}"
        if TICI:
          cmd = "LD_LIBRARY_PATH=/usr/local/lib " + cmd

        expected_frames = fps * SEGMENT_LENGTH
        probe = subprocess.check_output(cmd, shell=True, encoding='utf8').split('\n')[0].strip().split(',')
        frame_width, frame_count = int(probe[0]), int(probe[1])
        counts.append(frame_count)

        assert frame_count == expected_frames, \
                         f"segment #{i}: {camera} failed frame count check: expected {expected_frames}, got {frame_count}"

        # sanity check file size
        file_size = os.path.getsize(file_path)
        target_size = size_lambda(frame_width)
        assert math.isclose(file_size, target_size, rel_tol=FILE_SIZE_TOLERANCE), \
                        f"{file_path} size {file_size} isn't close to target size {target_size}"

        # Check encodeIdx
        if encode_idx_name is not None:
          rlog_path = f"{route_prefix_path}--{i}/rlog.zst"
          msgs = [m for m in LogReader(rlog_path) if m.which() == encode_idx_name]
          encode_msgs = [getattr(m, encode_idx_name) for m in msgs]

          valid = [m.valid for m in msgs]
          segment_idxs = [m.segmentId for m in encode_msgs]
          encode_idxs = [m.encodeId for m in encode_msgs]
          frame_idxs = [m.frameId for m in encode_msgs]

          # Check frame count
          assert frame_count == len(segment_idxs)
          assert frame_count == len(encode_idxs)

          # Check for duplicates or skips
          assert 0 == segment_idxs[0]
          assert len(set(segment_idxs)) == len(segment_idxs)

          assert all(valid)

          assert expected_frames * i == encode_idxs[0]
          first_frames.append(frame_idxs[0])
          assert len(set(encode_idxs)) == len(encode_idxs)

      assert 1 == len(set(first_frames))

      if TICI:
        expected_frames = fps * SEGMENT_LENGTH
        assert min(counts) == expected_frames
      shutil.rmtree(f"{route_prefix_path}--{i}")

    try:
      for i in trange(num_segments):
        # poll for next segment
        with Timeout(int(SEGMENT_LENGTH*10), error_msg=f"timed out waiting for segment {i}"):
          while Path(f"{route_prefix_path}--{i+1}") not in Path(Paths.log_root()).iterdir():
            time.sleep(0.1)
        check_seg(i)
    finally:
      managed_processes['loggerd'].stop()
      managed_processes['encoderd'].stop()
      managed_processes['camerad'].stop()
      managed_processes['sensord'].stop()
