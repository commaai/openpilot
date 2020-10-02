#!/usr/bin/env python3
import math
import os
import random
import shutil
import subprocess
import time
import unittest
from parameterized import parameterized
from pathlib import Path
from tqdm import trange

from common.params import Params
from common.hardware import EON, TICI
from common.timeout import Timeout
from selfdrive.test.helpers import with_processes
from selfdrive.loggerd.config import ROOT, CAMERA_FPS


# baseline file sizes for a 2s segment, in bytes
FULL_SIZE = 1253786
if EON:
  CAMERAS = {
    "fcamera": FULL_SIZE,
    "dcamera": 770920,
    "qcamera": 38533,
  }
elif TICI:
  CAMERAS = {f"{c}camera": FULL_SIZE for c in ["f", "e", "d"]}
else:
  CAMERAS = {}

ALL_CAMERA_COMBINATIONS = [(cameras,) for cameras in [CAMERAS, {k:CAMERAS[k] for k in CAMERAS if k!='dcamera'}]]

FRAME_TOLERANCE = 2
FILE_SIZE_TOLERANCE = 0.5

class TestLoggerd(unittest.TestCase):

  # TODO: all of loggerd should work on PC
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

  def setUp(self):
    self._clear_logs()

    self.segment_length = 2
    os.environ["LOGGERD_TEST"] = "1"
    os.environ["LOGGERD_SEGMENT_LENGTH"] = str(self.segment_length)

  def tearDown(self):
    self._clear_logs()

  def _clear_logs(self):
    if os.path.exists(ROOT):
      shutil.rmtree(ROOT)

  def _get_latest_segment_path(self):
    last_route = sorted(Path(ROOT).iterdir(), key=os.path.getmtime)[-1]
    return os.path.join(ROOT, last_route)

  # TODO: this should run faster than real time
  @parameterized.expand(ALL_CAMERA_COMBINATIONS)
  @with_processes(['camerad', 'loggerd'], init_time=5)
  def test_log_rotation(self, cameras):
    print("checking targets:", cameras)
    Params().put("RecordFront", "1" if 'dcamera' in cameras else "0")

    num_segments = random.randint(80, 150)
    if "CI" in os.environ:
      num_segments = random.randint(15, 20) # ffprobe is slow on comma two

    # wait for loggerd to make the dir for first segment
    time.sleep(10)
    route_prefix_path = None
    with Timeout(30):
      while route_prefix_path is None:
        try:
          route_prefix_path = self._get_latest_segment_path().rsplit("--", 1)[0]
        except Exception:
          time.sleep(2)
          continue

    for i in trange(num_segments):
      # poll for next segment
      if i < num_segments - 1:
        with Timeout(self.segment_length*3, error_msg=f"timed out waiting for segment {i}"):
          while True:
            seg_num = int(self._get_latest_segment_path().rsplit("--", 1)[1])
            if seg_num > i:
              break
            time.sleep(0.1)
      else:
        time.sleep(self.segment_length)

      # check each camera file size
      for camera, size in cameras.items():
        ext = "ts" if camera=='qcamera' else "hevc"
        file_path = f"{route_prefix_path}--{i}/{camera}.{ext}"

        # check file size
        self.assertTrue(os.path.exists(file_path), f"couldn't find {file_path}")
        file_size = os.path.getsize(file_path)
        self.assertTrue(math.isclose(file_size, size, rel_tol=FILE_SIZE_TOLERANCE),
                        f"{camera} failed size check: expected {size}, got {file_size}")

        if camera == 'qcamera':
          continue

        # check frame count
        cmd = f"ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames \
               -of default=nokey=1:noprint_wrappers=1 {file_path}"
        expected_frames = self.segment_length * CAMERA_FPS // 2 if (EON and camera=='dcamera') else self.segment_length * CAMERA_FPS
        frame_count = int(subprocess.check_output(cmd, shell=True, encoding='utf8').strip())
        self.assertTrue(abs(expected_frames - frame_count) <= FRAME_TOLERANCE,
                        f"{camera} failed frame count check: expected {expected_frames}, got {frame_count}")
      shutil.rmtree(f"{route_prefix_path}--{i}")

if __name__ == "__main__":
  unittest.main()
