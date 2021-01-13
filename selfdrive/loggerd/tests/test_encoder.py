#!/usr/bin/env python3
import math
import os
import random
import shutil
import subprocess
import threading
import time
import unittest
from parameterized import parameterized
from pathlib import Path
from tqdm import trange

from common.params import Params
from common.timeout import Timeout
from selfdrive.hardware import EON, TICI
from selfdrive.test.helpers import with_processes
from selfdrive.loggerd.config import ROOT, CAMERA_FPS


# baseline file sizes for a 2s segment, in bytes
SEGMENT_LENGTH = 2
FULL_SIZE = 1253786
if EON:
  CAMERAS = {
    "fcamera": FULL_SIZE,
    "dcamera": 770920,
    "qcamera": 38533,
  }
else:
  CAMERAS = {f"{c}camera": FULL_SIZE if c!="q" else 38533 for c in ["f", "e", "d", "q"]}

ALL_CAMERA_COMBINATIONS = [(cameras,) for cameras in [CAMERAS, {k:CAMERAS[k] for k in CAMERAS if k!='dcamera'}]]

# we check frame count, so we don't have to be too strict on size
FILE_SIZE_TOLERANCE = 0.5

class TestEncoder(unittest.TestCase):

  # TODO: all of loggerd should work on PC
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

  def setUp(self):
    self._clear_logs()
    os.environ["LOGGERD_TEST"] = "1"
    os.environ["LOGGERD_SEGMENT_LENGTH"] = str(SEGMENT_LENGTH)

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
  @with_processes(['camerad', 'sensord', 'loggerd'])
  def test_log_rotation(self, cameras):
    print("checking targets:", cameras)
    Params().put("RecordFront", "1" if 'dcamera' in cameras else "0")

    num_segments = random.randint(80, 150)
    if "CI" in os.environ:
      num_segments = random.randint(15, 20) # ffprobe is slow on comma two

    # wait for loggerd to make the dir for first segment
    route_prefix_path = None
    with Timeout(int(SEGMENT_LENGTH*2)):
      while route_prefix_path is None:
        try:
          route_prefix_path = self._get_latest_segment_path().rsplit("--", 1)[0]
        except Exception:
          time.sleep(0.1)
          continue

    def check_seg(i):
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

        # TODO: this ffprobe call is really slow
        # check frame count
        cmd = f"ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames \
               -of default=nokey=1:noprint_wrappers=1 {file_path}"
        expected_frames = SEGMENT_LENGTH * CAMERA_FPS // 2 if (EON and camera=='dcamera') else SEGMENT_LENGTH * CAMERA_FPS
        frame_tolerance = 1 if (EON and camera == 'dcamera') else 0
        frame_count = int(subprocess.check_output(cmd, shell=True, encoding='utf8').strip())

        self.assertTrue(abs(expected_frames - frame_count) <= frame_tolerance,
                        f"{camera} failed frame count check: expected {expected_frames}, got {frame_count}")
      shutil.rmtree(f"{route_prefix_path}--{i}")

    def join(ts, timeout):
      for t in ts:
        t.join(timeout)

    threads = []
    for i in trange(num_segments):
      # poll for next segment
      with Timeout(int(SEGMENT_LENGTH*2), error_msg=f"timed out waiting for segment {i}"):
        while int(self._get_latest_segment_path().rsplit("--", 1)[1]) <= i:
          time.sleep(0.1)
      t = threading.Thread(target=check_seg, args=(i, ))
      t.start()
      threads.append(t)
      join(threads, 0.1)

    with Timeout(20):
      join(threads, None)

if __name__ == "__main__":
  unittest.main()
