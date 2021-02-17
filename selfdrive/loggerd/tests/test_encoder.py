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
from common.timeout import Timeout
from selfdrive.hardware import EON, TICI
from selfdrive.test.helpers import with_processes
from selfdrive.loggerd.config import ROOT


SEGMENT_LENGTH = 2
if EON:
  FULL_SIZE = 1253786 # file size for a 2s segment in bytes
  CAMERAS = [
    ("fcamera.hevc", 20, FULL_SIZE),
    ("dcamera.hevc", 10, 770920),
    ("qcamera.ts", 20, 77066),
  ]
else:
  FULL_SIZE = 2507572
  CAMERAS = [
    ("fcamera.hevc", 20, FULL_SIZE),
    ("dcamera.hevc", 20, FULL_SIZE),
    ("ecamera.hevc", 20, FULL_SIZE),
    ("qcamera.ts", 20, 77066),
  ]

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
  @parameterized.expand([(True, ), (False, )])
  @with_processes(['camerad', 'sensord', 'loggerd'], init_time=3)
  def test_log_rotation(self, record_front):
    Params().put("RecordFront", str(int(record_front)))

    num_segments = int(os.getenv("SEGMENTS", random.randint(10, 15)))

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
      for camera, fps, size in CAMERAS:
        if not record_front and "dcamera" in camera:
          continue

        file_path = f"{route_prefix_path}--{i}/{camera}"

        # check file size
        self.assertTrue(os.path.exists(file_path))
        file_size = os.path.getsize(file_path)
        self.assertTrue(math.isclose(file_size, size, rel_tol=FILE_SIZE_TOLERANCE))

        # TODO: this ffprobe call is really slow
        # check frame count
        cmd = f"ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames \
                -of default=nokey=1:noprint_wrappers=1 {file_path}"
        expected_frames = fps * SEGMENT_LENGTH
        frame_tolerance = 1 if (EON and camera == 'dcamera.hevc') else 0
        probe = subprocess.check_output(cmd, shell=True, encoding='utf8')
        frame_count = int(probe.split('\n')[0].strip())
        counts.append(frame_count)

        if EON:
          self.assertTrue(abs(expected_frames - frame_count) <= frame_tolerance,
                          f"{camera} failed frame count check: expected {expected_frames}, got {frame_count}")
        else:
          # loggerd waits for the slowest camera, so check count is at least the expected count,
          # then check the min of the frame counts is exactly the expected frame count
          self.assertTrue(frame_count >= expected_frames,
                          f"{camera} failed frame count check: expected {expected_frames}, got {frame_count}")

      if TICI:
        expected_frames = fps * SEGMENT_LENGTH
        self.assertEqual(min(counts), expected_frames)
      shutil.rmtree(f"{route_prefix_path}--{i}")

    for i in trange(num_segments):
      # poll for next segment
      with Timeout(int(SEGMENT_LENGTH*2), error_msg=f"timed out waiting for segment {i}"):
        while int(self._get_latest_segment_path().rsplit("--", 1)[1]) <= i:
          time.sleep(0.1)
      check_seg(i)

if __name__ == "__main__":
  unittest.main()
