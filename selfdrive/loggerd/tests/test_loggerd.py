#!/usr/bin/env python3
import math
import os
import random
import shutil
import time
import unittest
from pathlib import Path

from common.params import Params
from common.hardware import EON, TICI
from common.timeout import Timeout
from selfdrive.test.helpers import with_processes
from selfdrive.loggerd.config import ROOT


# baseline file sizes for a 2s segment, in bytes
FULL_SIZE = 1253786
if EON:
  CAMERAS = {
    "fcamera": FULL_SIZE,
    "dcamera": 770920,
  }
elif TICI:
  CAMERAS = {f"{c}camera": FULL_SIZE for c in ["f", "e", "d"]}

FILE_SIZE_TOLERANCE = 0.25

class TestLoggerd(unittest.TestCase):

  # TODO: all of loggerd should work on PC
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

  def setUp(self):
    Params().put("RecordFront", "1")
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
  @with_processes(['camerad', 'loggerd'], init_time=5)
  def test_log_rotation(self):
    # wait for first seg to start being written
    time.sleep(5)
    route_prefix_path = self._get_latest_segment_path().rsplit("--", 1)[0]

    num_segments = random.randint(80, 150)
    for i in range(num_segments):
      if i < num_segments - 1:
        with Timeout(self.segment_length*3, error_msg=f"timed out waiting for segment {i}"):
          while True:
            seg_num = int(self._get_latest_segment_path().rsplit("--", 1)[1])
            if seg_num > i:
              break
            time.sleep(0.1)
      else:
        time.sleep(self.segment_length + 2)

      # check each camera file size
      for camera, size in CAMERAS.items():
        f = f"{route_prefix_path}--{i}/{camera}.hevc"
        self.assertTrue(os.path.exists(f), f"couldn't find {f}")
        file_size = os.path.getsize(f)
        self.assertTrue(math.isclose(file_size, size, rel_tol=FILE_SIZE_TOLERANCE), 
                        f"{camera} failed size check: expected {size}, got {file_size}")

if __name__ == "__main__":
  unittest.main()
