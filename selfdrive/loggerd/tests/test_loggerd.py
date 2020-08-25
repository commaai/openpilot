#!/usr/bin/env python3
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

# baseline file sizes for a 1s segment, in bytes
FULL_SIZE = 626893
if EON:
  CAMERAS = {
    "fcamera": FULL_SIZE,
    "dcamera": 325460,
  }
elif TICI:
  CAMERAS = {f"{c}camera": FULL_SIZE for c in ["f", "e", "d"]}

rTOL = 0.1 # tolerate a 10% fluctuation based on content

class TestLoggerd(unittest.TestCase):

  # TODO: all of loggerd should work on PC
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

  def setUp(self):
    Params().put("RecordFront", "1")
    self._clear_logs()

    self.test_start = time.monotonic()
    self.segment_length = random.randint(1, 4)
    os.environ["LOGGERD_TEST"] = "1"
    os.environ["LOGGERD_SEGMENT_LENGTH"] = self.segment_length

  def tearDown(self):
    self._clear_logs()

  def _clear_logs(self):
    if os.path.exists(ROOT):
      shutil.rmtree(ROOT)

  def _get_last_route_path(self):
    last_route = sorted(Path(ROOT).iterdir(), key=os.path.getmtime)[-1]
    return os.path.join(ROOT, last_route)

  # TODO: this should run faster than real time
  @with_processes(['camerad'])
  def test_log_rotation(self):
    # wait for everything to init
    time.sleep(10)

    # get the route prefix
    route_path = self._get_last_route_path()
    print("LOGGING TO PATH: ", route_path)

    num_segments = random.randint(80, 150)
    for i in range(num_segments):
      if i < num_segments - 1:
        with Timeout(self.segment_length*2, error_msg=f"timed out waiting for segment {i}"):
          while not os.path.exists(os.path.join(route_path, str(i))):
            time.sleep(0.1)
      else:
        time.sleep(self.segment_length + 2)

      # check each camera file size
      for camera, _ in CAMERAS.items():
        f = os.path.join(ROOT, f"{i}/{camera}.hevc")
        self.assertTrue(os.path.exists(f))
        # TODO: check file size

if __name__ == "__main__":
  unittest.main()
