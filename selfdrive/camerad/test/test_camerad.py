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

import cereal.messaging as messaging
from common.params import Params
from common.timeout import Timeout
from selfdrive.test.helpers import with_processes

from common.hardware import EON, TICI
# only tests for EON and TICI

TEST_TIMESPAN = random.randint(60, 180) # seconds
SKIP_FRAME_TOLERANCE = 0
FRAME_COUNT_TOLERANCE = 0 # over the whole test time

FPS_BASELINE = 20
CAMERAS = {
  "frame": FPS_BASELINE,
  "frontFrame": FPS_BASELINE // 2,
}

if TICI:
  CAMERAS["frontFrame"] = FPS_BASELINE
  CAMERAS["wideFrame"] = FPS_BASELINE

class TestCamerad(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

  def setUp(self):
    pass

  def tearDown(self):
    pass

  def _get_latest_segment_path(self):
    last_route = sorted(Path(ROOT).iterdir(), key=os.path.getmtime)[-1]
    return os.path.join(ROOT, last_route)

  @with_processes(['camerad'])
  def test_camera_operation(self):
    if EON:
      assertTrue(1)
      # run checks similar to prov
    elif TICI:
      raise unittest.SkipTest # TBD
    else:
      raise unittest.SkipTest

  @with_processes(['camerad'])
  def test_frame_packets(self):
    print("checking frame pkts continuity")
    print(TEST_TIMESPAN)

    sm = messaging.SubMaster([socket_name for socket_name in CAMERAS])

    last_frame_id = dict.fromkeys(CAMERAS, None)
    start_frame_id = dict.fromkeys(CAMERAS, None)
    start_time_milli = int(round(time.time() * 1000))
    while int(round(time.time() * 1000)) - start_time_milli < TEST_TIMESPAN * 1000:
      sm.update()

      for camera in CAMERAS:
        if sm.updated[camera]:
          if start_frame_id[camera] is None:
            start_frame_id[camera] = last_frame_id[camera] = sm[camera].frameId
            continue
          dfid = sm[camera].frameId - last_frame_id[camera]
          assertTrue(dfid - 1 <= SKIP_FRAME_TOLERANCE)
          last_frame_id[camera] = sm[camera].frameId

      time.sleep(0.01)

    for camera in CAMERAS:
      print(camera, (last_frame_id[camera] - start_frame_id[camera]))
      assertTrue((last_frame_id[camera] - start_frame_id[camera]) - TEST_TIMESPAN*CAMERAS[camera] <= FRAME_COUNT_TOLERANCE)

if __name__ == "__main__":
  unittest.main()
