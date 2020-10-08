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
from selfdrive.camerad.snapshot.visionipc import VisionIPC

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

  def _get_snapshots(self):
    ret = None
    start_time = time.time()
    while time.time() - start_time < 5.0:
      try:
        ipc = VisionIPC()
        pic = ipc.get()
        del ipc

        ipc_front = VisionIPC(front=True) # need to add another for tici
        fpic = ipc_front.get()
        del ipc_front

        ret = pic, fpic
        break
      except Exception:
        time.sleep(1)
    return ret

  def _is_really_sharp(i, threshold=800, roi_max=[8,6], roi_xxyy=[1,6,2,3]):
      i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
      x_pitch = i.shape[1] // roi_max[0]
      y_pitch = i.shape[0] // roi_max[1]
      lap = cv2.Laplacian(i, cv2.CV_16S)
      lap_map = numpy.zeros((roi_max[1], roi_max[0]))
      for r in range(lap_map.shape[0]):
        for c in range(lap_map.shape[1]):
          selected_lap = lap[r*y_pitch:(r+1)*y_pitch, c*x_pitch:(c+1)*x_pitch]
          lap_map[r][c] = 5*selected_lap.var() + selected_lap.max()
      if (lap_map[roi_xxyy[2]:roi_xxyy[3]+1,roi_xxyy[0]:roi_xxyy[1]+1] > threshold).sum() > \
            (roi_xxyy[1]+1-roi_xxyy[0]) * (roi_xxyy[3]+1-roi_xxyy[2]) * 0.9:
        return True
      else:
        return False

  @with_processes(['camerad'], init_time=15) # wait for startup and AF
  def test_camera_operation(self):
    print("checking image outputs")
    if EON:
      # run checks similar to prov
      pic, fpic = self._get_snapshots()
      self.assertTrue(self._is_really_sharp(pic))

      time.sleep(30)

      # check again for consistency
      pic, fpic = self._get_snapshots()
      self.assertTrue(self._is_really_sharp(pic))

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
          self.assertTrue(dfid - 1 <= SKIP_FRAME_TOLERANCE)
          last_frame_id[camera] = sm[camera].frameId

      time.sleep(0.01)

    for camera in CAMERAS:
      print(camera, (last_frame_id[camera] - start_frame_id[camera]))
      self.assertTrue((last_frame_id[camera] - start_frame_id[camera]) - TEST_TIMESPAN*CAMERAS[camera] <= FRAME_COUNT_TOLERANCE)

if __name__ == "__main__":
  unittest.main()
