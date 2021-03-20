#!/usr/bin/env python3

import time
import unittest
import os
import numpy as np

from selfdrive.test.helpers import with_processes
from selfdrive.camerad.snapshot.snapshot import get_snapshots

from selfdrive.hardware import EON, TICI

WAIT_TIME = 15 # wait for cameras startup and adjustment

CAMERAS = {
  "roadCameraState": [[0.25,0.35],[0.2,0.6]],
  "driverCameraState": [[0.25,0.35],[0.2,0.6]],
}

os.environ["SEND_ROAD"] = "1"
os.environ["SEND_DRIVER"] = "1"

if TICI:
  os.environ["SEND_WIDE_ROAD"] = "1"
  CAMERAS["wideRoadCameraState"] = [[0.2,0.4],[0.2,0.6]]

class TestCamerad(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

  def _numpy_rgb2gray(self, im):
    ret = np.clip(im[:,:,2] * 0.114 + im[:,:,1] * 0.587 + im[:,:,0] * 0.299, 0, 255).astype(np.uint8)
    return ret

  def _is_exposure_okay(self, i, med_mean=np.array([[-1,-1], [-1,-1]])):
    med_ex, mean_ex = med_mean
    i = self._numpy_rgb2gray(i)
    i_median = np.median(i) / 256
    i_mean = np.mean(i) / 256
    print([i_median, i_mean])
    return med_ex[0] < i_median < med_ex[1] and mean_ex[0] < i_mean < mean_ex[1]


  @with_processes(['camerad'])
  def test_camera_operation(self):
    print("checking image outputs")

    time.sleep(WAIT_TIME)
    rpic, dpic = get_snapshots(frame="roadCameraState", front_frame="driverCameraState")

    self.assertTrue(self._is_exposure_okay(rpic, CAMERAS["roadCameraState"]))
    self.assertTrue(self._is_exposure_okay(dpic, CAMERAS["driverCameraState"]))

    if TICI:
      wpic, _ = get_snapshots(frame="wideRoadCameraState")
      self.assertTrue(self._is_exposure_okay(wpic, CAMERAS["wideRoadCameraState"]))

if __name__ == "__main__":
  unittest.main()
