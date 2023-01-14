#!/usr/bin/env python3
import time
import unittest
import numpy as np

from selfdrive.test.helpers import with_processes
from system.camerad.snapshot.snapshot import get_snapshots

TEST_TIME = 45
REPEAT = 5

class TestCamerad(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    pass

  def _numpy_rgb2gray(self, im):
    ret = np.clip(im[:,:,2] * 0.114 + im[:,:,1] * 0.587 + im[:,:,0] * 0.299, 0, 255).astype(np.uint8)
    return ret

  def _is_exposure_okay(self, i, roi=None, med_mean=np.array([[0.2,0.4],[0.2,0.6]])):
    xmin, xmax, ymin, ymax = roi
    i = i[ymin:ymax,xmin:xmax]
    med_ex, mean_ex = med_mean
    i = self._numpy_rgb2gray(i)
    i_median = np.median(i) / 255.
    i_mean = np.mean(i) / 255.
    print([i_median, i_mean])
    return med_ex[0] < i_median < med_ex[1] and mean_ex[0] < i_mean < mean_ex[1]

  @with_processes(['camerad'])
  def test_camera_operation(self):
    passed = 0
    start = time.time()
    while time.time() - start < TEST_TIME and passed < REPEAT:
      rpic, dpic = get_snapshots(frame="roadCameraState", front_frame="driverCameraState")
      wpic, _ = get_snapshots(frame="wideRoadCameraState")

      res = self._is_exposure_okay(rpic, roi=[96, 1832, 604, 1112])
      res = res and self._is_exposure_okay(dpic, roi=[642, 1284, 96, 604])
      res = res and self._is_exposure_okay(wpic, roi=[642, 1284, 604, 1112])

      if passed > 0 and not res:
        passed = -passed # fails test if any failure after first sus
        break

      passed += int(res)
      time.sleep(2)
    self.assertGreaterEqual(passed, REPEAT)

if __name__ == "__main__":
  unittest.main()
