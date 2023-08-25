#!/usr/bin/env python3
import time
import unittest
import numpy as np

from openpilot.selfdrive.test.helpers import with_processes, phone_only
from openpilot.system.camerad.snapshot.snapshot import get_snapshots

TEST_TIME = 45
REPEAT = 5

class TestCamerad(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    pass

  def _numpy_rgb2gray(self, im):
    ret = np.clip(im[:,:,2] * 0.114 + im[:,:,1] * 0.587 + im[:,:,0] * 0.299, 0, 255).astype(np.uint8)
    return ret

  def _is_exposure_okay(self, i, med_mean=None):
    if med_mean is None:
      med_mean = np.array([[0.2,0.4],[0.2,0.6]])
    h, w = i.shape[:2]
    i = i[h//10:9*h//10,w//10:9*w//10]
    med_ex, mean_ex = med_mean
    i = self._numpy_rgb2gray(i)
    i_median = np.median(i) / 255.
    i_mean = np.mean(i) / 255.
    print([i_median, i_mean])
    return med_ex[0] < i_median < med_ex[1] and mean_ex[0] < i_mean < mean_ex[1]

  @phone_only
  @with_processes(['camerad'])
  def test_camera_operation(self):
    passed = 0
    start = time.time()
    while time.time() - start < TEST_TIME and passed < REPEAT:
      rpic, dpic = get_snapshots(frame="roadCameraState", front_frame="driverCameraState")
      wpic, _ = get_snapshots(frame="wideRoadCameraState")

      res = self._is_exposure_okay(rpic)
      res = res and self._is_exposure_okay(dpic)
      res = res and self._is_exposure_okay(wpic)

      if passed > 0 and not res:
        passed = -passed # fails test if any failure after first sus
        break

      passed += int(res)
      time.sleep(2)
    self.assertGreaterEqual(passed, REPEAT)

if __name__ == "__main__":
  unittest.main()
