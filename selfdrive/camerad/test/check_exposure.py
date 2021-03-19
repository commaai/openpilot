#!/usr/bin/env python3

import time
import unittest
import numpy as np

from selfdrive.test.helpers import with_processes
from selfdrive.camerad.snapshot.snapshot import get_snapshots

from selfdrive.hardware import EON, TICI

WAIT_TIME = 15 # wait for cameras startup and adjustment

class TestCamerad(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

  def _numpy_bgr2gray(self, im):
    ret = np.clip(im[:,:,0] * 0.114 + im[:,:,1] * 0.587 + im[:,:,2] * 0.299, 0, 255).astype(np.uint8)
    return ret

  def _numpy_lap(self, im):
    ret = np.zeros(im.shape)
    ret += -4 * im
    ret += np.concatenate([np.zeros((im.shape[0],1)),im[:,:-1]], axis=1)
    ret += np.concatenate([im[:,1:],np.zeros((im.shape[0],1))], axis=1)
    ret += np.concatenate([np.zeros((1,im.shape[1])),im[:-1,:]], axis=0)
    ret += np.concatenate([im[1:,:],np.zeros((1,im.shape[1]))], axis=0)
    ret = np.clip(ret, 0, 255).astype(np.uint8)
    return ret

  def _is_really_sharp(self, i, threshold=800, roi_max=np.array([8,6]), roi_xxyy=np.array([1,6,2,3])):
    i = self._numpy_bgr2gray(i)
    x_pitch = i.shape[1] // roi_max[0]
    y_pitch = i.shape[0] // roi_max[1]
    lap = self._numpy_lap(i)
    lap_map = np.zeros((roi_max[1], roi_max[0]))
    for r in range(lap_map.shape[0]):
      for c in range(lap_map.shape[1]):
        selected_lap = lap[r*y_pitch:(r+1)*y_pitch, c*x_pitch:(c+1)*x_pitch]
        lap_map[r][c] = 5*selected_lap.var() + selected_lap.max()
    print(lap_map[roi_xxyy[2]:roi_xxyy[3]+1,roi_xxyy[0]:roi_xxyy[1]+1])
    if (lap_map[roi_xxyy[2]:roi_xxyy[3]+1,roi_xxyy[0]:roi_xxyy[1]+1] > threshold).sum() > \
          (roi_xxyy[1]+1-roi_xxyy[0]) * (roi_xxyy[3]+1-roi_xxyy[2]) * 0.9:
      return True
    else:
      return False

  def _is_exposure_okay(self, i, med_ex=np.array([0.2,0.4]), mean_ex=np.array([0.2,0.6])):
    i = self._numpy_bgr2gray(i)
    i_median = np.median(i) / 256
    i_mean = np.mean(i) / 256
    print([i_median, i_mean])
    return med_ex[0] < i_median < med_ex[1] and mean_ex[0] < i_mean < mean_ex[1]


  @with_processes(['camerad'])
  def test_camera_operation(self):
    print("checking image outputs")

    time.sleep(WAIT_TIME)
    rpic, dpic = get_snapshots(frame="roadCameraState", front_frame="driverCameraState")

    self.assertTrue(self._is_exposure_okay(rpic))
    self.assertTrue(self._is_exposure_okay(dpic))

    if EON:
      self.assertTrue(self._is_really_sharp(rpic))
    elif TICI:
      time.sleep(1)
      wpic, _ = get_snapshots(frame="wideRoadCameraState")
      self.assertTrue(self._is_exposure_okay(wpic))

if __name__ == "__main__":
  unittest.main()
