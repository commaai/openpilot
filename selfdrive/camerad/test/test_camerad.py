#!/usr/bin/env python3

import random
import time
import unittest
import numpy as np

import cereal.messaging as messaging
from selfdrive.test.helpers import with_processes
from selfdrive.camerad.snapshot.visionipc import VisionIPC

# only tests for EON and TICI
from selfdrive.hardware import EON, TICI

TEST_TIMESPAN = random.randint(60, 180) # seconds
SKIP_FRAME_TOLERANCE = 0
FRAME_COUNT_TOLERANCE = 1 # over the whole test time

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
    if EON:
      # run checks similar to prov
      time.sleep(15) # wait for startup and AF
      pic, fpic = self._get_snapshots()
      self.assertTrue(self._is_really_sharp(pic))
      self.assertTrue(self._is_exposure_okay(pic))
      self.assertTrue(self._is_exposure_okay(fpic))

      time.sleep(30)
      # check again for consistency
      pic, fpic = self._get_snapshots()
      self.assertTrue(self._is_really_sharp(pic))
      self.assertTrue(self._is_exposure_okay(pic))
      self.assertTrue(self._is_exposure_okay(fpic))
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
    while int(round(time.time() * 1000)) - start_time_milli < (TEST_TIMESPAN+1) * 1000:
      sm.update()

      for camera in CAMERAS:
        if sm.updated[camera]:
          if start_frame_id[camera] is None:
            start_frame_id[camera] = last_frame_id[camera] = sm[camera].frameId
            continue
          dfid = sm[camera].frameId - last_frame_id[camera]
          self.assertTrue(abs(dfid - 1) <= SKIP_FRAME_TOLERANCE)
          last_frame_id[camera] = sm[camera].frameId

      time.sleep(0.01)

    for camera in CAMERAS:
      print(camera, (last_frame_id[camera] - start_frame_id[camera]))
      self.assertTrue(abs((last_frame_id[camera] - start_frame_id[camera]) - TEST_TIMESPAN*CAMERAS[camera]) <= FRAME_COUNT_TOLERANCE)

if __name__ == "__main__":
  unittest.main()
