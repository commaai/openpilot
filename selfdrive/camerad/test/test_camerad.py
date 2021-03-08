#!/usr/bin/env python3

import time
import unittest
import numpy as np

import cereal.messaging as messaging
from selfdrive.test.helpers import with_processes
from selfdrive.camerad.snapshot.snapshot import get_snapshots

# only tests for EON and TICI
from selfdrive.hardware import EON, TICI

TEST_TIMESPAN = 30 # random.randint(60, 180) # seconds
SKIP_FRAME_TOLERANCE = 0
LAG_FRAME_TOLERANCE = 2 # ms

FPS_BASELINE = 20
CAMERAS = {
  "roadCameraState": FPS_BASELINE,
  "driverCameraState": FPS_BASELINE // 2,
}

if TICI:
  CAMERAS["driverCameraState"] = FPS_BASELINE
  CAMERAS["wideRoadCameraState"] = FPS_BASELINE

class TestCamerad(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not (EON or TICI):
      raise unittest.SkipTest

    # assert "SEND_REAR" in os.environ
    # assert "SEND_FRONT" in os.environ

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

  @unittest.skip # skip for now
  @with_processes(['camerad'])
  def test_camera_operation(self):
    print("checking image outputs")
    if EON:
      # run checks similar to prov
      time.sleep(15) # wait for startup and AF
      pic, fpic = get_snapshots()
      self.assertTrue(self._is_really_sharp(pic))
      self.assertTrue(self._is_exposure_okay(pic))
      self.assertTrue(self._is_exposure_okay(fpic))

      time.sleep(30)
      # check again for consistency
      pic, fpic = get_snapshots()
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
    last_ts = dict.fromkeys(CAMERAS, None)
    start_time_sec = time.time()
    while time.time()- start_time_sec < TEST_TIMESPAN:
      sm.update()

      for camera in CAMERAS:
        if sm.updated[camera]:
          ct = (sm[camera].timestampEof if not TICI else sm[camera].timestampSof) / 1e6
          if last_frame_id[camera] is None:
            last_frame_id[camera] = sm[camera].frameId
            last_ts[camera] = ct
            continue

          dfid = sm[camera].frameId - last_frame_id[camera]
          self.assertTrue(abs(dfid - 1) <= SKIP_FRAME_TOLERANCE, "%s frame id diff is %d" % (camera, dfid))

          dts = ct - last_ts[camera]
          self.assertTrue(abs(dts - (1000/CAMERAS[camera])) < LAG_FRAME_TOLERANCE, "%s frame t(ms) diff is %f" % (camera, dts))

          last_frame_id[camera] = sm[camera].frameId
          last_ts[camera] = ct

      time.sleep(0.01)

if __name__ == "__main__":
  unittest.main()
