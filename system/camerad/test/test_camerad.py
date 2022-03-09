#!/usr/bin/env python3

import time
import unittest

import cereal.messaging as messaging
from selfdrive.test.helpers import with_processes

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
          self.assertTrue(abs(dts - (1000/CAMERAS[camera])) < LAG_FRAME_TOLERANCE, f"{camera} frame t(ms) diff is {dts:f}")

          last_frame_id[camera] = sm[camera].frameId
          last_ts[camera] = ct

      time.sleep(0.01)

if __name__ == "__main__":
  unittest.main()
