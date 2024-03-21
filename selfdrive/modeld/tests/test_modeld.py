#!/usr/bin/env python3
import unittest
import numpy as np
import random

import cereal.messaging as messaging
from cereal.visionipc import VisionIpcServer, VisionStreamType
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.car_helpers import write_car_param
from openpilot.selfdrive.manager.process_config import managed_processes
from openpilot.selfdrive.test.process_replay.vision_meta import meta_from_camera_state

CAM = DEVICE_CAMERAS[("tici", "ar0231")].fcam
IMG = np.zeros(int(CAM.width*CAM.height*(3/2)), dtype=np.uint8)
IMG_BYTES = IMG.flatten().tobytes()


class TestModeld(unittest.TestCase):

  def setUp(self):
    self.vipc_server = VisionIpcServer("camerad")
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 40, False, CAM.width, CAM.height)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, 40, False, CAM.width, CAM.height)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 40, False, CAM.width, CAM.height)
    self.vipc_server.start_listener()
    write_car_param()

    self.sm = messaging.SubMaster(['modelV2', 'cameraOdometry'])
    self.pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'liveCalibration'])

    managed_processes['modeld'].start()
    self.pm.wait_for_readers_to_update("roadCameraState", 10)

  def tearDown(self):
    managed_processes['modeld'].stop()
    del self.vipc_server

  def _send_frames(self, frame_id, cams=None):
    if cams is None:
      cams = ('roadCameraState', 'wideRoadCameraState')

    cs = None
    for cam in cams:
      msg = messaging.new_message(cam)
      cs = getattr(msg, cam)
      cs.frameId = frame_id
      cs.timestampSof = int((frame_id * DT_MDL) * 1e9)
      cs.timestampEof = int(cs.timestampSof + (DT_MDL * 1e9))
      cam_meta = meta_from_camera_state(cam)

      self.pm.send(msg.which(), msg)
      self.vipc_server.send(cam_meta.stream, IMG_BYTES, cs.frameId,
                            cs.timestampSof, cs.timestampEof)
    return cs

  def _wait(self):
    self.sm.update(5000)
    if self.sm['modelV2'].frameId != self.sm['cameraOdometry'].frameId:
      self.sm.update(1000)

  def test_modeld(self):
    for n in range(1, 500):
      cs = self._send_frames(n)
      self._wait()

      mdl = self.sm['modelV2']
      self.assertEqual(mdl.frameId, n)
      self.assertEqual(mdl.frameIdExtra, n)
      self.assertEqual(mdl.timestampEof, cs.timestampEof)
      self.assertEqual(mdl.frameAge, 0)
      self.assertEqual(mdl.frameDropPerc, 0)

      odo = self.sm['cameraOdometry']
      self.assertEqual(odo.frameId, n)
      self.assertEqual(odo.timestampEof, cs.timestampEof)

  def test_dropped_frames(self):
    """
      modeld should only run on consecutive road frames
    """
    frame_id = -1
    road_frames = list()
    for n in range(1, 50):
      if (random.random() < 0.1) and n > 3:
        cams = random.choice([(), ('wideRoadCameraState', )])
        self._send_frames(n, cams)
      else:
        self._send_frames(n)
        road_frames.append(n)
      self._wait()

      if len(road_frames) < 3 or road_frames[-1] - road_frames[-2] == 1:
        frame_id = road_frames[-1]

      mdl = self.sm['modelV2']
      odo = self.sm['cameraOdometry']
      self.assertEqual(mdl.frameId, frame_id)
      self.assertEqual(mdl.frameIdExtra, frame_id)
      self.assertEqual(odo.frameId, frame_id)
      if n != frame_id:
        self.assertFalse(self.sm.updated['modelV2'])
        self.assertFalse(self.sm.updated['cameraOdometry'])


if __name__ == "__main__":
  unittest.main()
