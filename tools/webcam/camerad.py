#!/usr/bin/env python3
import threading
import os
from collections import namedtuple

from cereal.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.webcam.camera import Camera
from openpilot.common.realtime import Ratekeeper

DUAL_CAM = os.getenv("DUAL_CAMERA")
CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id"])
CAMERAS = [
  CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, os.getenv("CAMERA_ROAD_ID", "0")),
  CameraType("driverCameraState", VisionStreamType.VISION_STREAM_DRIVER, os.getenv("CAMERA_DRIVER_ID", "1")),
]
if DUAL_CAM:
  CAMERAS.append(CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, DUAL_CAM))

class Camerad:
  def __init__(self):
    self.pm = messaging.PubMaster([c.msg_name for c in CAMERAS])
    self.vipc_server = VisionIpcServer("camerad")

    self.cameras = []
    for c in CAMERAS:
      cam = Camera(c.msg_name, c.stream_type, c.cam_id)
      assert cam.cap.isOpened(), f"Can't find camera {c}"
      self.cameras.append(cam)
      self.vipc_server.create_buffers(c.stream_type, 20, False, cam.W, cam.H)

    self.vipc_server.start_listener()

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    eof = int(frame_id * 0.05 * 1e9)
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)
    dat = messaging.new_message(pub_type, valid=True)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)

  def camera_runner(self, cam):
    rk = Ratekeeper(20, None)
    while cam.cap.isOpened():
      for yuv in cam.read_frames():
        self._send_yuv(yuv, cam.cur_frame_id, cam.cam_type_state, cam.stream_type)
        cam.cur_frame_id += 1
        rk.keep_time()

  def run(self):
    threads = []
    for cam in self.cameras:
      cam_thread = threading.Thread(target=self.camera_runner, args=(cam,))
      cam_thread.start()
      threads.append(cam_thread)

    for t in threads:
      t.join()


def main():
  camerad = Camerad()
  camerad.run()


if __name__ == "__main__":
  main()
