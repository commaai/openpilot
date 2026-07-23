#!/usr/bin/env python3
import threading
import os
import platform
from collections import namedtuple

from msgq.visionipc import VisionIpcServer, VisionStreamType
from openpilot.cereal import messaging

from openpilot.system.camerad.webcam.camera import Camera
from openpilot.common.realtime import Ratekeeper

ROAD_CAM = os.getenv("ROAD_CAM", "0")
WIDE_CAM = os.getenv("WIDE_CAM")
DRIVER_CAM = os.getenv("DRIVER_CAM")

CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id"])

CAMERAS = [CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, ROAD_CAM)]
if WIDE_CAM:
  CAMERAS.append(CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, WIDE_CAM))
if DRIVER_CAM:
  CAMERAS.append(CameraType("driverCameraState", VisionStreamType.VISION_STREAM_DRIVER, DRIVER_CAM))


def camera_device(camera_id: str, system: str | None = None):
  value = str(camera_id)
  system = platform.system() if system is None else system
  if system != "Darwin" and value.isdecimal():
    return f"/dev/video{value}"
  return camera_id


class Camerad:
  def __init__(self):
    self.pm = messaging.PubMaster([c.msg_name for c in CAMERAS])
    self.vipc_server = VisionIpcServer("camerad")

    self.cameras = []
    for c in CAMERAS:
      cam = Camera(c.msg_name, c.stream_type, camera_device(c.cam_id))
      self.cameras.append(cam)
      self.vipc_server.create_buffers(c.stream_type, 20, cam.W, cam.H)

    self.vipc_server.start_listener()

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type, timestamp_sof, timestamp_eof):
    self.vipc_server.send(yuv_type, yuv, frame_id, timestamp_sof, timestamp_eof)
    dat = messaging.new_message(pub_type, valid=True)
    msg = {"frameId": frame_id, "timestampSof": timestamp_sof, "timestampEof": timestamp_eof, "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)

  def camera_runner(self, cam):
    rk = Ratekeeper(20, None)
    for yuv, timestamp_sof, timestamp_eof in cam.read_frames():
      self._send_yuv(yuv, cam.cur_frame_id, cam.cam_type_state, cam.stream_type, timestamp_sof, timestamp_eof)
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
