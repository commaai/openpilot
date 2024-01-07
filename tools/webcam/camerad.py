#!/usr/bin/env python3
import threading
import os

from cereal.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.webcam.camera import Camera

YUV_BUFFER_COUNT = 20 # same as c++ of camerad
CAMERA_ROAD_ID, CAMERA_DRIVER_ID, CAMERA_WIDE_ID = 0, 1, 2

class Camerad:
  def __init__(self):
    self.pm = messaging.PubMaster(['roadCameraState', 'driverCameraState', 'wideRoadCameraState'])
    self.vipc_server = VisionIpcServer("camerad")
    self.frame_road_id, self.frame_driver_id, self.frame_wide_id = 0, 0, 0
    self.dual_camera = bool(int(os.getenv("DUAL","0")))

    self.cam_road = Camera(CAMERA_ROAD_ID)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, YUV_BUFFER_COUNT, False, self.cam_road.W, self.cam_road.H)
    assert self.cam_road.cam.isOpened(), "Can't find road camera"
    self.cam_driver = Camera(CAMERA_DRIVER_ID)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_DRIVER, YUV_BUFFER_COUNT, False, self.cam_driver.W, self.cam_driver.H)
    assert self.cam_driver.cam.isOpened(), "Can't find driver camera"
    if self.dual_camera:
      self.cam_wide = Camera(CAMERA_WIDE_ID)
      self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, YUV_BUFFER_COUNT, False, self.cam_wide.W, self.cam_wide.H)
      assert self.cam_driver.cam.isOpened(), "Can't find wide road camera"

    self.vipc_server.start_listener()

  #from sim
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

  @classmethod
  def daemon_alive(self, cam, send_yuv):
    while cam.cam.isOpened():
      for yuv in cam.read_frames():
        send_yuv(yuv)

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_driver(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'driverCameraState', VisionStreamType.VISION_STREAM_DRIVER)
    self.frame_driver_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def run(self):
    self.t_cam_road = threading.Thread(target=Camerad.daemon_alive, args=(self.cam_road, self.cam_send_yuv_road))
    self.t_cam_road.start()
    self.t_cam_driver = threading.Thread(target=Camerad.daemon_alive, args=(self.cam_driver, self.cam_send_yuv_driver))
    self.t_cam_driver.start()
    if self.dual_camera:
      self.t_cam_wide = threading.Thread(target=Camerad.daemon_alive, args=(self.cam_wide, self.cam_send_yuv_wide_road))
      self.t_cam_wide.start()

    self.t_cam_road.join()
    self.t_cam_driver.join()
    if self.dual_camera:
      self.t_cam_wide.join()

if __name__ == "__main__":
  camerad = Camerad()
  camerad.run()
