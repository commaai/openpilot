#!/usr/bin/env python3
import threading
import os

from cereal.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.webcam.camera import Camera
from openpilot.common.realtime import Ratekeeper


YUV_BUFFER_COUNT = int(os.getenv("YUV_BUFFER_COUNT", "20"))
CAMERA_TYPE_STATE = {"roadCameraState":VisionStreamType.VISION_STREAM_ROAD,
                    "driverCameraState":VisionStreamType.VISION_STREAM_DRIVER,
                    "wideRoadCameraState":VisionStreamType.VISION_STREAM_WIDE_ROAD}

def rk_loop(function, hz, exit_event: threading.Event):
  rk = Ratekeeper(hz, None)
  while not exit_event.is_set():
    function()
    rk.keep_time()

class Camerad:
  def __init__(self):
    self.pm = messaging.PubMaster(['roadCameraState', 'driverCameraState', 'wideRoadCameraState'])
    self.vipc_server = VisionIpcServer("camerad")

    self.dual_camera = bool(int(os.getenv("DUAL","0")))
    self.cameras, self.camera_threads = [], [] # ORDER: road_cam, driver_cam, wide_cam

    for cam_type, stream_type in CAMERA_TYPE_STATE.items():
      if cam_type == "roadCameraState":
        cam = Camera(cam_type, stream_type, int(os.getenv("CAMERA_WIDE_ID", "0")))
      elif cam_type == "driverCameraState":
        cam = Camera(cam_type, stream_type, int(os.getenv("CAMERA_WIDE_ID", "1")))
      else:
        if not self.dual_camera:
          break
        cam = Camera(cam_type, stream_type, int(os.getenv("CAMERA_WIDE_ID", "2")))
      assert cam.cap.isOpened(), f"Can't find {cam_type}"
      self.cameras.append(cam)
      self.vipc_server.create_buffers(stream_type, YUV_BUFFER_COUNT, False, cam.W, cam.H)

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

  def daemon_alive(self, cam):
    rk = Ratekeeper(20, None)
    while cam.cap.isOpened():
      for yuv in cam.read_frames():
        self._send_yuv(yuv, cam.cur_frame_id, cam.cam_type_state, cam.stream_type)
        cam.cur_frame_id += 1
        rk.keep_time()

  def start_camera_threads(self):
    assert len(self.cameras) == 3
    for cam in self.cameras:
      if cam.cam_type_state == "wideRoadCameraState" and not self.dual_camera:
        break
      cam_thread = threading.Thread(target=self.daemon_alive, args=(cam,))
      cam_thread.start()
      self.camera_threads.append(cam_thread)

  def join_camera_threads(self):
    for thread in self.camera_threads:
      thread.join()

  def run(self):
    self.start_camera_threads()
    self.join_camera_threads()

if __name__ == "__main__":
  camerad = Camerad()
  camerad.run()
