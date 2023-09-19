import cv2
import numpy as np

from cereal.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.tools.sim.lib.common import W, H


# Modified from: https://github.com/opencv/opencv/issues/22514#issuecomment-1369457841
def rgb2nv12(rgb):
    yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV_I420)
    uv_row_cnt = yuv.shape[0] // 3
    uv_plane = np.transpose(yuv[uv_row_cnt * 2:].reshape(2, -1), [1, 0])
    yuv[uv_row_cnt * 2:] = uv_plane.reshape(uv_row_cnt, -1)
    return yuv


class Camerad:
  """Simulates the camerad daemon"""
  def __init__(self, dual_camera):
    self.pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState'])

    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, W, H)
    if dual_camera:
      self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, W, H)

    self.vipc_server.start_listener()

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'roadCameraState', VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  # Returns: yuv bytes
  def rgb_to_yuv(self, rgb):
    assert rgb.shape == (H, W, 3), f"{rgb.shape}"
    assert rgb.dtype == np.uint8

    return rgb2nv12(rgb).tobytes()

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    eof = int(frame_id * 0.05 * 1e9)
    self.vipc_server.send(yuv_type, yuv, frame_id, eof, eof)

    dat = messaging.new_message(pub_type)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)