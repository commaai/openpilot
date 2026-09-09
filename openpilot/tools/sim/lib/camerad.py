import time

import numpy as np

from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcServer
from openpilot.cereal import messaging
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

from openpilot.tools.sim.lib.common import W, H

# Consumers read the frame straight out of the VisionIPC buffer, so it has to be laid out the way
# camerad lays it out: rows padded to a 128 byte stride and the plane heights aligned. Packing NV12
# tightly instead leaves modeld short of the bytes it copies, and it dies on the first frame.
STRIDE, Y_HEIGHT, UV_HEIGHT, YUV_SIZE = get_nv12_info(W, H)
UV_OFFSET = STRIDE * Y_HEIGHT


def rgb_to_nv12(rgb, out=None):
  """Convert an RGB image to a camerad-shaped NV12 buffer using BT.601 coefficients."""
  h, w = rgb.shape[:2]
  r = rgb[:, :, 0].astype(np.int32)
  g = rgb[:, :, 1].astype(np.int32)
  b = rgb[:, :, 2].astype(np.int32)

  # Y plane - BT.601 coefficients (matches original OpenCL kernel)
  y = (((b * 13 + g * 65 + r * 33) + 64) >> 7) + 16
  y = np.clip(y, 0, 255).astype(np.uint8)

  # Subsample RGB for UV (2x2 box filter)
  r_sub = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2] + 2) >> 2
  g_sub = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2
  b_sub = (b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2] + 2) >> 2

  # U and V planes
  u = np.clip((b_sub * 56 - g_sub * 37 - r_sub * 19 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  v = np.clip((r_sub * 56 - g_sub * 47 - b_sub * 9 + 0x8080) >> 8, 0, 255).astype(np.uint8)

  if out is None:
    out = np.zeros(YUV_SIZE, dtype=np.uint8)

  # Write into the padded planes, leaving the alignment padding zeroed
  out[:UV_OFFSET].reshape(Y_HEIGHT, STRIDE)[:h, :w] = y
  uv = out[UV_OFFSET:UV_OFFSET + STRIDE * UV_HEIGHT].reshape(UV_HEIGHT, STRIDE)
  uv[:h // 2, 0:w:2] = u
  uv[:h // 2, 1:w:2] = v

  return out


class Camerad:
  """Simulates the camerad daemon"""
  def __init__(self, dual_camera):
    self.pm = messaging.PubMaster(['narrowRoadCameraState', 'wideRoadCameraState'])

    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    self.vipc_server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_NARROW_ROAD, 5, W, H, YUV_SIZE, STRIDE, UV_OFFSET)
    if dual_camera:
      self.vipc_server.create_buffers_with_sizes(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, W, H, YUV_SIZE, STRIDE, UV_OFFSET)

    self.vipc_server.start_listener()

    # one scratch buffer per stream, these are 4.8MB each
    self.yuv_bufs = {t: np.zeros(YUV_SIZE, dtype=np.uint8) for t in
                     (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)}

  def cam_send_yuv_road(self, yuv):
    self._send_yuv(yuv, self.frame_road_id, 'narrowRoadCameraState', VisionStreamType.VISION_STREAM_NARROW_ROAD)
    self.frame_road_id += 1

  def cam_send_yuv_wide_road(self, yuv):
    self._send_yuv(yuv, self.frame_wide_id, 'wideRoadCameraState', VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def rgb_to_yuv(self, rgb, yuv_type=VisionStreamType.VISION_STREAM_NARROW_ROAD):
    """Convert RGB to NV12 YUV format."""
    assert rgb.shape == (H, W, 3), f"{rgb.shape}"
    assert rgb.dtype == np.uint8
    return rgb_to_nv12(rgb, self.yuv_bufs[yuv_type])

  def _send_yuv(self, yuv, frame_id, pub_type, yuv_type):
    # same clock as logMonoTime, otherwise locationd rejects the camera odometry these frames
    # produce as out of range and never gets a valid pose
    eof = int(time.monotonic() * 1e9)
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
