import numpy as np
import unittest

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.modeld.compile_modeld import nv12_copy_size
from openpilot.tools.sim.lib.camerad import rgb_to_nv12, STRIDE, Y_HEIGHT, UV_HEIGHT, UV_OFFSET, YUV_SIZE
from openpilot.tools.sim.lib.common import W, H


def reference_nv12(rgb):
  """The tightly packed conversion, kept here so the pixel math stays pinned to what it was."""
  h, w = rgb.shape[:2]
  r = rgb[:, :, 0].astype(np.int32)
  g = rgb[:, :, 1].astype(np.int32)
  b = rgb[:, :, 2].astype(np.int32)
  y = np.clip((((b * 13 + g * 65 + r * 33) + 64) >> 7) + 16, 0, 255).astype(np.uint8)
  r_s = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2] + 2) >> 2
  g_s = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2
  b_s = (b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2] + 2) >> 2
  u = np.clip((b_s * 56 - g_s * 37 - r_s * 19 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  v = np.clip((r_s * 56 - g_s * 47 - b_s * 9 + 0x8080) >> 8, 0, 255).astype(np.uint8)
  uv = np.empty((h // 2, w), dtype=np.uint8)
  uv[:, 0::2] = u
  uv[:, 1::2] = v
  return y, uv


class TestSimCamerad(unittest.TestCase):
  def test_buffer_is_big_enough_for_modeld(self):
    # modeld copies nv12_copy_size bytes straight out of the VisionIPC buffer. A tightly packed
    # NV12 frame is smaller than that, and modeld dies on its first frame with
    # "buffer is smaller than requested size".
    needed = nv12_copy_size(STRIDE, Y_HEIGHT, UV_HEIGHT)
    assert YUV_SIZE >= needed, f"sim camera buffer {YUV_SIZE} < the {needed} modeld reads"
    assert YUV_SIZE > W * H * 3 // 2, "buffer is tightly packed, which is what broke modeld"

  def test_layout_matches_camerad(self):
    stride, y_height, uv_height, size = get_nv12_info(W, H)
    assert (STRIDE, Y_HEIGHT, UV_HEIGHT, YUV_SIZE) == (stride, y_height, uv_height, size)
    assert UV_OFFSET == stride * y_height

  def test_planes_land_where_consumers_look(self):
    rgb = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    buf = rgb_to_nv12(rgb)
    assert buf.shape == (YUV_SIZE,) and buf.dtype == np.uint8

    y_ref, uv_ref = reference_nv12(rgb)
    y = buf[:UV_OFFSET].reshape(Y_HEIGHT, STRIDE)
    uv = buf[UV_OFFSET:UV_OFFSET + STRIDE * UV_HEIGHT].reshape(UV_HEIGHT, STRIDE)
    assert np.array_equal(y[:H, :W], y_ref)
    assert np.array_equal(uv[:H // 2, :W], uv_ref)

    # the alignment padding has to be defined, not whatever was in the buffer before
    assert y[:, W:].max() == 0 and y[H:, :].max() == 0
    assert uv[:, W:].max() == 0 and uv[H // 2:, :].max() == 0

  def test_channel_order(self):
    # a red/blue swap on the way to VisionIPC is silent and ruins the model's input
    red = np.zeros((H, W, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    blue = np.zeros((H, W, 3), dtype=np.uint8)
    blue[:, :, 2] = 255
    y_red = rgb_to_nv12(red)[0]
    y_blue = rgb_to_nv12(blue)[0]
    assert y_red > y_blue, f"channel 0 must be red: got Y(red)={y_red}, Y(blue)={y_blue}"

  def test_reuses_the_output_buffer(self):
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    out = np.zeros(YUV_SIZE, dtype=np.uint8)
    assert rgb_to_nv12(rgb, out) is out
