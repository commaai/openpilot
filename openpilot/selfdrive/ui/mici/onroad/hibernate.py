"""
While disengaged the road camera hibernates so it doesn't ask for attention: focus lets go first, then color drains,
then it settles dim with a slow breath. It follows status_fade, so it moves with the rest of the onroad UI and engaging
plays it backwards. Only the camera frame changes, overlays draw on top.
"""
import math
from collections.abc import Callable
import numpy as np
import raylib_nb as rl

from openpilot.system.ui.lib.application import gui_app

BREATH_PERIOD = 6.5  # seconds
BREATH_DEPTH = 0.07  # of exposure
REST_EXPOSURE = 0.5
REST_SATURATION = 0.3

BLUR_SCALE = 2  # the blurred frame is made at half resolution
VIGNETTE = 0.55
IDENTITY_CURVE = np.round(np.clip((np.arange(1280) - 512) / 255, 0, 1) * 255).astype(np.uint8)


def _smoother(a: float, b: float, x: float) -> float:
  t = min(max((x - a) / (b - a), 0.0), 1.0)
  return t * t * t * (t * (t * 6 - 15) + 10)


class Hibernate:
  """stages of the camera's sleep, from how disengaged the UI is (0 to 1)"""
  def __init__(self):
    self.focus = self.drain = self.dim = self.breath = 0.0
    self._rest = 0.0  # time spent asleep, held while waking so the breath fades out instead of jumping

  def update(self, level: float):
    if level > 0.99:
      self._rest += 1 / gui_app.target_fps
    elif level < 0.01:
      self._rest = 0.0

    # focus goes first, then color, then light. waking is the same in reverse
    self.focus, self.drain, self.dim = _smoother(0.0, 0.7, level), _smoother(0.2, 0.9, level), _smoother(0.35, 1.0, level)
    breath = 0.5 - 0.5 * math.cos(2 * math.pi * self._rest / BREATH_PERIOD)
    self.breath = breath * _smoother(0.0, 1.5, self._rest) * level

  @property
  def exposure(self) -> float:
    return (1 - (1 - REST_EXPOSURE) * self.dim) * (1 + BREATH_DEPTH * self.breath)

  def saturation(self, awake_saturation: float) -> float:
    return awake_saturation + (REST_SATURATION - awake_saturation) * self.drain


class BlurredFrame:
  """the camera frame at half resolution, blurred by averaging 4x4 blocks and scaling them back up, vignetted, and faded
  in by focus. it's built as nv12 so it draws as cheaply as the camera does, and only rebuilt when that changes"""
  def __init__(self):
    self._rgb = np.zeros((0, 0, 4), np.uint8)
    self._vignette = np.zeros((0, 0, 1), np.float32)
    self._nv12 = np.zeros(0, np.uint8)
    self._key: tuple | None = None

  def draw(self, draw_camera: Callable[[rl.Rectangle, np.ndarray, float, bool], None], rect: rl.Rectangle, dst_rect: rl.Rectangle,
           new_frame: bool, focus: float, vignette: float, curve: np.ndarray, saturation: float):
    w, h = 4 * math.ceil(rect.width / (4 * BLUR_SCALE)), 4 * math.ceil(rect.height / (4 * BLUR_SCALE))
    key = (w, h, round(focus, 3), round(vignette, 3))
    rebuild = new_frame or key != self._key
    if rebuild:
      if self._rgb.shape[:2] != (h, w):
        self._rgb = np.zeros((h, w, 4), np.uint8)
        y, x = np.mgrid[0:h // 4, 0:w // 4].astype(np.float32) * 4 + 2
        t = np.clip((np.hypot(x / h - 0.5 * w / h, y / h - 0.5) - 0.35) / 0.7, 0, 1)
        self._vignette = (VIGNETTE * t * t * (3 - 2 * t))[..., None]
      # uncolored, the curve and saturation apply when it's drawn
      rl.cpu.target(self._rgb, w, h, w * 4)
      draw_camera(rl.Rectangle((dst_rect.x - rect.x) / BLUR_SCALE, (dst_rect.y - rect.y) / BLUR_SCALE,
                               dst_rect.width / BLUR_SCALE, dst_rect.height / BLUR_SCALE), IDENTITY_CURVE, 1.0, True)

      # blur: average 4x4 blocks, vignette and convert those, then scale back up
      rgb = self._rgb[..., :3]
      blocks = rgb[0::2].astype(np.uint16) + rgb[1::2]
      blocks = blocks[0::2] + blocks[1::2]
      blocks = blocks[:, 0::2] + blocks[:, 1::2]
      blocks = blocks[:, 0::2] + blocks[:, 1::2]
      yuv = _rgb_to_yuv(blocks * ((1 - vignette * self._vignette) / 16))
      y_plane, uv_plane = _upsample(yuv[..., 0], 4), _upsample(yuv[..., 1:], 2)
      if focus < 1:
        sharp = _rgb_to_yuv(rgb.astype(np.float32))
        y_plane = sharp[..., 0] + (y_plane - sharp[..., 0]) * focus
        uv_plane = sharp[::2, ::2, 1:] + (uv_plane - sharp[::2, ::2, 1:]) * focus
      self._nv12 = np.concatenate([y_plane.ravel(), uv_plane.ravel()]).clip(0, 255).astype(np.uint8)
      self._key = key

    rl.cpu.target_raylib()
    rl.cpu.nv12(self._nv12, w * h, w, w, h, rl.Rectangle(rect.x, rect.y, w * BLUR_SCALE, h * BLUR_SCALE), False, curve,
                round(saturation * 256), rebuild)


# BT.601 limited range, what rl.cpu.nv12 draws
RGB_TO_YUV = np.array([[0.256788, -0.148223, 0.439216], [0.504129, -0.290993, -0.367788], [0.097906, 0.439216, -0.071427]], np.float32)
YUV_OFFSET = np.array([16, 128, 128], np.float32)


def _rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
  return rgb @ RGB_TO_YUV + YUV_OFFSET


def _upsample(a: np.ndarray, n: int) -> np.ndarray:
  """linear interpolation between pixel centers to n times the size, in the first two axes"""
  for _ in range(2):
    p = np.concatenate([a[:1], a, a[-1:]])
    out = np.empty((a.shape[0], n) + a.shape[1:], np.float32)
    for m in range(n):
      o = (m + 0.5) / n - 0.5
      out[:, m] = p[:-2] * -o + p[1:-1] * (1 + o) if o < 0 else p[1:-1] * (1 - o) + p[2:] * o
    a = out.reshape((a.shape[0] * n,) + a.shape[1:]).swapaxes(0, 1)
  return a
