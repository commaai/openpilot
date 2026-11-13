import functools
import raylib_nb as rl
import numpy as np
from dataclasses import dataclass
from typing import Optional
from openpilot.system.ui.lib.application import gui_app


@dataclass
class Gradient:
  start: tuple[float, float]
  end: tuple[float, float]
  colors: list[rl.Color]
  stops: list[float]

  def __post_init__(self):
    if not len(self.stops):
      self.stops = [i / max(1, len(self.colors) - 1) for i in range(len(self.colors))]


def _pack(color: rl.Color) -> int:
  return color.r | color.g << 8 | color.b << 16 | color.a << 24


@functools.lru_cache(maxsize=16)
def _ramp(colors: tuple[tuple[int, int, int, int], ...], stops: tuple[float, ...]) -> np.ndarray:
  """the gradient's colors at 256 steps of t"""
  t = np.linspace(0.0, 1.0, 256)
  channels = np.array(colors, dtype=np.float32)
  ramp = np.empty((256, 4), dtype=np.uint8)
  for i in range(4):
    ramp[:, i] = np.rint(np.interp(t, stops, channels[:, i]))
  return ramp.view(np.uint32).ravel()


def _gradient_ramp(gradient: Gradient, origin_rect: rl.Rectangle) -> tuple[np.ndarray, float, float, float]:
  """the gradient for rl.cpu.polygon: its ramp of colors, and t's coefficients in x and y.
  t measures y from the bottom of the screen, like the GL shader the UI's gradients were made with"""
  start_x = origin_rect.x + gradient.start[0] * origin_rect.width
  start_y = origin_rect.y + gradient.start[1] * origin_rect.height
  end_x = origin_rect.x + gradient.end[0] * origin_rect.width
  end_y = origin_rect.y + gradient.end[1] * origin_rect.height
  dx = start_x - end_x
  dy = start_y - end_y
  length2 = max(dx * dx + dy * dy, 1e-6)

  # t = ((x, height - y) - end) . (dx, dy) / length2
  tx = dx / length2
  ty = -dy / length2
  t0 = (-end_x * dx + (gui_app.height - end_y) * dy) / length2

  stops = tuple(min(max(float(s), 0.0), 1.0) for s in gradient.stops[:len(gradient.colors)])
  colors = tuple((c.r, c.g, c.b, c.a) for c in gradient.colors[:len(stops)])
  return _ramp(colors, stops), tx, ty, t0


def draw_polygon(origin_rect: rl.Rectangle, points: np.ndarray,
                 color: Optional[rl.Color] = None, gradient: Gradient | None = None):  # noqa: UP045  # rl.Color is a function, so `rl.Color | None` fails
  """Fill a polygon with a color or a gradient, by the nonzero winding rule"""
  assert (color is not None) != (gradient is not None), "Either color or gradient must be provided"
  if len(points) < 3:
    return

  pts = np.ascontiguousarray(points, dtype=np.float32)
  rl.cpu.target_raylib()
  if gradient is not None:
    rl.cpu.polygon(pts, *_gradient_ramp(gradient, origin_rect))
  else:
    rl.cpu.polygon(pts, np.array([_pack(color)], np.uint32))
