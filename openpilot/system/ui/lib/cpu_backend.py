"""Raylib-compatible CPU backend for GPU-free UI rendering.

It implements the public 2D surface used by the UIs rather than emulating
rlgl. It is loaded only when RAYLIB_BACKEND=cpu.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import re
import struct
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pyray as _pyray

ffi = _pyray.ffi
_LIBRARY = pathlib.Path(__file__).resolve().parent / "cpu_renderer/libraylib_cpu.so"
_TICI = pathlib.Path("/TICI").exists()


class _Surface(ctypes.Structure):
  _fields_ = [
    ("pixels", ctypes.POINTER(ctypes.c_uint8)),
    ("width", ctypes.c_int),
    ("height", ctypes.c_int),
    ("stride", ctypes.c_int),
    ("clip_x0", ctypes.c_int),
    ("clip_y0", ctypes.c_int),
    ("clip_x1", ctypes.c_int),
    ("clip_y1", ctypes.c_int),
  ]


class _Point(ctypes.Structure):
  _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class _BlitItem(ctypes.Structure):
  _fields_ = [
    ("surface", ctypes.POINTER(_Surface)),
    ("source_x", ctypes.c_float),
    ("source_y", ctypes.c_float),
    ("source_width", ctypes.c_float),
    ("source_height", ctypes.c_float),
    ("destination_x", ctypes.c_int),
    ("destination_y", ctypes.c_int),
    ("destination_width", ctypes.c_int),
    ("destination_height", ctypes.c_int),
  ]


def _load_native() -> ctypes.CDLL:
  lib = ctypes.CDLL(str(_LIBRARY))
  sp = ctypes.POINTER(_Surface)
  lib.sr_clear.argtypes = [sp, ctypes.c_uint32]
  lib.sr_set_clip.argtypes = [sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  lib.sr_reset_clip.argtypes = [sp]
  lib.sr_rect.argtypes = [sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
  lib.sr_gradient_v.argtypes = [
    sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
  ]
  lib.sr_gradient_4.argtypes = [
    sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
  ]
  lib.sr_rounded_rect.argtypes = [
    sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_uint32,
  ]
  lib.sr_circle.argtypes = [sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
  lib.sr_circle_gradient.argtypes = [
    sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
  ]
  lib.sr_ring_arc.argtypes = [
    sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_uint32,
  ]
  lib.sr_line.argtypes = [
    sp, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_uint32,
  ]
  lib.sr_triangle.argtypes = [sp, _Point, _Point, _Point, ctypes.c_uint32]
  lib.sr_ribbon.argtypes = [
    sp, ctypes.POINTER(_Point), ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_uint32,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
  ]
  lib.sr_blit_scaled.argtypes = [
    sp, sp, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
  ]
  lib.sr_blit_opaque.argtypes = [sp, sp, ctypes.c_int, ctypes.c_int]
  lib.sr_blit_many.argtypes = [sp, ctypes.POINTER(_BlitItem), ctypes.c_int, ctypes.c_uint32]
  lib.sr_blit_transform.argtypes = [
    sp, sp, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_uint32,
  ]
  lib.sr_draw_nv12_crop.argtypes = [
    sp, ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
  ]
  lib.sr_drm_init.argtypes = []
  lib.sr_drm_init.restype = ctypes.c_int
  lib.sr_drm_back_buffer.argtypes = [ctypes.POINTER(ctypes.c_int)]
  lib.sr_drm_back_buffer.restype = ctypes.POINTER(ctypes.c_uint8)
  lib.sr_drm_present.argtypes = [sp]
  lib.sr_drm_present.restype = ctypes.c_int
  lib.sr_drm_camera_begin_frame.argtypes = []
  lib.sr_drm_set_camera.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
  ]
  lib.sr_drm_set_camera.restype = ctypes.c_int
  lib.sr_clear_transparent.argtypes = [
    sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
  ]
  lib.sr_drm_close.argtypes = []
  for name in (
    "sr_clear", "sr_set_clip", "sr_reset_clip", "sr_rect", "sr_gradient_v", "sr_gradient_4",
    "sr_rounded_rect", "sr_circle", "sr_circle_gradient", "sr_ring_arc", "sr_line",
    "sr_triangle", "sr_ribbon", "sr_blit_scaled", "sr_blit_opaque", "sr_blit_many",
    "sr_blit_transform", "sr_draw_nv12_crop", "sr_drm_camera_begin_frame",
    "sr_clear_transparent", "sr_drm_close",
  ):
    getattr(lib, name).restype = None
  return lib


@dataclass
class _TextureData:
  pixels: np.ndarray
  surface: _Surface
  filter_mode: int = 0


@dataclass
class _Nv12Data:
  front: _TextureData
  back: _TextureData
  source: np.ndarray
  settings: tuple[object, ...]
  future: Future | None = None
  has_front: bool = False


class _State:
  def __init__(self):
    self.lib = _load_native()
    self.width = 0
    self.height = 0
    self.framebuffer: np.ndarray | None = None
    self.drm_buffer_owner = None
    self.surface = _Surface()
    self.ready = False
    self.start = time.monotonic()
    self.last_frame = self.start
    self.frame_time = 1 / 60
    self.next_texture_id = 1
    self.textures: dict[int, _TextureData] = {}
    self.scaled_textures: OrderedDict[tuple[object, ...], _TextureData] = OrderedDict()
    self.render_targets: dict[int, _TextureData] = {}
    self.surface_stack: list[_Surface] = []
    self.nv12_cache: dict[int, _Nv12Data] = {}
    self.nv12_executor: ThreadPoolExecutor | None = None
    self.image_premultiplied: set[int] = set()
    self.font_storage: dict[int, tuple[object, object]] = {}
    self.font_maps: dict[int, dict[int, int]] = {}
    self.gui_font = None
    self.gui_styles: dict[tuple[int, int], int] = {}
    self.drm = False
    self.touch_fd = -1
    self.touch_slot = 0
    self.touch_xy = [[-1.0, -1.0] for _ in range(10)]
    self.touch_down = [False] * 10
    self.touch_previous = [False] * 10
    self.touch_physical = [False] * 10
    self.touch_reset = [False] * 10
    self.touch_canonical_zero = False
    self.mouse_scale = (1.0, 1.0)
    self.transform = (1.0, 1.0, 0.0, 0.0)
    self.transform_stack: list[tuple[float, float, float, float]] = []

  @staticmethod
  def make_surface(pixels: np.ndarray) -> _Surface:
    return _Surface(pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    pixels.shape[1], pixels.shape[0], pixels.strides[0],
                    0, 0, pixels.shape[1], pixels.shape[0])


state = _State()


def _camera_worker_init() -> None:
  if not _TICI:
    return
  try:
    # ui.py runs FIFO on core 5. A worker created from that thread inherits
    # both settings, which would serialize conversion with rasterization.
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    os.sched_setaffinity(0, {4})
  except OSError:
    pass


def _pack(color) -> int:
  if hasattr(color, "a"):
    r, g, b, a = color.r, color.g, color.b, color.a
  else:
    r, g, b, a = color
  # Native RGBA byte order, which is DRM ABGR8888 on little-endian CPUs.
  return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r)


def _rect(rect) -> tuple[int, int, int, int]:
  return round(rect.x), round(rect.y), round(rect.width), round(rect.height)


def _xy(point) -> tuple[float, float]:
  return (float(point.x), float(point.y)) if hasattr(point, "x") else (float(point[0]), float(point[1]))


def _transform_xy(x: float, y: float) -> tuple[float, float]:
  sx, sy, tx, ty = state.transform
  return x * sx + tx, y * sy + ty


def _transform_rect(x: float, y: float, width: float, height: float) -> tuple[float, float, float, float]:
  px, py = _transform_xy(x, y)
  return px, py, width * state.transform[0], height * state.transform[1]


def _activate_drm_back_buffer() -> bool:
  if not state.drm:
    return False
  stride = ctypes.c_int()
  pixels = state.lib.sr_drm_back_buffer(ctypes.byref(stride))
  if not pixels:
    return False
  size = stride.value * state.height
  owner = (ctypes.c_uint8 * size).from_address(ctypes.addressof(pixels.contents))
  state.drm_buffer_owner = owner
  state.framebuffer = np.ndarray((state.height, state.width, 4), dtype=np.uint8,
                                 buffer=owner, strides=(stride.value, 4, 1))
  state.surface = state.make_surface(state.framebuffer)
  return True


def init_window(width: int, height: int, title: str) -> None:
  del title
  if state.ready:
    close_window()
  state.width, state.height = width, height
  state.transform = (1.0, 1.0, 0.0, 0.0)
  state.transform_stack.clear()
  state.start = state.last_frame = time.monotonic()
  state.frame_time = 1 / 60
  state.touch_slot = 0
  state.touch_xy = [[-1.0, -1.0] for _ in state.touch_xy]
  state.touch_down = [False] * len(state.touch_down)
  state.touch_previous = [False] * len(state.touch_previous)
  state.touch_physical = [False] * len(state.touch_physical)
  state.touch_reset = [False] * len(state.touch_reset)
  state.touch_canonical_zero = False
  state.mouse_scale = (1.0, 1.0)
  if _TICI and os.getenv("CPU_OFFSCREEN") != "1":
    if state.lib.sr_drm_init() != 0:
      raise RuntimeError("CPU backend failed to initialize DRM/KMS")
    state.drm = True
  if not _activate_drm_back_buffer():
    state.framebuffer = np.zeros((height, width, 4), dtype=np.uint8)
    state.surface = state.make_surface(state.framebuffer)
  if _TICI:
    try:
      state.touch_fd = os.open("/dev/input/event2", os.O_RDONLY | os.O_NONBLOCK)
      origin_path = pathlib.Path("/sys/devices/platform/vendor/vendor:gpio-som-id/som_id")
      state.touch_canonical_zero = origin_path.read_text().strip() == "1"
    except OSError:
      state.touch_fd = -1
  state.ready = True


def close_window() -> None:
  if state.drm:
    state.lib.sr_drm_close()
    state.drm = False
  if state.touch_fd >= 0:
    os.close(state.touch_fd)
    state.touch_fd = -1
  if state.nv12_executor is not None:
    state.nv12_executor.shutdown(wait=True, cancel_futures=True)
    state.nv12_executor = None
  state.nv12_cache.clear()
  state.scaled_textures.clear()
  state.render_targets.clear()
  state.textures.clear()
  state.surface_stack.clear()
  state.image_premultiplied.clear()
  state.font_storage.clear()
  state.font_maps.clear()
  state.gui_font = None
  state.surface = _Surface()
  state.framebuffer = None
  state.drm_buffer_owner = None
  state.ready = False


def is_window_ready() -> bool:
  return state.ready


def window_should_close() -> bool:
  return False


def get_monitor_width(monitor: int) -> int:
  del monitor
  return 536


def get_monitor_height(monitor: int) -> int:
  del monitor
  return 240


def begin_drawing() -> None:
  now = time.monotonic()
  state.frame_time = now - state.last_frame
  state.last_frame = now
  if state.drm:
    _activate_drm_back_buffer()
    state.lib.sr_drm_camera_begin_frame()


def end_drawing() -> None:
  if state.drm:
    result = state.lib.sr_drm_present(ctypes.byref(state.surface))
    if result != 0:
      raise RuntimeError(f"CPU backend failed to present DRM framebuffer: errno {-result}")


def clear_background(color) -> None:
  state.lib.sr_clear(ctypes.byref(state.surface), _pack(color))


def draw_rectangle(x: int, y: int, width: int, height: int, color) -> None:
  tx, ty, tw, th = _transform_rect(x, y, width, height)
  state.lib.sr_rect(ctypes.byref(state.surface), round(tx), round(ty), round(tw), round(th), _pack(color))


def draw_rectangle_rec(rect, color) -> None:
  draw_rectangle(*_rect(rect), color)


def draw_rectangle_gradient_v(x: int, y: int, width: int, height: int, top, bottom) -> None:
  tx, ty, tw, th = _transform_rect(x, y, width, height)
  state.lib.sr_gradient_v(ctypes.byref(state.surface), round(tx), round(ty), round(tw), round(th),
                          _pack(top), _pack(bottom))


def draw_rectangle_gradient_h(x: int, y: int, width: int, height: int, left, right) -> None:
  tx, ty, tw, th = (round(value) for value in _transform_rect(x, y, width, height))
  state.lib.sr_gradient_4(
    ctypes.byref(state.surface), tx, ty, tw, th,
    _pack(left), _pack(left), _pack(right), _pack(right),
  )


def draw_rectangle_gradient_ex(rect, top_left, bottom_left, top_right, bottom_right) -> None:
  x, y, w, h = _transform_rect(rect.x, rect.y, rect.width, rect.height)
  x, y, w, h = round(x), round(y), round(w), round(h)
  state.lib.sr_gradient_4(
    ctypes.byref(state.surface), x, y, w, h,
    _pack(top_left), _pack(bottom_left), _pack(top_right), _pack(bottom_right),
  )


def draw_circle(x: int, y: int, radius: float, color) -> None:
  tx, ty = _transform_xy(x, y)
  radius *= (abs(state.transform[0]) + abs(state.transform[1])) * .5
  state.lib.sr_circle(ctypes.byref(state.surface), round(tx), round(ty), round(radius), _pack(color))


def draw_circle_gradient(center, radius: float, inner, outer) -> None:
  x, y = _xy(center)
  x, y = _transform_xy(x, y)
  radius *= (abs(state.transform[0]) + abs(state.transform[1])) * .5
  state.lib.sr_circle_gradient(
    ctypes.byref(state.surface), round(x), round(y), round(radius), _pack(inner), _pack(outer),
  )


def draw_ring(center, inner_radius: float, outer_radius: float, start_angle: float,
              end_angle: float, segments: int, color) -> None:
  del segments
  cx, cy = _xy(center)
  cx, cy = _transform_xy(cx, cy)
  scale = (abs(state.transform[0]) + abs(state.transform[1])) * .5
  state.lib.sr_ring_arc(
    ctypes.byref(state.surface), round(cx), round(cy),
    round(inner_radius * scale), round(outer_radius * scale),
    start_angle, end_angle, _pack(color),
  )


def draw_line(start_x: int, start_y: int, end_x: int, end_y: int, color) -> None:
  x0, y0 = _transform_xy(start_x, start_y)
  x1, y1 = _transform_xy(end_x, end_y)
  state.lib.sr_line(ctypes.byref(state.surface), x0, y0, x1, y1, 1.0, _pack(color))


def draw_line_ex(start, end, thick: float, color) -> None:
  x0, y0 = _xy(start)
  x1, y1 = _xy(end)
  x0, y0 = _transform_xy(x0, y0)
  x1, y1 = _transform_xy(x1, y1)
  scale = (abs(state.transform[0]) + abs(state.transform[1])) * .5
  state.lib.sr_line(ctypes.byref(state.surface), x0, y0, x1, y1, thick * scale, _pack(color))


def _point(value) -> _Point:
  x, y = _xy(value)
  x, y = _transform_xy(x, y)
  return _Point(x, y)


def draw_triangle_fan(points, point_count: int, color) -> None:
  for idx in range(1, max(1, point_count - 1)):
    state.lib.sr_triangle(ctypes.byref(state.surface), _point(points[0]), _point(points[idx]),
                          _point(points[idx + 1]), _pack(color))


def draw_polygon_cpu(origin_rect, points: np.ndarray, color, gradient) -> None:
  points = np.ascontiguousarray(points, dtype=np.float32)
  sx, sy, tx, ty = state.transform
  point_ptr = points.ctypes.data_as(ctypes.POINTER(_Point))
  if color is not None:
    state.lib.sr_ribbon(
      ctypes.byref(state.surface), point_ptr, len(points), sx, sy, tx, ty, _pack(color),
      0, 0, 0, 0, None, None, 0,
    )
    return

  start_x, start_y = _transform_xy(
    origin_rect.x + gradient.start[0] * origin_rect.width,
    origin_rect.y + gradient.start[1] * origin_rect.height,
  )
  end_x, end_y = _transform_xy(
    origin_rect.x + gradient.end[0] * origin_rect.width,
    origin_rect.y + gradient.end[1] * origin_rect.height,
  )
  colors = np.ascontiguousarray([_pack(value) for value in gradient.colors], dtype=np.uint32)
  stops = np.ascontiguousarray(gradient.stops, dtype=np.float32)
  state.lib.sr_ribbon(
    ctypes.byref(state.surface), point_ptr, len(points), sx, sy, tx, ty, 0,
    start_x, start_y, end_x, end_y,
    colors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    stops.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(colors),
  )


def draw_rectangle_rounded(rect, roundness: float, segments: int, color) -> None:
  del segments
  x, y, w, h = _transform_rect(rect.x, rect.y, rect.width, rect.height)
  radius = max(0.0, min(w, h) * roundness * .5)
  state.lib.sr_rounded_rect(
    ctypes.byref(state.surface), round(x), round(y), round(w), round(h), radius, 0, _pack(color),
  )


def draw_rectangle_lines_ex(rect, line_thickness: float, color) -> None:
  x, y, w, h = _rect(rect)
  thickness = max(1, round(line_thickness))
  draw_rectangle(x, y, w, thickness, color)
  draw_rectangle(x, y + h - thickness, w, thickness, color)
  draw_rectangle(x, y + thickness, thickness, h - 2 * thickness, color)
  draw_rectangle(x + w - thickness, y + thickness, thickness, h - 2 * thickness, color)


def draw_rectangle_lines(x: int, y: int, width: int, height: int, color) -> None:
  draw_rectangle_lines_ex(_pyray.Rectangle(x, y, width, height), 1, color)


def draw_rectangle_rounded_lines_ex(rect, roundness: float, segments: int, line_thickness: float, color) -> None:
  del segments
  x, y, w, h = _transform_rect(rect.x, rect.y, rect.width, rect.height)
  x, y, w, h = round(x), round(y), round(w), round(h)
  radius = max(0, round(min(w, h) * roundness * .5))
  scale = (abs(state.transform[0]) + abs(state.transform[1])) * .5
  thickness = max(1, round(line_thickness * scale))
  packed = _pack(color)
  surface = ctypes.byref(state.surface)
  if radius <= 0:
    draw_rectangle_lines_ex(_pyray.Rectangle(rect.x, rect.y, rect.width, rect.height), line_thickness, color)
    return
  state.lib.sr_rect(surface, x + radius, y, max(0, w - 2 * radius), thickness, packed)
  state.lib.sr_rect(surface, x + radius, y + h - thickness, max(0, w - 2 * radius), thickness, packed)
  state.lib.sr_rect(surface, x, y + radius, thickness, max(0, h - 2 * radius), packed)
  state.lib.sr_rect(surface, x + w - thickness, y + radius, thickness, max(0, h - 2 * radius), packed)
  inner = max(0, radius - thickness)
  for cx, cy, start, end in (
    (x + radius, y + radius, 180, 270),
    (x + w - radius, y + radius, 270, 360),
    (x + w - radius, y + h - radius, 0, 90),
    (x + radius, y + h - radius, 90, 180),
  ):
    state.lib.sr_ring_arc(surface, cx, cy, inner, radius, start, end, packed)


def _rgba_image(image) -> np.ndarray:
  width, height = int(image.width), int(image.height)
  pixel_format = int(image.format)
  rgba = np.zeros((height, width, 4), dtype=np.uint8)
  if image.data == ffi.NULL:
    return rgba
  if pixel_format == int(_pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8):
    return np.frombuffer(ffi.buffer(image.data, width * height * 4), dtype=np.uint8).reshape(height, width, 4).copy()
  if pixel_format == int(_pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8):
    rgba[:, :, :3] = np.frombuffer(
      ffi.buffer(image.data, width * height * 3), dtype=np.uint8,
    ).reshape(height, width, 3)
    rgba[:, :, 3] = 255
    return rgba
  if pixel_format == int(_pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE):
    gray = np.frombuffer(ffi.buffer(image.data, width * height), dtype=np.uint8).reshape(height, width)
    rgba[:, :, :3] = gray[:, :, None]
    rgba[:, :, 3] = 255
    return rgba
  if pixel_format == int(_pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA):
    gray_alpha = np.frombuffer(
      ffi.buffer(image.data, width * height * 2), dtype=np.uint8,
    ).reshape(height, width, 2)
    rgba[:, :, :3] = gray_alpha[:, :, 0, None]
    rgba[:, :, 3] = gray_alpha[:, :, 1]
    return rgba
  raise NotImplementedError(f"CPU image format {pixel_format}")


def load_image(path: str):
  return _pyray.load_image(path)


def gen_image_color(width: int, height: int, color):
  return _pyray.gen_image_color(width, height, color)


def unload_image(image) -> None:
  pointer = int(ffi.cast("uintptr_t", image.data))
  state.image_premultiplied.discard(pointer)
  _pyray.unload_image(image)


def image_resize(image, width: int, height: int) -> None:
  pointer = int(ffi.cast("uintptr_t", image.data))
  premultiplied = pointer in state.image_premultiplied
  state.image_premultiplied.discard(pointer)
  _pyray.image_resize(image, width, height)
  if premultiplied:
    state.image_premultiplied.add(int(ffi.cast("uintptr_t", image.data)))


def image_flip_horizontal(image) -> None:
  _pyray.image_flip_horizontal(image)


def image_alpha_premultiply(image) -> None:
  _pyray.image_alpha_premultiply(image)
  state.image_premultiplied.add(int(ffi.cast("uintptr_t", image.data)))


def load_texture_from_image(image):
  rgba = _rgba_image(image)
  if int(ffi.cast("uintptr_t", image.data)) not in state.image_premultiplied:
    wide = rgba.astype(np.uint16)
    wide[:, :, :3] = (wide[:, :, :3] * wide[:, :, 3:4] + 127) // 255
    rgba = wide.astype(np.uint8)
  texture_id = state.next_texture_id
  state.next_texture_id += 1
  state.textures[texture_id] = _TextureData(rgba, state.make_surface(rgba))
  return _pyray.Texture(texture_id, int(image.width), int(image.height), 1,
                     _pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)


def _invalidate_scaled_texture(texture_id: int) -> None:
  for key in [key for key in state.scaled_textures if key[0] == texture_id]:
    del state.scaled_textures[key]


def unload_texture(texture) -> None:
  texture_id = int(texture.id)
  state.textures.pop(texture_id, None)
  _invalidate_scaled_texture(texture_id)


def update_texture(texture, pixels) -> None:
  data = state.textures.get(int(texture.id))
  if data is None:
    return
  width, height = int(texture.width), int(texture.height)
  pixel_format = int(texture.format)
  if pixel_format == int(_pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8):
    raw = np.frombuffer(ffi.buffer(pixels, width * height * 4), dtype=np.uint8).reshape(height, width, 4)
    rgba = raw.astype(np.uint16)
    rgba[:, :, :3] = (rgba[:, :, :3] * rgba[:, :, 3:4] + 127) // 255
    data.pixels[:] = rgba.astype(np.uint8)
  elif pixel_format == int(_pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE):
    gray = np.frombuffer(ffi.buffer(pixels, width * height), dtype=np.uint8).reshape(height, width)
    data.pixels[:, :, :3] = gray[:, :, None]
    data.pixels[:, :, 3] = 255
  elif pixel_format == int(_pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA):
    raw = np.frombuffer(ffi.buffer(pixels, width * height * 2), dtype=np.uint8).reshape(height, width, 2)
    alpha = raw[:, :, 1].astype(np.uint16)
    gray = ((raw[:, :, 0].astype(np.uint16) * alpha + 127) // 255).astype(np.uint8)
    data.pixels[:, :, :3] = gray[:, :, None]
    data.pixels[:, :, 3] = alpha.astype(np.uint8)
  else:
    raise NotImplementedError(f"CPU update_texture format {pixel_format}")
  _invalidate_scaled_texture(int(texture.id))


def load_render_texture(width: int, height: int):
  pixels = np.zeros((height, width, 4), dtype=np.uint8)
  texture_id = state.next_texture_id
  state.next_texture_id += 1
  data = _TextureData(pixels, state.make_surface(pixels))
  state.textures[texture_id] = data
  state.render_targets[texture_id] = data
  texture = _pyray.Texture(texture_id, width, height, 1,
                        _pyray.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
  return _pyray.RenderTexture(texture_id, texture, _pyray.Texture())


def unload_render_texture(target) -> None:
  texture_id = int(target.texture.id)
  state.render_targets.pop(texture_id, None)
  state.textures.pop(texture_id, None)
  _invalidate_scaled_texture(texture_id)


def begin_texture_mode(target) -> None:
  data = state.render_targets.get(int(target.texture.id))
  if data is None:
    raise RuntimeError("unknown CPU render texture")
  state.surface_stack.append(state.surface)
  state.surface = data.surface


def end_texture_mode() -> None:
  if not state.surface_stack:
    raise RuntimeError("end_texture_mode without begin_texture_mode")
  state.surface = state.surface_stack.pop()


def load_image_from_texture(texture):
  data = state.textures.get(int(texture.id))
  if data is None:
    return _pyray.Image()
  pixels = data.pixels[::-1] if int(texture.id) in state.render_targets else data.pixels
  image = _pyray.gen_image_color(pixels.shape[1], pixels.shape[0], _pyray.BLANK)
  output = np.frombuffer(ffi.buffer(image.data, pixels.size), dtype=np.uint8).reshape(pixels.shape)
  output[:] = pixels
  return image


def set_texture_filter(texture, filter_mode) -> None:
  data = state.textures.get(int(texture.id))
  if data is not None:
    data.filter_mode = int(filter_mode)


def set_texture_wrap(texture, wrap_mode) -> None:
  del texture, wrap_mode


def gen_texture_mipmaps(texture) -> None:
  del texture


def _fields(line: str) -> dict[str, str]:
  return {key: value.strip('"') for key, value in re.findall(r'(\w+)=("[^"]*"|\S+)', line)}


def load_font(path: str):
  font_path = pathlib.Path(path)
  lines = font_path.read_text().splitlines()
  common = _fields(next(line for line in lines if line.startswith("common ")))
  page = _fields(next(line for line in lines if line.startswith("page ")))
  entries = [_fields(line) for line in lines if line.startswith("char ")]
  texture_image = load_image(str(font_path.with_name(page["file"])))
  texture = load_texture_from_image(texture_image)
  unload_image(texture_image)

  recs = ffi.new("Rectangle[]", len(entries))
  glyphs = ffi.new("GlyphInfo[]", len(entries))
  glyph_map: dict[int, int] = {}
  for idx, entry in enumerate(entries):
    value = int(entry["id"])
    recs[idx] = _pyray.Rectangle(float(entry["x"]), float(entry["y"]),
                              float(entry["width"]), float(entry["height"]))
    glyphs[idx].value = value
    glyphs[idx].offsetX = int(entry["xoffset"])
    glyphs[idx].offsetY = int(entry["yoffset"])
    glyphs[idx].advanceX = int(entry["xadvance"])
    glyph_map[value] = idx
  font = _pyray.Font(int(common["lineHeight"]), len(entries), 0, texture, recs, glyphs)
  key = int(ffi.cast("uintptr_t", glyphs))
  state.font_storage[key] = (recs, glyphs)
  state.font_maps[key] = glyph_map
  return font


def unload_font(font) -> None:
  unload_texture(font.texture)
  key = int(ffi.cast("uintptr_t", font.glyphs))
  state.font_storage.pop(key, None)
  state.font_maps.pop(key, None)


def measure_text_ex(font, text: str, font_size: float, spacing: float):
  glyph_map = state.font_maps.get(int(ffi.cast("uintptr_t", font.glyphs)), {})
  scale = font_size / font.baseSize
  line_width = max_width = 0.0
  lines = 1
  for char in text:
    if char == "\n":
      max_width = max(max_width, line_width)
      line_width = 0
      lines += 1
      continue
    idx = glyph_map.get(ord(char), glyph_map.get(ord("?"), 0))
    line_width += font.glyphs[idx].advanceX * scale + spacing
  max_width = max(max_width, line_width)
  return _pyray.Vector2(max_width, lines * font_size)


def draw_text_ex(font, text: str, position, font_size: float, spacing: float, tint) -> None:
  glyph_map = state.font_maps.get(int(ffi.cast("uintptr_t", font.glyphs)), {})
  atlas_id = int(font.texture.id)
  atlas = state.textures.get(atlas_id)
  if atlas is None:
    return
  scale = font_size / font.baseSize
  x, y = _xy(position)
  line_x = x
  items = []
  for char in text:
    if char == "\n":
      x = line_x
      y += font_size
      continue
    idx = glyph_map.get(ord(char), glyph_map.get(ord("?"), 0))
    glyph = font.glyphs[idx]
    rec = font.recs[idx]
    dx, dy, dw, dh = _transform_rect(
      x + glyph.offsetX * scale, y + glyph.offsetY * scale,
      rec.width * scale, rec.height * scale,
    )
    width, height = max(1, round(abs(dw))), max(1, round(abs(dh)))
    glyph_data = atlas
    source_x, source_y = float(rec.x), float(rec.y)
    source_width, source_height = float(rec.width), float(rec.height)
    if (atlas.filter_mode != int(_pyray.TextureFilter.TEXTURE_FILTER_POINT) and
        (abs(source_width - width) > .001 or abs(source_height - height) > .001)):
      glyph_data = _scaled_texture(atlas_id, atlas, rec, width, height)
      source_x = source_y = 0
      source_width, source_height = width, height
    items.append(_BlitItem(
      ctypes.pointer(glyph_data.surface),
      source_x, source_y, source_width, source_height,
      round(dx), round(dy), round(dw), round(dh),
    ))
    x += glyph.advanceX * scale + spacing
  if items:
    item_array = (_BlitItem * len(items))(*items)
    state.lib.sr_blit_many(ctypes.byref(state.surface), item_array, len(items), _pack(tint))


def gui_set_font(font) -> None:
  state.gui_font = font


def gui_set_style(control, prop, value: int) -> None:
  state.gui_styles[(int(control), int(prop))] = int(value)


def gui_get_style(control, prop) -> int:
  key = (int(control), int(prop))
  if key in state.gui_styles:
    return state.gui_styles[key]
  defaults = {
    int(_pyray.GuiDefaultProperty.TEXT_SIZE): 10,
    int(_pyray.GuiDefaultProperty.TEXT_LINE_SPACING): 15,
    int(_pyray.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL): int(_pyray.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE),
    int(_pyray.GuiDefaultProperty.TEXT_WRAP_MODE): int(_pyray.GuiTextWrapMode.TEXT_WRAP_NONE),
    int(_pyray.GuiControlProperty.TEXT_ALIGNMENT): int(_pyray.GuiTextAlignment.TEXT_ALIGN_LEFT),
    int(_pyray.GuiControlProperty.TEXT_COLOR_NORMAL): _pyray.color_to_int(_pyray.DARKGRAY),
  }
  return defaults.get(int(prop), 0)


def _gui_color(value: int):
  value &= 0xFFFFFFFF
  return _pyray.Color((value >> 24) & 255, (value >> 16) & 255, (value >> 8) & 255, value & 255)


def gui_label(rect, text: str) -> None:
  if state.gui_font is None:
    return
  control = _pyray.GuiControl.DEFAULT
  font_size = gui_get_style(control, _pyray.GuiDefaultProperty.TEXT_SIZE)
  line_spacing = gui_get_style(control, _pyray.GuiDefaultProperty.TEXT_LINE_SPACING)
  alignment = gui_get_style(control, _pyray.GuiControlProperty.TEXT_ALIGNMENT)
  vertical = gui_get_style(control, _pyray.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL)
  wrap = gui_get_style(control, _pyray.GuiDefaultProperty.TEXT_WRAP_MODE)
  lines: list[str] = []
  for paragraph in text.splitlines() or [""]:
    if wrap == int(_pyray.GuiTextWrapMode.TEXT_WRAP_NONE):
      lines.append(paragraph)
      continue
    current = ""
    for word in paragraph.split():
      candidate = word if not current else f"{current} {word}"
      if current and measure_text_ex(state.gui_font, candidate, font_size, 0).x > rect.width:
        lines.append(current)
        current = word
      else:
        current = candidate
    lines.append(current)
  total_height = font_size + max(0, len(lines) - 1) * line_spacing
  y = rect.y
  if vertical == int(_pyray.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE):
    y += (rect.height - total_height) / 2
  elif vertical == int(_pyray.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM):
    y += rect.height - total_height
  tint = _gui_color(gui_get_style(control, _pyray.GuiControlProperty.TEXT_COLOR_NORMAL))
  for line in lines:
    width = measure_text_ex(state.gui_font, line, font_size, 0).x
    x = rect.x
    if alignment == int(_pyray.GuiTextAlignment.TEXT_ALIGN_CENTER):
      x += (rect.width - width) / 2
    elif alignment == int(_pyray.GuiTextAlignment.TEXT_ALIGN_RIGHT):
      x += rect.width - width
    draw_text_ex(state.gui_font, line, _pyray.Vector2(x, y), font_size, 0, tint)
    y += line_spacing


def _scaled_texture(texture_id: int, data: _TextureData, source, width: int, height: int) -> _TextureData:
  key = (
    texture_id, float(source.x), float(source.y), float(source.width), float(source.height), width, height,
  )
  cached = state.scaled_textures.get(key)
  if cached is not None:
    state.scaled_textures.move_to_end(key)
    return cached
  pixels = np.zeros((height, width, 4), dtype=np.uint8)
  cached = _TextureData(pixels, state.make_surface(pixels))
  state.lib.sr_blit_scaled(
    ctypes.byref(cached.surface), ctypes.byref(data.surface),
    float(source.x), float(source.y), float(source.width), float(source.height),
    0, 0, width, height, 0xFFFFFFFF,
  )
  state.scaled_textures[key] = cached
  if len(state.scaled_textures) > 512:
    state.scaled_textures.popitem(last=False)
  return cached


def draw_texture_pro(texture, source, dest, origin, rotation: float, tint) -> None:
  texture_id = int(texture.id)
  data = state.textures.get(texture_id)
  if data is None:
    return
  # CPU render targets use top-left storage. Raylib exposes render textures as
  # bottom-left textures, and callers conventionally request a negative source
  # height to restore screen orientation. Cancel that API-level flip here.
  if texture_id in state.render_targets and source.height < 0:
    source = _pyray.Rectangle(source.x, source.y, source.width, -source.height)
  ox, oy = _xy(origin)
  dx, dy, dw, dh = _transform_rect(dest.x, dest.y, dest.width, dest.height)
  ox *= state.transform[0]
  oy *= state.transform[1]
  width, height = max(1, round(abs(dw))), max(1, round(abs(dh)))
  if (data.filter_mode != int(_pyray.TextureFilter.TEXTURE_FILTER_POINT) and
      texture_id not in state.render_targets and
      (abs(abs(float(source.width)) - width) > .001 or abs(abs(float(source.height)) - height) > .001)):
    data = _scaled_texture(texture_id, data, source, width, height)
    source = _pyray.Rectangle(0, 0, width, height)
  if abs(rotation) > 1e-4:
    state.lib.sr_blit_transform(
      ctypes.byref(state.surface), ctypes.byref(data.surface),
      float(source.x), float(source.y), float(source.width), float(source.height),
      dx, dy, dw, dh,
      ox, oy, rotation, _pack(tint),
    )
    return
  state.lib.sr_blit_scaled(
    ctypes.byref(state.surface), ctypes.byref(data.surface),
    float(source.x), float(source.y), float(source.width), float(source.height),
    round(dx - ox), round(dy - oy), round(dw), round(dh), _pack(tint),
  )


def draw_texture_ex(texture, position, rotation: float, scale: float, tint) -> None:
  px, py = _xy(position)
  source = _pyray.Rectangle(0, 0, texture.width, texture.height)
  dest = _pyray.Rectangle(px, py, texture.width * scale, texture.height * scale)
  draw_texture_pro(texture, source, dest, _pyray.Vector2(0, 0), rotation, tint)


def draw_texture_v(texture, position, tint) -> None:
  draw_texture_ex(texture, position, 0, 1, tint)


def _convert_nv12(entry: _Nv12Data, frame_width: int, frame_height: int, stride: int, uv_offset: int,
                  source_rect: tuple[float, float, float, float], flip_x: bool,
                  engaged: bool, enhance_driver: bool) -> None:
  source_x, source_y, source_width, source_height = source_rect
  state.lib.sr_draw_nv12_crop(
    ctypes.byref(entry.back.surface), entry.source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    frame_width, frame_height, stride, uv_offset, source_x, source_y, source_width, source_height,
    0, 0, entry.back.surface.width, entry.back.surface.height,
    int(flip_x), int(engaged), int(enhance_driver),
  )


def draw_nv12(frame, dest, engaged: bool, enhance_driver: bool, flip_x: bool = False,
              cache_key: int = 0, needs_update: bool = True) -> None:
  dx, dy, dw, dh = _transform_rect(dest.x, dest.y, dest.width, dest.height)
  full_x, full_y = round(dx), round(dy)
  full_width, full_height = max(1, round(dw)), max(1, round(dh))
  visible_x = max(full_x, state.surface.clip_x0)
  visible_y = max(full_y, state.surface.clip_y0)
  visible_x1 = min(full_x + full_width, state.surface.clip_x1)
  visible_y1 = min(full_y + full_height, state.surface.clip_y1)
  width, height = visible_x1 - visible_x, visible_y1 - visible_y
  if width <= 0 or height <= 0:
    return
  source_x = (visible_x - full_x) * int(frame.width) / full_width
  source_y = (visible_y - full_y) * int(frame.height) / full_height
  source_width = width * int(frame.width) / full_width
  source_height = height * int(frame.height) / full_height
  settings = (width, height, visible_x, visible_y, source_x, source_y, source_width, source_height,
              int(frame.width), int(frame.height), int(frame.stride), int(frame.uv_offset),
              int(flip_x), int(engaged), int(enhance_driver))
  source_size = int(frame.uv_offset) + int(frame.stride) * ((int(frame.height) + 1) // 2)
  if state.drm:
    result = state.lib.sr_drm_set_camera(
      int(frame.fd), int(frame.width), int(frame.height), int(frame.stride), int(frame.uv_offset),
      source_x, source_y, source_width, source_height,
      visible_x, visible_y, width, height, int(flip_x), int(engaged), int(enhance_driver),
    )
    if result == 0:
      state.lib.sr_clear_transparent(ctypes.byref(state.surface), visible_x, visible_y, width, height)
      return
  entry = state.nv12_cache.get(cache_key)
  if entry is None or entry.settings != settings:
    front_pixels = np.empty((height, width, 4), dtype=np.uint8)
    back_pixels = np.empty((height, width, 4), dtype=np.uint8)
    entry = _Nv12Data(
      _TextureData(front_pixels, state.make_surface(front_pixels)),
      _TextureData(back_pixels, state.make_surface(back_pixels)),
      np.empty(source_size, dtype=np.uint8), settings,
    )
    state.nv12_cache[cache_key] = entry
    needs_update = True

  if entry.future is not None and entry.future.done():
    entry.future.result()
    entry.front, entry.back = entry.back, entry.front
    entry.has_front = True
    entry.future = None

  if needs_update and entry.future is None:
    np.copyto(entry.source, np.frombuffer(frame.data, dtype=np.uint8, count=source_size))
    if state.nv12_executor is None:
      state.nv12_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cpu-camera", initializer=_camera_worker_init,
      )
    entry.future = state.nv12_executor.submit(
      _convert_nv12, entry, int(frame.width), int(frame.height), int(frame.stride), int(frame.uv_offset),
      (source_x, source_y, source_width, source_height), flip_x, engaged, enhance_driver,
    )
  if entry.has_front:
    state.lib.sr_blit_opaque(ctypes.byref(state.surface), ctypes.byref(entry.front.surface), visible_x, visible_y)


def set_target_fps(fps: int) -> None:
  del fps


def get_frame_time() -> float:
  return state.frame_time


def get_time() -> float:
  return time.monotonic() - state.start


def get_fps() -> int:
  return round(1 / state.frame_time) if state.frame_time > 0 else 0


def draw_fps(pos_x: int, pos_y: int) -> None:
  if state.gui_font is not None:
    draw_text_ex(state.gui_font, f"{get_fps()} FPS", _pyray.Vector2(pos_x, pos_y), 20, 1, _pyray.LIME)


def set_config_flags(flags) -> None:
  del flags


def set_trace_log_callback(callback) -> None:
  _pyray.set_trace_log_callback(callback)


def set_trace_log_level(level) -> None:
  _pyray.set_trace_log_level(level)


def poll_input_events() -> None:
  state.touch_previous[:] = state.touch_down
  for index, reset in enumerate(state.touch_reset):
    if reset:
      state.touch_down[index] = False
      state.touch_reset[index] = False
  if state.touch_fd < 0:
    return
  event_struct = struct.Struct("llHHi")
  while True:
    try:
      raw = os.read(state.touch_fd, event_struct.size * 32)
    except BlockingIOError:
      break
    if not raw:
      break
    for offset in range(0, len(raw) - event_struct.size + 1, event_struct.size):
      _, _, event_type, code, value = event_struct.unpack_from(raw, offset)
      if event_type == 0 and code == 0:  # EV_SYN / SYN_REPORT
        for index, physical in enumerate(state.touch_physical):
          if physical:
            state.touch_down[index] = True
          elif state.touch_down[index] and not state.touch_previous[index]:
            state.touch_reset[index] = True
          else:
            state.touch_down[index] = False
        continue
      if event_type != 3:  # EV_ABS
        continue
      if code == 0x2f:  # ABS_MT_SLOT
        state.touch_slot = max(0, min(len(state.touch_down) - 1, value))
      elif code == 0x39:  # ABS_MT_TRACKING_ID
        state.touch_physical[state.touch_slot] = value != -1
      elif code == 0x35:  # ABS_MT_POSITION_X
        state.touch_xy[state.touch_slot][1] = (
          value if state.touch_canonical_zero else state.height - value
        )
      elif code == 0x36:  # ABS_MT_POSITION_Y
        state.touch_xy[state.touch_slot][0] = (
          state.width - value if state.touch_canonical_zero else value
        )


def begin_scissor_mode(x: int, y: int, width: int, height: int) -> None:
  state.lib.sr_set_clip(ctypes.byref(state.surface), x, y, width, height)


def end_scissor_mode() -> None:
  state.lib.sr_reset_clip(ctypes.byref(state.surface))


def rl_push_matrix() -> None:
  state.transform_stack.append(state.transform)


def rl_pop_matrix() -> None:
  if not state.transform_stack:
    raise RuntimeError("rl_pop_matrix without rl_push_matrix")
  state.transform = state.transform_stack.pop()


def rl_scalef(x: float, y: float, z: float) -> None:
  del z
  sx, sy, tx, ty = state.transform
  state.transform = (sx * x, sy * y, tx, ty)


def rl_translatef(x: float, y: float, z: float) -> None:
  del z
  sx, sy, tx, ty = state.transform
  state.transform = (sx, sy, tx + sx * x, ty + sy * y)


def get_touch_position(index: int):
  if index < 0 or index >= len(state.touch_xy):
    return _pyray.Vector2(-1, -1)
  return _pyray.Vector2(*state.touch_xy[index])


def is_mouse_button_pressed(button) -> bool:
  index = int(button)
  return 0 <= index < len(state.touch_down) and state.touch_down[index] and not state.touch_previous[index]


def is_mouse_button_released(button) -> bool:
  index = int(button)
  return 0 <= index < len(state.touch_down) and not state.touch_down[index] and state.touch_previous[index]


def is_mouse_button_down(button) -> bool:
  index = int(button)
  return 0 <= index < len(state.touch_down) and state.touch_down[index]


def get_mouse_position():
  x, y = state.touch_xy[0]
  return _pyray.Vector2(x * state.mouse_scale[0], y * state.mouse_scale[1])


def set_mouse_scale(scale_x: float, scale_y: float) -> None:
  state.mouse_scale = (scale_x, scale_y)


def get_mouse_wheel_move() -> float:
  return 0.0


def get_key_pressed() -> int:
  return 0


def get_char_pressed() -> int:
  return 0


def is_key_down(key) -> bool:
  del key
  return False


def is_key_pressed(key) -> bool:
  del key
  return False


def is_key_released(key) -> bool:
  del key
  return False


EXPORTED = {
  name: value for name, value in globals().items()
  if not name.startswith("_") and callable(value) and getattr(value, "__module__", None) == __name__
}
