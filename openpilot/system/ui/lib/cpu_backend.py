"""Raylib-compatible CPU backend for GPU-free UI rendering.

It implements the public 2D surface used by the UIs rather than emulating
rlgl. It is loaded only when RAYLIB_BACKEND=cpu. SCons builds the native raster
core for production; a temporary local build keeps standalone development
scripts convenient.
"""

from __future__ import annotations

import ctypes
import colorsys
import io
import os
import pathlib
import platform
import re
import struct
import subprocess
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage
import pyray as gpu

ffi = gpu.ffi
_SOURCE = pathlib.Path(__file__).resolve().parent / "cpu_renderer/renderer.c"


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


def _build_native() -> tuple[ctypes.CDLL, tempfile.TemporaryDirectory | None]:
  built_library = _SOURCE.with_name("libraylib_cpu.so")
  if built_library.exists():
    build_dir = None
    library = built_library
  else:
    build_dir = tempfile.TemporaryDirectory(prefix="mici_cpu_backend_")
    library = pathlib.Path(build_dir.name) / "renderer.so"
    command = ["cc", "-O3", "-shared", "-fPIC", str(_SOURCE), "-lm"]
    if platform.system() == "Linux":
      command += ["-I/usr/include/libdrm", "-ldrm"]
    command += ["-o", str(library)]
    subprocess.run(command, check=True)
  lib = ctypes.CDLL(str(library))
  sp = ctypes.POINTER(_Surface)
  lib.sr_clear.argtypes = [sp, ctypes.c_uint32]
  lib.sr_set_opacity_culling.argtypes = [ctypes.c_int]
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
  lib.sr_ring.argtypes = [sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
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
  lib.sr_burn_in_filter.argtypes = [
    sp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
  ]
  lib.sr_blit_transform.argtypes = [
    sp, sp, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_uint32,
  ]
  lib.sr_draw_nv12.argtypes = [
    sp, ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
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
  lib.sr_drm_last_copy_ms.argtypes = []
  lib.sr_drm_last_copy_ms.restype = ctypes.c_double
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
  return lib, build_dir


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
    self.lib, self.build_dir = _build_native()
    self.width = 0
    self.height = 0
    self.framebuffer: np.ndarray | None = None
    self.drm_buffer_owner = None
    self.surface: _Surface | None = None
    self.ready = False
    self.start = time.monotonic()
    self.last_frame = self.start
    self.frame_time = 1 / 60
    self.target_fps = 60
    self.next_texture_id = 1
    self.textures: dict[int, _TextureData] = {}
    self.scaled_textures: OrderedDict[tuple[object, ...], _TextureData] = OrderedDict()
    self.render_targets: dict[int, _TextureData] = {}
    self.surface_stack: list[_Surface] = []
    self.nv12_cache: dict[int, _Nv12Data] = {}
    self.nv12_executor: ThreadPoolExecutor | None = None
    self.image_buffers: dict[int, object] = {}
    self.image_premultiplied: set[int] = set()
    self.font_storage: list[tuple[object, object, dict[int, int]]] = []
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
    self.profile_enabled = os.getenv("CPU_RENDER_PROFILE") == "1"
    self.opacity_culling = os.getenv("CPU_RENDER_OPACITY_CULLING", "1") != "0"
    self.lib.sr_set_opacity_culling(int(self.opacity_culling))
    self.profile: dict[str, list[float]] = {}
    self.next_shader_id = 1
    self.shader_effects: dict[int, str] = {}
    self.active_shader_effect: str | None = None

  @staticmethod
  def make_surface(pixels: np.ndarray) -> _Surface:
    return _Surface(pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    pixels.shape[1], pixels.shape[0], pixels.strides[0],
                    0, 0, pixels.shape[1], pixels.shape[0])


state = _State()


def _camera_worker_init() -> None:
  if not pathlib.Path("/TICI").exists():
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
  if pathlib.Path("/TICI").exists() and os.getenv("CPU_OFFSCREEN") != "1":
    if state.lib.sr_drm_init() != 0:
      raise RuntimeError("CPU backend failed to initialize DRM/KMS")
    state.drm = True
  if not _activate_drm_back_buffer():
    state.framebuffer = np.zeros((height, width, 4), dtype=np.uint8)
    state.surface = state.make_surface(state.framebuffer)
  if pathlib.Path("/TICI").exists():
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
  state.image_buffers.clear()
  state.image_premultiplied.clear()
  state.font_storage.clear()
  state.font_maps.clear()
  state.shader_effects.clear()
  state.active_shader_effect = None
  state.gui_font = None
  state.surface = None
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
    if state.profile_enabled:
      state.profile.setdefault("drm_rotate_copy", []).append(state.lib.sr_drm_last_copy_ms())


def clear_background(color) -> None:
  state.lib.sr_clear(ctypes.byref(state.surface), _pack(color))


def draw_rectangle(x: int, y: int, width: int, height: int, color) -> None:
  x, y, width, height = _transform_rect(x, y, width, height)
  state.lib.sr_rect(ctypes.byref(state.surface), round(x), round(y), round(width), round(height), _pack(color))


def draw_rectangle_rec(rect, color) -> None:
  draw_rectangle(*_rect(rect), color)


def draw_rectangle_gradient_v(x: int, y: int, width: int, height: int, top, bottom) -> None:
  x, y, width, height = _transform_rect(x, y, width, height)
  state.lib.sr_gradient_v(ctypes.byref(state.surface), round(x), round(y), round(width), round(height),
                          _pack(top), _pack(bottom))


def draw_rectangle_gradient_h(x: int, y: int, width: int, height: int, left, right) -> None:
  x, y, width, height = _transform_rect(x, y, width, height)
  x, y, width, height = round(x), round(y), round(width), round(height)
  state.lib.sr_gradient_4(
    ctypes.byref(state.surface), x, y, width, height,
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
  x, y = _transform_xy(x, y)
  radius *= (abs(state.transform[0]) + abs(state.transform[1])) * .5
  state.lib.sr_circle(ctypes.byref(state.surface), round(x), round(y), round(radius), _pack(color))


def draw_circle_v(center, radius: float, color) -> None:
  x, y = _xy(center)
  draw_circle(round(x), round(y), radius, color)


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
  start_x, start_y = _transform_xy(start_x, start_y)
  end_x, end_y = _transform_xy(end_x, end_y)
  state.lib.sr_line(ctypes.byref(state.surface), start_x, start_y, end_x, end_y, 1.0, _pack(color))


def draw_line_ex(start, end, thick: float, color) -> None:
  x0, y0 = _xy(start)
  x1, y1 = _xy(end)
  x0, y0 = _transform_xy(x0, y0)
  x1, y1 = _transform_xy(x1, y1)
  scale = (abs(state.transform[0]) + abs(state.transform[1])) * .5
  state.lib.sr_line(ctypes.byref(state.surface), x0, y0, x1, y1, thick * scale, _pack(color))


def draw_spline_linear(points, point_count: int, thick: float, color) -> None:
  for idx in range(max(0, point_count - 1)):
    draw_line_ex(points[idx], points[idx + 1], thick, color)


def _point(value) -> _Point:
  x, y = _xy(value)
  x, y = _transform_xy(x, y)
  return _Point(x, y)


def draw_triangle_strip(points, point_count: int, color) -> None:
  for idx in range(max(0, point_count - 2)):
    if idx & 1:
      a, b, c = points[idx + 1], points[idx], points[idx + 2]
    else:
      a, b, c = points[idx], points[idx + 1], points[idx + 2]
    state.lib.sr_triangle(ctypes.byref(state.surface), _point(a), _point(b), _point(c), _pack(color))


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
  draw_rectangle_lines_ex(gpu.Rectangle(x, y, width, height), 1, color)


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
    draw_rectangle_lines_ex(gpu.Rectangle(rect.x, rect.y, rect.width, rect.height), line_thickness, color)
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
  if pixel_format == int(gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8):
    return np.frombuffer(ffi.buffer(image.data, width * height * 4), dtype=np.uint8).reshape(height, width, 4).copy()
  if pixel_format == int(gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8):
    rgba[:, :, :3] = np.frombuffer(
      ffi.buffer(image.data, width * height * 3), dtype=np.uint8,
    ).reshape(height, width, 3)
    rgba[:, :, 3] = 255
    return rgba
  if pixel_format == int(gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE):
    gray = np.frombuffer(ffi.buffer(image.data, width * height), dtype=np.uint8).reshape(height, width)
    rgba[:, :, :3] = gray[:, :, None]
    rgba[:, :, 3] = 255
    return rgba
  if pixel_format == int(gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA):
    gray_alpha = np.frombuffer(
      ffi.buffer(image.data, width * height * 2), dtype=np.uint8,
    ).reshape(height, width, 2)
    rgba[:, :, :3] = gray_alpha[:, :, 0, None]
    rgba[:, :, 3] = gray_alpha[:, :, 1]
    return rgba
  raise NotImplementedError(f"CPU image format {pixel_format}")


def load_image(path: str):
  pixels = np.asarray(PILImage.open(path).convert("RGBA"), dtype=np.uint8).copy()
  buf = ffi.new("unsigned char[]", pixels.tobytes())
  ptr = int(ffi.cast("uintptr_t", buf))
  state.image_buffers[ptr] = buf
  return gpu.Image(ffi.cast("void *", buf), pixels.shape[1], pixels.shape[0], 1,
                   gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)


def gen_image_color(width: int, height: int, color):
  if hasattr(color, "a"):
    red, green, blue, alpha = color.r, color.g, color.b, color.a
  else:
    red, green, blue, alpha = color
  pixels = np.empty((height, width, 4), dtype=np.uint8)
  pixels[:, :, 0] = red
  pixels[:, :, 1] = green
  pixels[:, :, 2] = blue
  pixels[:, :, 3] = alpha
  buf = ffi.new("unsigned char[]", pixels.tobytes())
  state.image_buffers[int(ffi.cast("uintptr_t", buf))] = buf
  return gpu.Image(ffi.cast("void *", buf), width, height, 1,
                   gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)


def load_image_from_memory(file_type: str, file_data, data_size: int):
  del file_type
  if isinstance(file_data, (bytes, bytearray, memoryview)):
    encoded = bytes(file_data[:data_size])
  else:
    encoded = bytes(ffi.buffer(file_data, data_size))
  pixels = np.asarray(PILImage.open(io.BytesIO(encoded)).convert("RGBA"), dtype=np.uint8).copy()
  buf = ffi.new("unsigned char[]", pixels.tobytes())
  state.image_buffers[int(ffi.cast("uintptr_t", buf))] = buf
  return gpu.Image(ffi.cast("void *", buf), pixels.shape[1], pixels.shape[0], 1,
                   gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)


def unload_image(image) -> None:
  pointer = int(ffi.cast("uintptr_t", image.data))
  state.image_buffers.pop(pointer, None)
  state.image_premultiplied.discard(pointer)


def _replace_image(image, pixels: np.ndarray) -> None:
  old = int(ffi.cast("uintptr_t", image.data))
  premultiplied = old in state.image_premultiplied
  state.image_buffers.pop(old, None)
  state.image_premultiplied.discard(old)
  buf = ffi.new("unsigned char[]", pixels.tobytes())
  pointer = int(ffi.cast("uintptr_t", buf))
  state.image_buffers[pointer] = buf
  if premultiplied:
    state.image_premultiplied.add(pointer)
  image.data = ffi.cast("void *", buf)
  image.width = pixels.shape[1]
  image.height = pixels.shape[0]
  image.format = gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8


def image_resize(image, width: int, height: int) -> None:
  pil = PILImage.fromarray(_rgba_image(image), "RGBA").resize((width, height), PILImage.Resampling.BILINEAR)
  _replace_image(image, np.asarray(pil, dtype=np.uint8))


def image_flip_horizontal(image) -> None:
  _replace_image(image, _rgba_image(image)[:, ::-1].copy())


def image_alpha_premultiply(image) -> None:
  pixels = _rgba_image(image).astype(np.uint16)
  pixels[:, :, :3] = (pixels[:, :, :3] * pixels[:, :, 3:4] + 127) // 255
  _replace_image(image, pixels.astype(np.uint8))
  state.image_premultiplied.add(int(ffi.cast("uintptr_t", image.data)))


def load_texture_from_image(image):
  rgba = _rgba_image(image).astype(np.uint16)
  if int(ffi.cast("uintptr_t", image.data)) not in state.image_premultiplied:
    rgba[:, :, :3] = (rgba[:, :, :3] * rgba[:, :, 3:4] + 127) // 255
  rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
  texture_id = state.next_texture_id
  state.next_texture_id += 1
  state.textures[texture_id] = _TextureData(rgba, state.make_surface(rgba))
  return gpu.Texture(texture_id, int(image.width), int(image.height), 1,
                     gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)


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
  if pixel_format == int(gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8):
    raw = np.frombuffer(ffi.buffer(pixels, width * height * 4), dtype=np.uint8).reshape(height, width, 4)
    rgba = raw.astype(np.uint16)
    rgba[:, :, :3] = (rgba[:, :, :3] * rgba[:, :, 3:4] + 127) // 255
    data.pixels[:] = rgba.astype(np.uint8)
  elif pixel_format == int(gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAYSCALE):
    gray = np.frombuffer(ffi.buffer(pixels, width * height), dtype=np.uint8).reshape(height, width)
    data.pixels[:, :, :3] = gray[:, :, None]
    data.pixels[:, :, 3] = 255
  elif pixel_format == int(gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA):
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
  texture = gpu.Texture(texture_id, width, height, 1,
                        gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
  return gpu.RenderTexture(texture_id, texture, gpu.Texture())


def unload_render_texture(target) -> None:
  texture_id = int(target.texture.id)
  state.render_targets.pop(texture_id, None)
  state.textures.pop(texture_id, None)


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
    return gpu.Image()
  pixels = data.pixels[::-1] if int(texture.id) in state.render_targets else data.pixels
  rgba = np.ascontiguousarray(pixels)
  buf = ffi.new("unsigned char[]", rgba.tobytes())
  state.image_buffers[int(ffi.cast("uintptr_t", buf))] = buf
  return gpu.Image(ffi.cast("void *", buf), rgba.shape[1], rgba.shape[0], 1,
                   gpu.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)


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
    recs[idx] = gpu.Rectangle(float(entry["x"]), float(entry["y"]),
                              float(entry["width"]), float(entry["height"]))
    glyphs[idx].value = value
    glyphs[idx].offsetX = int(entry["xoffset"])
    glyphs[idx].offsetY = int(entry["yoffset"])
    glyphs[idx].advanceX = int(entry["xadvance"])
    glyph_map[value] = idx
  font = gpu.Font(int(common["lineHeight"]), len(entries), 0, texture, recs, glyphs)
  key = int(ffi.cast("uintptr_t", glyphs))
  state.font_storage.append((recs, glyphs, glyph_map))
  state.font_maps[key] = glyph_map
  return font


def unload_font(font) -> None:
  unload_texture(font.texture)


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
  return gpu.Vector2(max_width, lines * font_size)


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
    if (atlas.filter_mode != int(gpu.TextureFilter.TEXTURE_FILTER_POINT) and
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


def color_to_int(color) -> int:
  if hasattr(color, "r"):
    red, green, blue, alpha = color.r, color.g, color.b, color.a
  else:
    red, green, blue, alpha = color
  return (int(red) << 24) | (int(green) << 16) | (int(blue) << 8) | int(alpha)


def color_to_hsv(color):
  red, green, blue = (color.r, color.g, color.b) if hasattr(color, "r") else color[:3]
  hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
  return gpu.Vector3(hue * 360, saturation, value)


def color_from_hsv(hue: float, saturation: float, value: float):
  red, green, blue = colorsys.hsv_to_rgb((hue % 360) / 360, saturation, value)
  return gpu.Color(round(red * 255), round(green * 255), round(blue * 255), 255)


def check_collision_point_rec(point, rect) -> bool:
  x, y = _xy(point)
  return rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height


def check_collision_recs(first, second) -> bool:
  return (first.x < second.x + second.width and first.x + first.width > second.x and
          first.y < second.y + second.height and first.y + first.height > second.y)


def get_collision_rec(first, second):
  x = max(first.x, second.x)
  y = max(first.y, second.y)
  right = min(first.x + first.width, second.x + second.width)
  bottom = min(first.y + first.height, second.y + second.height)
  return gpu.Rectangle(x, y, max(0, right - x), max(0, bottom - y))


def gui_set_font(font) -> None:
  state.gui_font = font


def gui_set_style(control, prop, value: int) -> None:
  state.gui_styles[(int(control), int(prop))] = int(value)


def gui_get_style(control, prop) -> int:
  key = (int(control), int(prop))
  if key in state.gui_styles:
    return state.gui_styles[key]
  defaults = {
    int(gpu.GuiDefaultProperty.TEXT_SIZE): 10,
    int(gpu.GuiDefaultProperty.TEXT_LINE_SPACING): 15,
    int(gpu.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL): int(gpu.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE),
    int(gpu.GuiDefaultProperty.TEXT_WRAP_MODE): int(gpu.GuiTextWrapMode.TEXT_WRAP_NONE),
    int(gpu.GuiControlProperty.TEXT_ALIGNMENT): int(gpu.GuiTextAlignment.TEXT_ALIGN_LEFT),
    int(gpu.GuiControlProperty.TEXT_COLOR_NORMAL): color_to_int(gpu.DARKGRAY),
  }
  return defaults.get(int(prop), 0)


def _gui_color(value: int):
  value &= 0xFFFFFFFF
  return gpu.Color((value >> 24) & 255, (value >> 16) & 255, (value >> 8) & 255, value & 255)


def gui_label(rect, text: str) -> None:
  if state.gui_font is None:
    return
  control = gpu.GuiControl.DEFAULT
  font_size = gui_get_style(control, gpu.GuiDefaultProperty.TEXT_SIZE)
  line_spacing = gui_get_style(control, gpu.GuiDefaultProperty.TEXT_LINE_SPACING)
  alignment = gui_get_style(control, gpu.GuiControlProperty.TEXT_ALIGNMENT)
  vertical = gui_get_style(control, gpu.GuiDefaultProperty.TEXT_ALIGNMENT_VERTICAL)
  wrap = gui_get_style(control, gpu.GuiDefaultProperty.TEXT_WRAP_MODE)
  lines: list[str] = []
  for paragraph in text.splitlines() or [""]:
    if wrap == int(gpu.GuiTextWrapMode.TEXT_WRAP_NONE):
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
  if vertical == int(gpu.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE):
    y += (rect.height - total_height) / 2
  elif vertical == int(gpu.GuiTextAlignmentVertical.TEXT_ALIGN_BOTTOM):
    y += rect.height - total_height
  tint = _gui_color(gui_get_style(control, gpu.GuiControlProperty.TEXT_COLOR_NORMAL))
  for line in lines:
    width = measure_text_ex(state.gui_font, line, font_size, 0).x
    x = rect.x
    if alignment == int(gpu.GuiTextAlignment.TEXT_ALIGN_CENTER):
      x += (rect.width - width) / 2
    elif alignment == int(gpu.GuiTextAlignment.TEXT_ALIGN_RIGHT):
      x += rect.width - width
    draw_text_ex(state.gui_font, line, gpu.Vector2(x, y), font_size, 0, tint)
    y += line_spacing


def load_shader_from_memory(vertex_source: str | None, fragment_source: str | None):
  del vertex_source
  shader_id = state.next_shader_id
  state.next_shader_id += 1
  effect = "burn_in" if fragment_source and "highlight burn-in risk" in fragment_source else "passthrough"
  state.shader_effects[shader_id] = effect
  return gpu.Shader(shader_id, ffi.NULL)


def unload_shader(shader) -> None:
  state.shader_effects.pop(int(shader.id), None)


def get_shader_location(shader, name: str) -> int:
  del shader, name
  return 0


def begin_shader_mode(shader) -> None:
  state.active_shader_effect = state.shader_effects.get(int(shader.id))


def end_shader_mode() -> None:
  state.active_shader_effect = None


def set_shader_value(shader, location: int, value, uniform_type) -> None:
  del shader, location, value, uniform_type


def set_shader_value_v(shader, location: int, value, uniform_type, count: int) -> None:
  del shader, location, value, uniform_type, count


def set_shader_value_matrix(shader, location: int, matrix) -> None:
  del shader, location, matrix


def set_shader_value_texture(shader, location: int, texture) -> None:
  del shader, location, texture


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
    source = gpu.Rectangle(source.x, source.y, source.width, -source.height)
  ox, oy = _xy(origin)
  dx, dy, dw, dh = _transform_rect(dest.x, dest.y, dest.width, dest.height)
  ox *= state.transform[0]
  oy *= state.transform[1]
  width, height = max(1, round(abs(dw))), max(1, round(abs(dh)))
  if (data.filter_mode != int(gpu.TextureFilter.TEXTURE_FILTER_POINT) and
      texture_id not in state.render_targets and
      (abs(abs(float(source.width)) - width) > .001 or abs(abs(float(source.height)) - height) > .001)):
    data = _scaled_texture(texture_id, data, source, width, height)
    source = gpu.Rectangle(0, 0, width, height)
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
  if state.active_shader_effect == "burn_in":
    state.lib.sr_burn_in_filter(
      ctypes.byref(state.surface), round(dx - ox), round(dy - oy), round(dw), round(dh),
    )


def draw_texture_ex(texture, position, rotation: float, scale: float, tint) -> None:
  px, py = _xy(position)
  source = gpu.Rectangle(0, 0, texture.width, texture.height)
  dest = gpu.Rectangle(px, py, texture.width * scale, texture.height * scale)
  draw_texture_pro(texture, source, dest, gpu.Vector2(0, 0), rotation, tint)


def draw_texture_v(texture, position, tint) -> None:
  draw_texture_ex(texture, position, 0, 1, tint)


def _convert_nv12(entry: _Nv12Data, frame_width: int, frame_height: int, stride: int, uv_offset: int,
                  source_rect: tuple[float, float, float, float], flip_x: bool,
                  engaged: bool, enhance_driver: bool) -> None:
  started = time.perf_counter_ns()
  source_x, source_y, source_width, source_height = source_rect
  state.lib.sr_draw_nv12_crop(
    ctypes.byref(entry.back.surface), entry.source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    frame_width, frame_height, stride, uv_offset, source_x, source_y, source_width, source_height,
    0, 0, entry.back.surface.width, entry.back.surface.height,
    int(flip_x), int(engaged), int(enhance_driver),
  )
  if state.profile_enabled:
    state.profile.setdefault("nv12_convert", []).append((time.perf_counter_ns() - started) / 1e6)


def draw_nv12(frame, dest, engaged: bool, enhance_driver: bool, flip_x: bool = False,
              cache_key: int = 0, needs_update: bool = True) -> None:
  started = time.perf_counter_ns()
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
  if state.profile_enabled and "nv12_shape" not in state.profile:
    description = f"source={frame.width}x{frame.height} stride={frame.stride} uv_offset={frame.uv_offset} "
    description += f"destination={full_width}x{full_height} visible={width}x{height} "
    description += f"raw_destination={dest.width}x{dest.height} transform={state.transform}"
    print(f"CPU NV12 shape: {description}")
    state.profile["nv12_shape"] = []
  if state.drm:
    mdp_started = time.perf_counter_ns()
    result = state.lib.sr_drm_set_camera(
      int(frame.fd), int(frame.width), int(frame.height), int(frame.stride), int(frame.uv_offset),
      source_x, source_y, source_width, source_height,
      visible_x, visible_y, width, height, int(flip_x), int(engaged), int(enhance_driver),
    )
    if result == 0:
      state.lib.sr_clear_transparent(ctypes.byref(state.surface), visible_x, visible_y, width, height)
      if state.profile_enabled:
        state.profile.setdefault("mdp_camera_setup", []).append((time.perf_counter_ns() - mdp_started) / 1e6)
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

  prepared = time.perf_counter_ns()
  copied = prepared
  if needs_update and entry.future is None:
    np.copyto(entry.source, np.frombuffer(frame.data, dtype=np.uint8, count=source_size))
    copied = time.perf_counter_ns()
    if state.nv12_executor is None:
      state.nv12_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cpu-camera", initializer=_camera_worker_init,
      )
    entry.future = state.nv12_executor.submit(
      _convert_nv12, entry, int(frame.width), int(frame.height), int(frame.stride), int(frame.uv_offset),
      (source_x, source_y, source_width, source_height), flip_x, engaged, enhance_driver,
    )
  submitted = time.perf_counter_ns()
  if entry.has_front:
    state.lib.sr_blit_opaque(ctypes.byref(state.surface), ctypes.byref(entry.front.surface), visible_x, visible_y)
  if state.profile_enabled:
    finished = time.perf_counter_ns()
    state.profile.setdefault("nv12_prepare", []).append((prepared - started) / 1e6)
    state.profile.setdefault("nv12_copy", []).append((copied - prepared) / 1e6)
    state.profile.setdefault("nv12_submit", []).append((submitted - copied) / 1e6)
    state.profile.setdefault("nv12_blit", []).append((finished - submitted) / 1e6)


def set_target_fps(fps: int) -> None:
  state.target_fps = fps


def get_frame_time() -> float:
  return state.frame_time


def get_time() -> float:
  return time.monotonic() - state.start


def get_fps() -> int:
  return round(1 / state.frame_time) if state.frame_time > 0 else 0


_DEBUG_GLYPHS = {
  "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
  "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
  "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
  "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
  "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
  "F": ("111", "100", "110", "100", "100"), "P": ("110", "101", "110", "100", "100"),
  "S": ("111", "100", "111", "001", "111"), " ": ("000", "000", "000", "000", "000"),
}


def draw_fps(pos_x: int, pos_y: int) -> None:
  cursor = pos_x
  for char in f"{get_fps()} FPS":
    rows = _DEBUG_GLYPHS.get(char, _DEBUG_GLYPHS[" "])
    for row, bits in enumerate(rows):
      for column, bit in enumerate(bits):
        if bit == "1":
          draw_rectangle(cursor + column * 2, pos_y + row * 2, 2, 2, gpu.LIME)
    cursor += 8


def set_config_flags(flags) -> None:
  del flags


def set_trace_log_callback(callback) -> None:
  del callback


def set_trace_log_level(level) -> None:
  del level


def matrix_ortho(left: float, right: float, bottom: float, top: float, near_plane: float, far_plane: float):
  width = right - left
  height = top - bottom
  depth = far_plane - near_plane
  return gpu.Matrix(
    2 / width, 0, 0, -(right + left) / width,
    0, 2 / height, 0, -(top + bottom) / height,
    0, 0, -2 / depth, -(far_plane + near_plane) / depth,
    0, 0, 0, 1,
  )


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
    return gpu.Vector2(-1, -1)
  return gpu.Vector2(*state.touch_xy[index])


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
  return gpu.Vector2(x * state.mouse_scale[0], y * state.mouse_scale[1])


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


def framebuffer() -> np.ndarray:
  if state.framebuffer is None:
    raise RuntimeError("CPU window is not initialized")
  return state.framebuffer


def cpu_profile_stats() -> dict[str, tuple[int, float, float]]:
  return {
    name: (len(samples), float(np.percentile(samples, 50)), float(np.percentile(samples, 95)))
    for name, samples in state.profile.items() if samples
  }


EXPORTED = {
  name: value for name, value in globals().items()
  if not name.startswith("_") and callable(value) and getattr(value, "__module__", None) == __name__
}
