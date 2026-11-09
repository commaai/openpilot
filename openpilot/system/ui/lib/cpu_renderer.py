"""raylib on the CPU, keeping the GPU free for the driving model.

install() replaces the drawing functions of raylib_nb with its software rasterizer (rl.cpu)
that scans out through KMS. Structs, images and math still come from raylib.
"""
import os
import socket
import struct
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import raylib_nb as rl

from openpilot.common.swaglog import cloudlog

SOM_ID = Path("/sys/devices/platform/vendor/vendor:gpio-som-id/som_id")
COLOR_CAL = ("/data/misc/display/color_cal/color_cal", "/sys/devices/platform/soc/894000.i2c/i2c-2/2-0017/color_cal",
             "/persist/comma/color_cal")
TOUCH_DEVICE = "/dev/input/event2"
DSPP_COLOR = ("SDE_DSPP_IGC_V3", "SDE_DSPP_PCC_V4", "SDE_DSPP_GC_V1")
TEXT_LINE_SPACING = 2  # raylib's default

cpu = rl.cpu


def _pack(c) -> int:
  return c.r | (c.g << 8) | (c.b << 16) | (c.a << 24)


class _Font:
  def __init__(self, path: str, size: int, codepoints: list[int]):
    self.data = Path(path).read_bytes()
    self.base_size = size
    # base size metrics, used for layout exactly like raylib. glyphs without an advance are measured with their offset
    glyphs = self._load(size, codepoints)
    self.advance = {cp: adv or a.shape[1] for cp, adv, (a, _, _) in glyphs}
    self.measure = {cp: adv or a.shape[1] + ox for cp, adv, (a, ox, _) in glyphs}
    self.fallback = 63 if 63 in self.advance else next(iter(self.advance))
    self.bitmaps: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}

  def _load(self, size: int, codepoints: list[int]) -> list[tuple[int, int, tuple[np.ndarray, int, int]]]:
    """(codepoint, advance, (alpha bitmap, offset x, offset y)) of each glyph found in the font"""
    return [(cp, adv, (np.frombuffer(a, np.uint8).reshape(h, w), ox, oy))
            for cp, adv, ox, oy, a, w, h in cpu.glyphs(self.data, size, codepoints)]

  def glyph(self, cp: int, size: int) -> tuple[np.ndarray, int, int]:
    """alpha bitmap of a codepoint rendered at size, and its offset from the pen"""
    if (cp, size) not in self.bitmaps:
      glyphs = self._load(size, [cp])
      self.bitmaps[(cp, size)] = glyphs[0][2] if glyphs else (np.zeros((0, 0), np.uint8), 0, 0)
    return self.bitmaps[(cp, size)]

  def layout(self, text: str, font_size: float, spacing: float):
    """pen positions of each drawn codepoint, same as raylib's DrawTextEx"""
    scale = font_size / self.base_size
    x = y = 0.0
    for ch in text:
      cp = ord(ch)
      if cp == 10:
        x, y = 0.0, y + font_size + TEXT_LINE_SPACING
        continue
      if cp not in self.advance:
        cp = self.fallback
      if ch not in " \t":
        yield cp, x, y
      x += self.advance[cp] * scale + spacing


class _Renderer:
  def __init__(self):
    self.fb = np.zeros((1, 1, 4), np.uint8)
    self.textures: dict[int, tuple[np.ndarray, bool]] = {}  # id -> (premultiplied RGBA, bilinear)
    self.fonts: dict[int, _Font] = {}
    self.text_cache: OrderedDict = OrderedDict()
    self.next_id = 1
    self.transform = (1.0, 1.0, 0.0, 0.0)  # sx, sy, tx, ty
    self.transform_stack: list[tuple[float, float, float, float]] = []
    self.display = False
    self.drm_sock: socket.socket | None = None
    self.drm_fd = -1
    self.last_frame = self.frame_time = 0.0
    self.frame_times: deque[float] = deque(maxlen=30)  # raylib averages the fps over 30 frames
    self.target_frame_time = 0.0
    # touch, as raylib's comma platform
    self.touch_fd = -1
    self.canonical_zero = False
    self.slot = 0
    self.finger_down = [False] * 2
    self.finger_released = [False] * 2
    self.pos = [rl.Vector2(0, 0) for _ in range(2)]
    self.down = [False] * 2
    self.prev_down = [False] * 2
    self.reset_next = [False] * 2

  def new_id(self) -> int:
    self.next_id += 1
    return self.next_id

  # ***** window *****

  def init_window(self, width: int, height: int, title: str):
    self.last_frame = time.monotonic()
    self.fb = np.zeros((height, width, 4), np.uint8)
    cpu.target(self.fb, width, height, width * 4)
    if not os.path.exists("/dev/dri/card0"):
      return
    try:
      self.canonical_zero = SOM_ID.read_text().strip() == "1"
    except OSError:
      pass
    if "DRM_FD" in os.environ:
      fd = int(os.environ["DRM_FD"])
    else:
      self.drm_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
      self.drm_sock.connect("/tmp/drmfd.sock")  # magic hands out DRM master while this stays open
      fd = self.drm_fd = socket.recv_fds(self.drm_sock, 1, 1)[1][0]
    ret = cpu.display_init(fd, self.canonical_zero)
    if ret != 0:
      raise RuntimeError(f"failed to init display: {os.strerror(-ret)}")
    self.display = True
    # drawn straight into the buffer the display's rotator reads
    if (target := cpu.display_target(width, height)) is None:
      raise RuntimeError(f"{width}x{height} window doesn't fit the display")
    self.fb = np.frombuffer(target, np.uint8).reshape(height, width, 4)
    cpu.target(self.fb, width, height, width * 4)
    cpu.display_present()  # the CRTC has to be on to take color correction
    for name, blob in self._color_correction().items():
      if (ret := cpu.display_blob(name, blob)) != 0:
        cloudlog.warning(f"cpu renderer: failed to set {name}: {os.strerror(-ret)}")
    self.touch_fd = os.open(TOUCH_DEVICE, os.O_RDONLY | os.O_NONBLOCK)

  def _color_correction(self) -> dict[str, bytes | None]:
    """raylib's color correction shader as the display's own degamma, matrix and gamma blocks"""
    off = dict.fromkeys(DSPP_COLOR)
    if self.canonical_zero:
      return off
    for path in COLOR_CAL:
      try:
        gamma, *rest = struct.unpack("<13e", Path(path).read_bytes()[:26])
      except (OSError, struct.error):
        continue
      ccm, gains = np.array(rest[:9]).reshape(3, 3), np.array(rest[9:])
      if 0 < gamma < np.inf and np.isfinite(ccm).all() and np.isfinite(gains).all() and (gains > 0).all():
        break
    else:
      return off
    # 256 entry 12 bit degamma and 1024 entry 10 bit gamma luts per channel, and a matrix in 1/32768 with a plane per output channel
    igc = np.round((np.arange(256) / 255) ** 2.2 * 4095).astype("<u4")
    gc = np.round((np.arange(1024) / 1023) ** (1 / gamma / 2.2) * 1023).astype("<u4")
    pcc = np.zeros((3, 8), "<i4")  # c, r, g, b, rg, gb, rb, rgb
    pcc[:, 1:4] = np.round((ccm / gains[:, None]).T * 32768)
    # structs of msm_drm_pp.h, flags first and padded to 8 bytes
    return {
      "SDE_DSPP_IGC_V3": bytes(8) + np.tile(igc, 3).tobytes() + bytes(8),
      "SDE_DSPP_PCC_V4": bytes(8) + pcc.tobytes() + bytes(40),
      "SDE_DSPP_GC_V1": bytes(8) + np.tile(gc[0::2] | gc[1::2] << 16, 3).tobytes(),
    }

  def close_window(self):
    if self.display:
      for name in DSPP_COLOR:
        cpu.display_blob(name, None)
      self.fb = self.fb.copy()  # the display's buffer goes away
      cpu.display_close()
      cpu.target(self.fb, self.fb.shape[1], self.fb.shape[0], self.fb.shape[1] * 4)
      self.display = False
    if self.drm_fd >= 0:
      os.close(self.drm_fd)
      self.drm_fd = -1
    if self.drm_sock is not None:
      self.drm_sock.close()
      self.drm_sock = None
    if self.touch_fd >= 0:
      os.close(self.touch_fd)
      self.touch_fd = -1

  def begin_drawing(self):
    now = time.monotonic()
    self.frame_time, self.last_frame = now - self.last_frame, now
    self.frame_times.append(self.frame_time)

  def end_drawing(self):
    if self.display:
      ret = cpu.display_present()
      if ret != 0:
        cloudlog.warning(f"cpu renderer: failed to present: {os.strerror(-ret)}")
    # like raylib, sleep out the rest of the frame
    if (wait := self.last_frame + self.target_frame_time - time.monotonic()) > 0:
      time.sleep(wait)

  # ***** input *****

  def poll_input_events(self):
    for i in range(2):
      self.prev_down[i] = self.down[i]
      if self.reset_next[i]:
        self.down[i] = self.reset_next[i] = False
    if self.touch_fd < 0:
      return
    data = b""
    while True:
      try:
        data += os.read(self.touch_fd, 24 * 64)
      except BlockingIOError:
        break
    w, h = self.fb.shape[1], self.fb.shape[0]
    for _, _, etype, code, value in struct.iter_unpack("qqHHi", data):
      if etype == 0:  # EV_SYN
        for i in range(2):
          if self.finger_down[i]:
            self.down[i] = True
          elif self.finger_released[i]:
            # down and up within one frame: report the release on the next one
            self.reset_next[i] = not self.prev_down[i]
            self.down[i] = not self.prev_down[i]
            self.finger_released[i] = False
      elif etype == 3:  # EV_ABS
        if code == 0x2f:  # ABS_MT_SLOT
          self.slot = value
        elif self.slot >= 2:
          continue
        elif code == 0x39:  # ABS_MT_TRACKING_ID
          self.finger_down[self.slot] = value != -1
          self.finger_released[self.slot] = value == -1
        elif code == 0x35:  # ABS_MT_POSITION_X
          self.pos[self.slot].y = value if self.canonical_zero else h - value
        elif code == 0x36:  # ABS_MT_POSITION_Y
          self.pos[self.slot].x = w - value if self.canonical_zero else value

  # ***** drawing *****

  def xy(self, x: float, y: float) -> tuple[float, float]:
    sx, sy, tx, ty = self.transform
    return x * sx + tx, y * sy + ty

  def rect(self, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    sx, sy, tx, ty = self.transform
    return x * sx + tx, y * sy + ty, w * sx, h * sy

  def fill(self, x, y, w, h, tl, tr, bl, br):
    x, y, w, h = self.rect(x, y, w, h)
    cpu.rect(x, y, x + w, y + h, tl, tr, bl, br)

  def rounded(self, rec, roundness: float, thickness: float, color, outside: bool = False):
    x, y, w, h = self.rect(rec.x, rec.y, rec.width, rec.height)
    radius = min(w, h) * min(roundness, 1.0) / 2
    t = thickness * self.transform[1]
    if outside:
      x, y, w, h, radius = x - t, y - t, w + 2 * t, h + 2 * t, radius + t
    cpu.rounded_rect(rl.Rectangle(x, y, w, h), radius, t, color)

  def ring(self, cx, cy, r0, r1, a0, a1, inner, outer):
    cx, cy = self.xy(cx, cy)
    s = self.transform[1]
    cpu.ring(rl.Vector2(cx, cy), r0 * s, r1 * s, min(a0, a1), max(a0, a1), inner, outer)

  def triangles(self, points, colors, mode: int):
    if not isinstance(points, np.ndarray):
      points = [(p.x, p.y) if isinstance(p, rl.Vector2) else p for p in points]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    sx, sy, tx, ty = self.transform
    pts = np.ascontiguousarray(pts * (sx, sy) + (tx, ty), dtype=np.float32)
    cols = np.ascontiguousarray(colors, dtype=np.uint32)
    cpu.triangles(pts, cols, mode)

  def line(self, x0, y0, x1, y1, thick, color):
    d = np.array([x1 - x0, y1 - y0], dtype=np.float32)
    length = float(np.hypot(*d))
    if length == 0 or thick <= 0:
      return
    nx, ny = -d[1] / length * thick / 2, d[0] / length * thick / 2
    self.triangles([(x0 - nx, y0 - ny), (x0 + nx, y0 + ny), (x1 - nx, y1 - ny), (x1 + nx, y1 + ny)], [_pack(color)], 1)

  def spline(self, points, count: int, thick: float, color):
    """raylib's DrawSplineLinear: a quad along each segment"""
    p = np.array([(v.x, v.y) for v in points[:count]], dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
      return
    d = p[1:] - p[:-1]
    length = np.hypot(d[:, 0], d[:, 1])
    r = np.column_stack([-d[:, 1], d[:, 0]]) * np.divide(thick / 2, length, out=np.zeros_like(length), where=length > 0)[:, None]
    a0, a1, b0, b1 = p[:-1] - r, p[:-1] + r, p[1:] - r, p[1:] + r
    self.triangles(np.stack([a0, a1, b0, a1, b0, b1], axis=1).reshape(-1, 2), [_pack(color)], 0)

  def blit(self, pixels: np.ndarray, a8: bool, src, dst, origin, rotation: float, tint, bilinear: bool):
    sx, sy, tx, ty = self.transform
    cpu.blit(pixels, pixels.shape[1], pixels.shape[0], pixels.strides[0], a8, src,
             rl.Rectangle(dst.x * sx + tx, dst.y * sy + ty, dst.width * sx, dst.height * sy), rl.Vector2(origin.x * sx, origin.y * sy),
             rotation, tint, bilinear)

  def draw_texture_pro(self, texture, src, dst, origin, rotation, tint):
    if texture.id in self.textures:
      pixels, bilinear = self.textures[texture.id]
      # unscaled draws sample texel centers, filtering them would only cost time
      scaled = (rotation != 0 or abs(abs(src.width) - dst.width * self.transform[0]) > 1e-3 or
                abs(abs(src.height) - dst.height * self.transform[1]) > 1e-3)
      self.blit(pixels, False, src, dst, origin, rotation, tint, bilinear and scaled)

  # ***** textures *****

  def load_texture_from_image(self, image):
    w, h, fmt = image.width, image.height, image.format
    texture = rl.Texture(self.new_id(), w, h, 1, fmt)
    self.textures[texture.id] = (np.zeros((h, w, 4), np.uint8), True)
    try:
      data = image.to_bytes(w * h * {1: 1, 2: 2, 4: 3, 7: 4}[fmt])
    except ValueError:
      return texture  # no pixels yet
    self.update_texture(texture, data)
    return texture

  def update_texture(self, texture, data):
    pixels = self.textures[texture.id][0]
    h, w = pixels.shape[:2]
    channels = {1: 1, 2: 2, 4: 3, 7: 4}[texture.format]
    src = np.frombuffer(data, np.uint8, count=w * h * channels).reshape(h, w, channels)
    pixels[..., 3] = src[..., 3] if channels == 4 else src[..., 1] if channels == 2 else 255
    pixels[..., :3] = (src[..., :3 if channels >= 3 else 1].astype(np.uint16) * pixels[..., 3:4] + 127) // 255

  def unload_texture(self, texture):
    self.textures.pop(texture.id, None)

  def set_texture_filter(self, texture, filter_mode):
    if texture.id in self.textures:
      self.textures[texture.id] = (self.textures[texture.id][0], filter_mode != rl.TextureFilter.TEXTURE_FILTER_POINT)

  # ***** text *****

  def load_font_ex(self, path: str, size: int, codepoints, count: int):
    font = rl.Font()
    font.baseSize = size
    font.glyphCount = count or 95
    font.texture = rl.Texture(self.new_id(), 1, 1, 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
    self.fonts[font.texture.id] = _Font(path, size, list(codepoints) if codepoints else list(range(32, 127)))
    return font

  def unload_font(self, font):
    self.fonts.pop(font.texture.id, None)

  def measure_text_ex(self, font, text: str, font_size: float, spacing: float):
    f = self.fonts.get(font.texture.id)
    if f is None or not text:
      return rl.Vector2(0, 0)
    scale = font_size / f.base_size
    lines = text.split("\n")
    width = max(sum(f.measure.get(ord(ch), f.measure[f.fallback]) for ch in line) for line in lines)
    chars = max(len(line) for line in lines)
    return rl.Vector2(width * scale + (chars - 1) * spacing, font_size + (len(lines) - 1) * (font_size + TEXT_LINE_SPACING))

  def draw_text_ex(self, font, text: str, position, font_size: float, spacing: float, tint):
    f = self.fonts.get(font.texture.id)
    if f is None or not text:
      return
    size = font_size * self.transform[1]
    key = (font.texture.id, text, round(size, 2), spacing)
    if key in self.text_cache:
      self.text_cache.move_to_end(key)
    else:
      self.text_cache[key] = self._render_text(f, text, size, spacing)
      if len(self.text_cache) > 512:
        self.text_cache.popitem(last=False)
    mask, ox, oy = self.text_cache[key]
    if mask.size:
      x, y = self.xy(position.x, position.y)
      r = rl.Rectangle(0, 0, mask.shape[1], mask.shape[0])
      sx, sy, tx, ty = self.transform
      self.transform = (1.0, 1.0, 0.0, 0.0)
      self.blit(mask, True, r, rl.Rectangle(round(x) + ox, round(y) + oy, r.width, r.height), rl.Vector2(0, 0), 0, tint, False)
      self.transform = (sx, sy, tx, ty)

  def _render_text(self, f: _Font, text: str, size: float, spacing: float) -> tuple[np.ndarray, int, int]:
    """alpha mask of a whole string, and its offset from the text position"""
    px = max(1, round(size))
    glyphs = [(f.glyph(cp, px), round(x), round(y)) for cp, x, y in f.layout(text, size, spacing)]
    glyphs = [((a, gx + x, gy + y)) for (a, gx, gy), x, y in glyphs if a.size]
    if not glyphs:
      return np.zeros((0, 0), np.uint8), 0, 0
    x0, y0 = min(g[1] for g in glyphs), min(g[2] for g in glyphs)
    x1, y1 = max(g[1] + g[0].shape[1] for g in glyphs), max(g[2] + g[0].shape[0] for g in glyphs)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for a, x, y in glyphs:
      region = mask[y - y0:y - y0 + a.shape[0], x - x0:x - x0 + a.shape[1]]
      np.maximum(region, a, out=region)
    return mask, x0, y0

  # ***** transform *****

  def push_matrix(self):
    self.transform_stack.append(self.transform)

  def pop_matrix(self):
    self.transform = self.transform_stack.pop()

  def scalef(self, x: float, y: float, z: float):
    sx, sy, tx, ty = self.transform
    self.transform = (sx * x, sy * y, tx, ty)

  def translatef(self, x: float, y: float, z: float):
    sx, sy, tx, ty = self.transform
    self.transform = (sx, sy, tx + x * sx, ty + y * sy)


_r = _Renderer()


def draw_nv12(frame, dst, flip_x: bool, curve: np.ndarray, saturation: float, new_frame: bool):
  """draw a VisionIPC frame mixed with its luma by saturation and each channel mapped through curve, in place of a frame shader"""
  cpu.nv12(frame.data, frame.uv_offset, frame.stride, frame.width, frame.height, rl.Rectangle(*_r.rect(dst.x, dst.y, dst.width, dst.height)),
           flip_x, curve, round(saturation * 256), new_frame)


def draw_triangle_strip_colors(points: np.ndarray, colors: np.ndarray):
  """triangle strip with a straight alpha RGBA color per point"""
  _r.triangles(points, np.ascontiguousarray(colors, dtype=np.uint8).view(np.uint32).ravel(), 1)


def install():
  noop = lambda *args, **kwargs: None  # noqa: E731
  rl.__dict__.update({
    "init_window": _r.init_window,
    "close_window": _r.close_window,
    "is_window_ready": lambda: True,
    "window_should_close": lambda: False,
    "get_monitor_width": lambda monitor: _r.fb.shape[1],
    "get_monitor_height": lambda monitor: _r.fb.shape[0],
    "set_config_flags": noop,
    "set_target_fps": lambda fps: setattr(_r, "target_frame_time", 1 / fps if fps > 0 else 0.0),
    "begin_drawing": _r.begin_drawing,
    "end_drawing": _r.end_drawing,
    "get_time": time.monotonic,
    "get_frame_time": lambda: _r.frame_time,
    "get_fps": lambda: round(len(_r.frame_times) / sum(_r.frame_times)) if sum(_r.frame_times) else 0,
    "draw_fps": noop,
    "clear_background": cpu.clear,
    "begin_scissor_mode": lambda x, y, w, h: cpu.clip(x, y, x + w, y + h),
    "end_scissor_mode": lambda: cpu.clip(0, 0, _r.fb.shape[1], _r.fb.shape[0]),
    "rl_push_matrix": _r.push_matrix,
    "rl_pop_matrix": _r.pop_matrix,
    "rl_scalef": _r.scalef,
    "rl_translatef": _r.translatef,

    "poll_input_events": _r.poll_input_events,
    "get_touch_position": lambda index: _r.pos[index],
    "get_mouse_position": lambda: _r.pos[0],
    "is_mouse_button_down": lambda button: _r.down[button],
    "is_mouse_button_pressed": lambda button: _r.down[button] and not _r.prev_down[button],
    "is_mouse_button_released": lambda button: _r.prev_down[button] and not _r.down[button],
    "set_mouse_scale": noop,
    "get_mouse_wheel_move": lambda: 0.0,
    "get_key_pressed": lambda: 0,
    "get_char_pressed": lambda: 0,
    "is_key_down": lambda key: False,
    "is_key_pressed": lambda key: False,
    "is_key_released": lambda key: False,

    "draw_rectangle": lambda x, y, w, h, c: _r.fill(x, y, w, h, c, c, c, c),
    "draw_rectangle_rec": lambda rec, c: _r.fill(rec.x, rec.y, rec.width, rec.height, c, c, c, c),
    "draw_rectangle_gradient_v": lambda x, y, w, h, top, bottom: _r.fill(x, y, w, h, top, top, bottom, bottom),
    "draw_rectangle_gradient_h": lambda x, y, w, h, left, right: _r.fill(x, y, w, h, left, right, left, right),
    "draw_rectangle_gradient_ex": lambda rec, tl, bl, br, tr: _r.fill(rec.x, rec.y, rec.width, rec.height, tl, tr, bl, br),
    "draw_rectangle_rounded": lambda rec, roundness, segments, c: _r.rounded(rec, roundness, 0, c),
    "draw_rectangle_lines": lambda x, y, w, h, c: _r.rounded(rl.Rectangle(x, y, w, h), 0, 1, c),
    "draw_rectangle_lines_ex": lambda rec, thick, c: _r.rounded(rec, 0, thick, c),
    "draw_rectangle_rounded_lines_ex": lambda rec, roundness, segments, thick, c: _r.rounded(rec, roundness, thick, c, outside=True),
    "draw_circle": lambda x, y, radius, c: _r.ring(x, y, 0, radius, 0, 360, c, c),
    "draw_circle_v": lambda center, radius, c: _r.ring(center.x, center.y, 0, radius, 0, 360, c, c),
    "draw_circle_gradient": lambda center, radius, inner, outer: _r.ring(center.x, center.y, 0, radius, 0, 360, inner, outer),
    "draw_ring": lambda center, r0, r1, a0, a1, segments, c: _r.ring(center.x, center.y, min(r0, r1), max(r0, r1), a0, a1, c, c),
    "draw_line": lambda x0, y0, x1, y1, c: _r.line(x0 + 0.5, y0 + 0.5, x1 + 0.5, y1 + 0.5, 1, c),
    "draw_line_ex": lambda start, end, thick, c: _r.line(start.x, start.y, end.x, end.y, thick, c),
    "draw_spline_linear": _r.spline,
    "draw_triangle_fan": lambda points, count, c: _r.triangles(points[:count], [_pack(c)], 2),
    "draw_triangle_strip": lambda points, count, c: _r.triangles(points[:count], [_pack(c)], 1),

    "load_texture_from_image": _r.load_texture_from_image,
    "update_texture": _r.update_texture,
    "unload_texture": _r.unload_texture,
    "set_texture_filter": _r.set_texture_filter,
    "set_texture_wrap": noop,
    "gen_texture_mipmaps": noop,
    "draw_texture_pro": _r.draw_texture_pro,
    "draw_texture_ex": lambda t, pos, rotation, scale, tint: _r.draw_texture_pro(
      t, rl.Rectangle(0, 0, t.width, t.height), rl.Rectangle(pos.x, pos.y, t.width * scale, t.height * scale), rl.Vector2(0, 0), rotation, tint),
    "draw_texture": lambda t, x, y, tint: _r.draw_texture_pro(
      t, rl.Rectangle(0, 0, t.width, t.height), rl.Rectangle(x, y, t.width, t.height), rl.Vector2(0, 0), 0, tint),
    "draw_texture_v": lambda t, pos, tint: _r.draw_texture_pro(
      t, rl.Rectangle(0, 0, t.width, t.height), rl.Rectangle(pos.x, pos.y, t.width, t.height), rl.Vector2(0, 0), 0, tint),

    "load_font_ex": _r.load_font_ex,
    "unload_font": _r.unload_font,
    "measure_text_ex": _r.measure_text_ex,
    "draw_text_ex": _r.draw_text_ex,

    # frame shaders have CPU versions at their call sites
    "load_shader_from_memory": lambda vs, fs: rl.Shader(),
    "unload_shader": noop,
    "get_shader_location": lambda shader, name: -1,
    "set_shader_value": noop,
    "set_shader_value_v": noop,
    "set_shader_value_matrix": noop,
    "set_shader_value_texture": noop,
    "begin_shader_mode": noop,
    "end_shader_mode": noop,
  })
