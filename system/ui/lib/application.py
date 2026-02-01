import atexit
import cffi
import os
import queue
import time
import signal
import sys
import pyray as rl
import threading
import platform
import subprocess
from contextlib import contextmanager
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple
from importlib.resources import as_file, files
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE, PC
from openpilot.system.ui.lib.multilang import multilang
from openpilot.common.realtime import Ratekeeper

_DEFAULT_FPS = int(os.getenv("FPS", {'tizi': 20}.get(HARDWARE.get_device_type(), 60)))
FPS_LOG_INTERVAL = 5  # Seconds between logging FPS drops
FPS_DROP_THRESHOLD = 0.9  # FPS drop threshold for triggering a warning
FPS_CRITICAL_THRESHOLD = 0.5  # Critical threshold for triggering strict actions
MOUSE_THREAD_RATE = 140  # touch controller runs at 140Hz
MAX_TOUCH_SLOTS = 2
TOUCH_HISTORY_TIMEOUT = 3.0  # Seconds before touch points fade out

BIG_UI = os.getenv("BIG", "0") == "1"
ENABLE_VSYNC = os.getenv("ENABLE_VSYNC", "0") == "1"
SHOW_FPS = os.getenv("SHOW_FPS") == "1"
SHOW_TOUCHES = os.getenv("SHOW_TOUCHES") == "1"
STRICT_MODE = os.getenv("STRICT_MODE") == "1"
SCALE = float(os.getenv("SCALE", "1.0"))
GRID_SIZE = int(os.getenv("GRID", "0"))
PROFILE_RENDER = int(os.getenv("PROFILE_RENDER", "0"))
PROFILE_STATS = int(os.getenv("PROFILE_STATS", "100"))  # Number of functions to show in profile output
RECORD = os.getenv("RECORD") == "1"
RECORD_OUTPUT = str(Path(os.getenv("RECORD_OUTPUT", "output")).with_suffix(".mp4"))
RECORD_BITRATE = os.getenv("RECORD_BITRATE", "")  # Target bitrate e.g. "2000k"
RECORD_SPEED = int(os.getenv("RECORD_SPEED", "1"))  # Speed multiplier
OFFSCREEN = os.getenv("OFFSCREEN") == "1"  # Disable FPS limiting for fast offline rendering

GL_VERSION = """
#version 300 es
precision highp float;
"""
if platform.system() == "Darwin":
  GL_VERSION = """
    #version 330 core
  """

BURN_IN_MODE = "BURN_IN" in os.environ
BURN_IN_VERTEX_SHADER = GL_VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
uniform mat4 mvp;
out vec2 fragTexCoord;
void main() {
  fragTexCoord = vertexTexCoord;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""
BURN_IN_FRAGMENT_SHADER = GL_VERSION + """
in vec2 fragTexCoord;
uniform sampler2D texture0;
out vec4 fragColor;
void main() {
  vec4 sampled = texture(texture0, fragTexCoord);
  float intensity = sampled.b;
  // Map blue intensity to green -> yellow -> red to highlight burn-in risk.
  vec3 start = vec3(0.0, 1.0, 0.0);
  vec3 middle = vec3(1.0, 1.0, 0.0);
  vec3 end = vec3(1.0, 0.0, 0.0);
  vec3 gradient = mix(start, middle, clamp(intensity * 2.0, 0.0, 1.0));
  gradient = mix(gradient, end, clamp((intensity - 0.5) * 2.0, 0.0, 1.0));
  fragColor = vec4(gradient, sampled.a);
}
"""

DEFAULT_TEXT_SIZE = 60
DEFAULT_TEXT_COLOR = rl.Color(255, 255, 255, int(255 * 0.9))

# Qt draws fonts accounting for ascent/descent differently, so compensate to match old styles
# The real scales for the fonts below range from 1.212 to 1.266
FONT_SCALE = 1.242 if BIG_UI else 1.16

ASSETS_DIR = files("openpilot.selfdrive").joinpath("assets")
FONT_DIR = ASSETS_DIR.joinpath("fonts")


class FontWeight(StrEnum):
  LIGHT = "Inter-Light.fnt"
  NORMAL = "Inter-Regular.fnt" if BIG_UI else "Inter-Medium.fnt"
  MEDIUM = "Inter-Medium.fnt"
  BOLD = "Inter-Bold.fnt"
  SEMI_BOLD = "Inter-SemiBold.fnt"
  UNIFONT = "unifont.fnt"

  # Small UI fonts
  DISPLAY_REGULAR = "Inter-Regular.fnt"
  ROMAN = "Inter-Regular.fnt"
  DISPLAY = "Inter-Bold.fnt"


def font_fallback(font: rl.Font) -> rl.Font:
  """Fall back to unifont for languages that require it."""
  if multilang.requires_unifont():
    return gui_app.font(FontWeight.UNIFONT)
  return font


@dataclass
class ModalOverlay:
  overlay: object = None
  callback: Callable | None = None


class MousePos(NamedTuple):
  x: float
  y: float


class MousePosWithTime(NamedTuple):
  x: float
  y: float
  t: float


class MouseEvent(NamedTuple):
  pos: MousePos
  slot: int
  left_pressed: bool
  left_released: bool
  left_down: bool
  t: float


class MouseState:
  def __init__(self, scale: float = 1.0):
    self._scale = scale
    self._events: deque[MouseEvent] = deque(maxlen=MOUSE_THREAD_RATE)  # bound event list
    self._prev_mouse_event: list[MouseEvent | None] = [None] * MAX_TOUCH_SLOTS

    self._rk = Ratekeeper(MOUSE_THREAD_RATE, print_delay_threshold=None)
    self._lock = threading.Lock()
    self._exit_event = threading.Event()
    self._thread = None

  def get_events(self) -> list[MouseEvent]:
    with self._lock:
      events = list(self._events)
      self._events.clear()
    return events

  def start(self):
    self._exit_event.clear()
    if self._thread is None or not self._thread.is_alive():
      self._thread = threading.Thread(target=self._run_thread, daemon=True)
      self._thread.start()

  def stop(self):
    self._exit_event.set()
    if self._thread is not None and self._thread.is_alive():
      self._thread.join()

  def _run_thread(self):
    while not self._exit_event.is_set():
      rl.poll_input_events()
      self._handle_mouse_event()
      self._rk.keep_time()

  def _handle_mouse_event(self):
    for slot in range(MAX_TOUCH_SLOTS):
      mouse_pos = rl.get_touch_position(slot)
      x = mouse_pos.x / self._scale if self._scale != 1.0 else mouse_pos.x
      y = mouse_pos.y / self._scale if self._scale != 1.0 else mouse_pos.y
      ev = MouseEvent(
        MousePos(x, y),
        slot,
        rl.is_mouse_button_pressed(slot),  # noqa: TID251
        rl.is_mouse_button_released(slot),  # noqa: TID251
        rl.is_mouse_button_down(slot),
        time.monotonic(),
      )
      # Only add changes
      prev = self._prev_mouse_event[slot]
      if prev is None or ev[:-1] != prev[:-1]:
        with self._lock:
          self._events.append(ev)
        self._prev_mouse_event[slot] = ev


class GuiApplication:
  def __init__(self, width: int | None = None, height: int | None = None):
    self._set_log_callback()

    self._fonts: dict[FontWeight, rl.Font] = {}
    self._width = width if width is not None else GuiApplication._default_width()
    self._height = height if height is not None else GuiApplication._default_height()

    if PC and os.getenv("SCALE") is None:
      self._scale = self._calculate_auto_scale()
    else:
      self._scale = SCALE

    # Scale, then ensure dimensions are even
    self._scaled_width = int(self._width * self._scale)
    self._scaled_height = int(self._height * self._scale)
    self._scaled_width += self._scaled_width % 2
    self._scaled_height += self._scaled_height % 2

    self._render_texture: rl.RenderTexture | None = None
    self._burn_in_shader: rl.Shader | None = None
    self._ffmpeg_proc: subprocess.Popen | None = None
    self._ffmpeg_queue: queue.Queue | None = None
    self._ffmpeg_thread: threading.Thread | None = None
    self._ffmpeg_stop_event: threading.Event | None = None
    self._textures: dict[str, rl.Texture] = {}
    self._target_fps: int = _DEFAULT_FPS
    self._last_fps_log_time: float = time.monotonic()
    self._frame = 0
    self._window_close_requested = False
    self._modal_overlay = ModalOverlay()
    self._modal_overlay_shown = False
    self._modal_overlay_tick: Callable[[], None] | None = None

    self._mouse = MouseState(self._scale)
    self._mouse_events: list[MouseEvent] = []
    self._last_mouse_event: MouseEvent = MouseEvent(MousePos(0, 0), 0, False, False, False, 0.0)

    self._should_render = True

    # Debug variables
    self._mouse_history: deque[MousePosWithTime] = deque(maxlen=MOUSE_THREAD_RATE)
    self._show_touches = SHOW_TOUCHES
    self._show_fps = SHOW_FPS
    self._grid_size = GRID_SIZE
    self._profile_render_frames = PROFILE_RENDER
    self._render_profiler = None
    self._render_profile_start_time = None

  @property
  def frame(self):
    return self._frame

  def set_show_touches(self, show: bool):
    self._show_touches = show

  def set_show_fps(self, show: bool):
    self._show_fps = show

  @property
  def target_fps(self):
    return self._target_fps

  def request_close(self):
    self._window_close_requested = True

  def init_window(self, title: str, fps: int = _DEFAULT_FPS):
    with self._startup_profile_context():
      def _close(sig, frame):
        self.close()
        sys.exit(0)
      signal.signal(signal.SIGINT, _close)
      atexit.register(self.close)

      flags = rl.ConfigFlags.FLAG_MSAA_4X_HINT
      if ENABLE_VSYNC:
        flags |= rl.ConfigFlags.FLAG_VSYNC_HINT
      rl.set_config_flags(flags)

      rl.init_window(self._scaled_width, self._scaled_height, title)

      needs_render_texture = self._scale != 1.0 or BURN_IN_MODE or RECORD
      if self._scale != 1.0:
        rl.set_mouse_scale(1 / self._scale, 1 / self._scale)
      if needs_render_texture:
        self._render_texture = rl.load_render_texture(self._width, self._height)
        rl.set_texture_filter(self._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

      if RECORD:
        output_fps = fps * RECORD_SPEED
        ffmpeg_args = [
          'ffmpeg',
          '-v', 'warning',          # Reduce ffmpeg log spam
          '-nostats',               # Suppress encoding progress
          '-f', 'rawvideo',         # Input format
          '-pix_fmt', 'rgba',       # Input pixel format
          '-s', f'{self._width}x{self._height}',  # Input resolution
          '-r', str(fps),           # Input frame rate
          '-i', 'pipe:0',           # Input from stdin
          '-vf', 'vflip,format=yuv420p',  # Flip vertically and convert to yuv420p
          '-r', str(output_fps),    # Output frame rate (for speed multiplier)
          '-c:v', 'libx264',
          '-preset', 'ultrafast',
        ]
        if RECORD_BITRATE:
          ffmpeg_args += ['-b:v', RECORD_BITRATE, '-maxrate', RECORD_BITRATE, '-bufsize', RECORD_BITRATE]
        ffmpeg_args += [
          '-y',                     # Overwrite existing file
          '-f', 'mp4',              # Output format
          RECORD_OUTPUT,            # Output file path
        ]
        self._ffmpeg_proc = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE)
        self._ffmpeg_queue = queue.Queue(maxsize=60)  # Buffer up to 60 frames
        self._ffmpeg_stop_event = threading.Event()
        self._ffmpeg_thread = threading.Thread(target=self._ffmpeg_writer_thread, daemon=True)
        self._ffmpeg_thread.start()

      # OFFSCREEN disables FPS limiting for fast offline rendering (e.g. clips)
      rl.set_target_fps(0 if OFFSCREEN else fps)

      self._target_fps = fps
      self._set_styles()
      self._load_fonts()
      self._patch_text_functions()
      if BURN_IN_MODE and self._burn_in_shader is None:
        self._burn_in_shader = rl.load_shader_from_memory(BURN_IN_VERTEX_SHADER, BURN_IN_FRAGMENT_SHADER)

      if not PC:
        self._mouse.start()

  @contextmanager
  def _startup_profile_context(self):
    if "PROFILE_STARTUP" not in os.environ:
      yield
      return

    import cProfile
    import io
    import pstats

    profiler = cProfile.Profile()
    start_time = time.monotonic()
    profiler.enable()

    # do the init
    yield

    profiler.disable()
    elapsed_ms = (time.monotonic() - start_time) * 1e3

    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).sort_stats("cumtime").print_stats(25)
    print("\n=== Startup profile ===")
    print(stats_stream.getvalue().rstrip())

    green = "\033[92m"
    reset = "\033[0m"
    print(f"{green}UI window ready in {elapsed_ms:.1f} ms{reset}")
    sys.exit(0)

  def _ffmpeg_writer_thread(self):
    """Background thread that writes frames to ffmpeg."""
    while True:
      try:
        data = self._ffmpeg_queue.get(timeout=1.0)
        if data is None:  # Sentinel to stop
          break
        self._ffmpeg_proc.stdin.write(data)
      except queue.Empty:
        if self._ffmpeg_stop_event.is_set():
          break
        continue
      except Exception:
        break

  def set_modal_overlay(self, overlay, callback: Callable | None = None):
    if self._modal_overlay.overlay is not None:
      if hasattr(self._modal_overlay.overlay, 'hide_event'):
        self._modal_overlay.overlay.hide_event()

      if self._modal_overlay.callback is not None:
        self._modal_overlay.callback(-1)

    self._modal_overlay = ModalOverlay(overlay=overlay, callback=callback)

  def set_modal_overlay_tick(self, tick_function: Callable | None):
    self._modal_overlay_tick = tick_function

  def set_should_render(self, should_render: bool):
    self._should_render = should_render

  def texture(self, asset_path: str, width: int | None = None, height: int | None = None,
              alpha_premultiply=False, keep_aspect_ratio=True):
    cache_key = f"{asset_path}_{width}_{height}_{alpha_premultiply}{keep_aspect_ratio}"
    if cache_key in self._textures:
      return self._textures[cache_key]

    with as_file(ASSETS_DIR.joinpath(asset_path)) as fspath:
      image_obj = self._load_image_from_path(fspath.as_posix(), width, height, alpha_premultiply, keep_aspect_ratio)
      texture_obj = self._load_texture_from_image(image_obj)
    self._textures[cache_key] = texture_obj
    return texture_obj

  def _load_image_from_path(self, image_path: str, width: int | None = None, height: int | None = None,
                            alpha_premultiply: bool = False, keep_aspect_ratio: bool = True) -> rl.Image:
    """Load and resize an image, storing it for later automatic unloading."""
    image = rl.load_image(image_path)

    if alpha_premultiply:
      rl.image_alpha_premultiply(image)

    if width is not None and height is not None:
      same_dimensions = image.width == width and image.height == height

      # Resize with aspect ratio preservation if requested
      if not same_dimensions:
        if keep_aspect_ratio:
          orig_width = image.width
          orig_height = image.height

          scale_width = width / orig_width
          scale_height = height / orig_height

          # Calculate new dimensions
          scale = min(scale_width, scale_height)
          new_width = int(orig_width * scale)
          new_height = int(orig_height * scale)

          rl.image_resize(image, new_width, new_height)
        else:
          rl.image_resize(image, width, height)
    else:
      assert keep_aspect_ratio, "Cannot resize without specifying width and height"
    return image

  def _load_texture_from_image(self, image: rl.Image) -> rl.Texture:
    """Send image to GPU and unload original image."""
    texture = rl.load_texture_from_image(image)
    # Set texture filtering to smooth the result
    rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    # prevent artifacts from wrapping coordinates
    rl.set_texture_wrap(texture, rl.TextureWrap.TEXTURE_WRAP_CLAMP)

    rl.unload_image(image)
    return texture

  def close_ffmpeg(self):
    if self._ffmpeg_thread is not None:
      # Signal thread to stop, send sentinel, then wait for it to drain
      self._ffmpeg_stop_event.set()
      self._ffmpeg_queue.put(None)
      self._ffmpeg_thread.join(timeout=30)

    if self._ffmpeg_proc is not None:
      self._ffmpeg_proc.stdin.flush()
      self._ffmpeg_proc.stdin.close()
      try:
        self._ffmpeg_proc.wait(timeout=30)
      except subprocess.TimeoutExpired:
        self._ffmpeg_proc.terminate()
        self._ffmpeg_proc.wait()

  def close(self):
    if not rl.is_window_ready():
      return

    for texture in self._textures.values():
      rl.unload_texture(texture)
    self._textures = {}

    for font in self._fonts.values():
      rl.unload_font(font)
    self._fonts = {}

    if self._render_texture is not None:
      rl.unload_render_texture(self._render_texture)
      self._render_texture = None

    if self._burn_in_shader:
      rl.unload_shader(self._burn_in_shader)
      self._burn_in_shader = None

    if not PC:
      self._mouse.stop()

    self.close_ffmpeg()

    rl.close_window()

  @property
  def mouse_events(self) -> list[MouseEvent]:
    return self._mouse_events

  @property
  def last_mouse_event(self) -> MouseEvent:
    return self._last_mouse_event

  def render(self):
    try:
      if self._profile_render_frames > 0:
        import cProfile
        self._render_profiler = cProfile.Profile()
        self._render_profile_start_time = time.monotonic()
        self._render_profiler.enable()

      while not (self._window_close_requested or rl.window_should_close()):
        if PC:
          # Thread is not used on PC, need to manually add mouse events
          self._mouse._handle_mouse_event()

        # Store all mouse events for the current frame
        self._mouse_events = self._mouse.get_events()
        if len(self._mouse_events) > 0:
          self._last_mouse_event = self._mouse_events[-1]

        # Skip rendering when screen is off
        if not self._should_render:
          if PC:
            rl.poll_input_events()
          time.sleep(1 / self._target_fps)
          yield False
          continue

        if self._render_texture:
          rl.begin_texture_mode(self._render_texture)
          rl.clear_background(rl.BLACK)
        else:
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)

        # Handle modal overlay rendering and input processing
        if self._handle_modal_overlay():
          # Allow a Widget to still run a function while overlay is shown
          if self._modal_overlay_tick is not None:
            self._modal_overlay_tick()
          yield False
        else:
          yield True

        if self._render_texture:
          rl.end_texture_mode()
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)
          src_rect = rl.Rectangle(0, 0, float(self._width), -float(self._height))
          dst_rect = rl.Rectangle(0, 0, float(self._scaled_width), float(self._scaled_height))
          texture = self._render_texture.texture
          if texture:
            if BURN_IN_MODE and self._burn_in_shader:
              rl.begin_shader_mode(self._burn_in_shader)
              rl.draw_texture_pro(texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)
              rl.end_shader_mode()
            else:
              rl.draw_texture_pro(texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)

        if self._show_fps:
          rl.draw_fps(10, 10)

        if self._show_touches:
          self._draw_touch_points()

        if self._grid_size > 0:
          self._draw_grid()

        rl.end_drawing()

        if RECORD:
          image = rl.load_image_from_texture(self._render_texture.texture)
          data_size = image.width * image.height * 4
          data = bytes(rl.ffi.buffer(image.data, data_size))
          self._ffmpeg_queue.put(data)  # Async write via background thread
          rl.unload_image(image)

        self._monitor_fps()
        self._frame += 1

        if self._profile_render_frames > 0 and self._frame >= self._profile_render_frames:
          self._output_render_profile()
    except KeyboardInterrupt:
      pass

  def font(self, font_weight: FontWeight = FontWeight.NORMAL) -> rl.Font:
    return self._fonts[font_weight]

  @property
  def width(self):
    return self._width

  @property
  def height(self):
    return self._height

  def _handle_modal_overlay(self) -> bool:
    if self._modal_overlay.overlay:
      if hasattr(self._modal_overlay.overlay, 'render'):
        result = self._modal_overlay.overlay.render(rl.Rectangle(0, 0, self.width, self.height))
      elif callable(self._modal_overlay.overlay):
        result = self._modal_overlay.overlay()
      else:
        raise Exception

      # Send show event to Widget
      if not self._modal_overlay_shown and hasattr(self._modal_overlay.overlay, 'show_event'):
        self._modal_overlay.overlay.show_event()
        self._modal_overlay_shown = True

      if result >= 0:
        # Clear the overlay and execute the callback
        original_modal = self._modal_overlay
        self._modal_overlay = ModalOverlay()
        if hasattr(original_modal.overlay, 'hide_event'):
          original_modal.overlay.hide_event()
        if original_modal.callback is not None:
          original_modal.callback(result)
      return True
    else:
      self._modal_overlay_shown = False
      return False

  def _load_fonts(self):
    for font_weight_file in FontWeight:
      with as_file(FONT_DIR) as fspath:
        fnt_path = fspath / font_weight_file
        font = rl.load_font(fnt_path.as_posix())
        if font_weight_file != FontWeight.UNIFONT:
          rl.set_texture_filter(font.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self._fonts[font_weight_file] = font
    rl.gui_set_font(self._fonts[FontWeight.NORMAL])

  def _set_styles(self):
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BORDER_WIDTH, 0)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, DEFAULT_TEXT_SIZE)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.BACKGROUND_COLOR, rl.color_to_int(rl.BLACK))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(DEFAULT_TEXT_COLOR))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BASE_COLOR_NORMAL, rl.color_to_int(rl.Color(50, 50, 50, 255)))

  def _patch_text_functions(self):
    # Wrap pyray text APIs to apply a global text size scale so our px sizes match Qt
    if not hasattr(rl, "_orig_draw_text_ex"):
      rl._orig_draw_text_ex = rl.draw_text_ex

    def _draw_text_ex_scaled(font, text, position, font_size, spacing, tint):
      font = font_fallback(font)
      return rl._orig_draw_text_ex(font, text, position, font_size * FONT_SCALE, spacing, tint)

    rl.draw_text_ex = _draw_text_ex_scaled

  def _set_log_callback(self):
    ffi_libc = cffi.FFI()
    ffi_libc.cdef("""
      int vasprintf(char **strp, const char *fmt, void *ap);
      void free(void *ptr);
    """)
    libc = ffi_libc.dlopen(None)

    @rl.ffi.callback("void(int, char *, void *)")
    def trace_log_callback(log_level, text, args):
      try:
        text_addr = int(rl.ffi.cast("uintptr_t", text))
        args_addr = int(rl.ffi.cast("uintptr_t", args))
        text_libc = ffi_libc.cast("char *", text_addr)
        args_libc = ffi_libc.cast("void *", args_addr)

        out = ffi_libc.new("char **")
        if libc.vasprintf(out, text_libc, args_libc) >= 0 and out[0] != ffi_libc.NULL:
          text_str = ffi_libc.string(out[0]).decode("utf-8", "replace")
          libc.free(out[0])
        else:
          text_str = rl.ffi.string(text).decode("utf-8", "replace")
      except Exception as e:
        text_str = f"[Log decode error: {e}]"

      if log_level == rl.TraceLogLevel.LOG_ERROR:
        cloudlog.error(f"raylib: {text_str}")
      elif log_level == rl.TraceLogLevel.LOG_WARNING:
        cloudlog.warning(f"raylib: {text_str}")
      elif log_level == rl.TraceLogLevel.LOG_INFO:
        cloudlog.info(f"raylib: {text_str}")
      elif log_level == rl.TraceLogLevel.LOG_DEBUG:
        cloudlog.debug(f"raylib: {text_str}")
      else:
        cloudlog.error(f"raylib: Unknown level {log_level}: {text_str}")

    # ensure we get all the logs forwarded to us
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_DEBUG)

    # Store callback reference
    self._trace_log_callback = trace_log_callback
    rl.set_trace_log_callback(self._trace_log_callback)

  def _monitor_fps(self):
    fps = rl.get_fps()

    # Log FPS drop below threshold at regular intervals
    if fps < self._target_fps * FPS_DROP_THRESHOLD:
      current_time = time.monotonic()
      if current_time - self._last_fps_log_time >= FPS_LOG_INTERVAL:
        cloudlog.warning(f"FPS dropped below {self._target_fps}: {fps}")
        self._last_fps_log_time = current_time

    # Strict mode: terminate UI if FPS drops too much
    if STRICT_MODE and fps < self._target_fps * FPS_CRITICAL_THRESHOLD:
      cloudlog.error(f"FPS dropped critically below {fps}. Shutting down UI.")
      self.close_ffmpeg()
      os._exit(1)

  def _draw_touch_points(self):
    current_time = time.monotonic()

    for mouse_event in self._mouse_events:
      if mouse_event.left_pressed:
        self._mouse_history.clear()
      self._mouse_history.append(MousePosWithTime(mouse_event.pos.x * self._scale, mouse_event.pos.y * self._scale, current_time))

    # Remove old touch points that exceed the timeout
    while self._mouse_history and (current_time - self._mouse_history[0].t) > TOUCH_HISTORY_TIMEOUT:
      self._mouse_history.popleft()

    if self._mouse_history:
      mouse_pos = self._mouse_history[-1]
      rl.draw_circle(int(mouse_pos.x), int(mouse_pos.y), 15, rl.RED)
      for idx, mouse_pos in enumerate(self._mouse_history):
        perc = idx / len(self._mouse_history)
        color = rl.Color(min(int(255 * (1.5 - perc)), 255), int(min(255 * (perc + 0.5), 255)), 50, 255)
        rl.draw_circle(int(mouse_pos.x), int(mouse_pos.y), 5, color)

  def _draw_grid(self):
    grid_color = rl.Color(60, 60, 60, 255)
    # Draw vertical lines
    x = 0
    while x <= self._scaled_width:
      rl.draw_line(x, 0, x, self._scaled_height, grid_color)
      x += self._grid_size
    # Draw horizontal lines
    y = 0
    while y <= self._scaled_height:
      rl.draw_line(0, y, self._scaled_width, y, grid_color)
      y += self._grid_size

  def _output_render_profile(self):
    import io
    import pstats

    self._render_profiler.disable()
    elapsed_ms = (time.monotonic() - self._render_profile_start_time) * 1e3
    avg_frame_time = elapsed_ms / self._frame if self._frame > 0 else 0

    stats_stream = io.StringIO()
    pstats.Stats(self._render_profiler, stream=stats_stream).sort_stats("cumtime").print_stats(PROFILE_STATS)
    print("\n=== Render loop profile ===")
    print(stats_stream.getvalue().rstrip())

    green = "\033[92m"
    reset = "\033[0m"
    print(f"\n{green}Rendered {self._frame} frames in {elapsed_ms:.1f} ms{reset}")
    print(f"{green}Average frame time: {avg_frame_time:.2f} ms ({1000/avg_frame_time:.1f} FPS){reset}")
    sys.exit(0)

  def _calculate_auto_scale(self) -> float:
     # Create temporary window to query monitor info
    rl.init_window(1, 1, "")
    w, h = rl.get_monitor_width(0), rl.get_monitor_height(0)
    rl.close_window()

    if w == 0 or h == 0 or (w >= self._width and h >= self._height):
      return 1.0

    # Apply 0.95 factor for window decorations/taskbar margin
    return max(0.3, min(w / self._width, h / self._height) * 0.95)

  @staticmethod
  def _default_width() -> int:
    return 2160 if GuiApplication.big_ui() else 536

  @staticmethod
  def _default_height() -> int:
    return 1080 if GuiApplication.big_ui() else 240

  @staticmethod
  def big_ui() -> bool:
    return HARDWARE.get_device_type() in ('tici', 'tizi') or BIG_UI


gui_app = GuiApplication()
