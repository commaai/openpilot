import atexit
import cffi
import os
import time
import signal
import sys
import pyray as rl
import threading
from contextlib import contextmanager
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
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

ENABLE_VSYNC = os.getenv("ENABLE_VSYNC", "0") == "1"
SHOW_FPS = os.getenv("SHOW_FPS") == "1"
SHOW_TOUCHES = os.getenv("SHOW_TOUCHES") == "1"
STRICT_MODE = os.getenv("STRICT_MODE") == "1"
SCALE = float(os.getenv("SCALE", "1.0"))
PROFILE_RENDER = int(os.getenv("PROFILE_RENDER", "0"))
PROFILE_STATS = int(os.getenv("PROFILE_STATS", "100"))  # Number of functions to show in profile output

DEFAULT_TEXT_SIZE = 60
DEFAULT_TEXT_COLOR = rl.WHITE

# Qt draws fonts accounting for ascent/descent differently, so compensate to match old styles
# The real scales for the fonts below range from 1.212 to 1.266
FONT_SCALE = 1.242

ASSETS_DIR = files("openpilot.selfdrive").joinpath("assets")
FONT_DIR = ASSETS_DIR.joinpath("fonts")


class FontWeight(StrEnum):
  LIGHT = "Inter-Light.fnt"
  NORMAL = "Inter-Regular.fnt"
  MEDIUM = "Inter-Medium.fnt"
  SEMI_BOLD = "Inter-SemiBold.fnt"
  BOLD = "Inter-Bold.fnt"
  UNIFONT = "unifont.fnt"


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
      if self._prev_mouse_event[slot] is None or ev[:-1] != self._prev_mouse_event[slot][:-1]:
        with self._lock:
          self._events.append(ev)
        self._prev_mouse_event[slot] = ev


class GuiApplication:
  def __init__(self, width: int, height: int):
    self._fonts: dict[FontWeight, rl.Font] = {}
    self._width = width
    self._height = height

    if PC and os.getenv("SCALE") is None:
      self._scale = self._calculate_auto_scale()
    else:
      self._scale = SCALE

    self._scaled_width = int(self._width * self._scale)
    self._scaled_height = int(self._height * self._scale)
    self._render_texture: rl.RenderTexture | None = None
    self._textures: dict[str, rl.Texture] = {}
    self._target_fps: int = _DEFAULT_FPS
    self._last_fps_log_time: float = time.monotonic()
    self._frame = 0
    self._window_close_requested = False
    self._trace_log_callback = None
    self._modal_overlay = ModalOverlay()
    self._modal_overlay_shown = False

    self._mouse = MouseState(self._scale)
    self._mouse_events: list[MouseEvent] = []
    self._last_mouse_event: MouseEvent = MouseEvent(MousePos(0, 0), 0, False, False, False, 0.0)

    self._should_render = True

    # Debug variables
    self._mouse_history: deque[MousePosWithTime] = deque(maxlen=MOUSE_THREAD_RATE)
    self._show_touches = SHOW_TOUCHES
    self._show_fps = SHOW_FPS
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

      self._set_log_callback()
      rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)

      flags = rl.ConfigFlags.FLAG_MSAA_4X_HINT
      if ENABLE_VSYNC:
        flags |= rl.ConfigFlags.FLAG_VSYNC_HINT
      rl.set_config_flags(flags)

      rl.init_window(self._scaled_width, self._scaled_height, title)
      if self._scale != 1.0:
        rl.set_mouse_scale(1 / self._scale, 1 / self._scale)
        self._render_texture = rl.load_render_texture(self._width, self._height)
        rl.set_texture_filter(self._render_texture.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      rl.set_target_fps(fps)

      self._target_fps = fps
      self._set_styles()
      self._load_fonts()
      self._patch_text_functions()

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

  def set_modal_overlay(self, overlay, callback: Callable | None = None):
    if self._modal_overlay.overlay is not None:
      if self._modal_overlay.callback is not None:
        self._modal_overlay.callback(-1)

    self._modal_overlay = ModalOverlay(overlay=overlay, callback=callback)

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
      # Resize with aspect ratio preservation if requested
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

    rl.unload_image(image)
    return texture

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

    if not PC:
      self._mouse.stop()

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
          yield False
        else:
          yield True

        if self._render_texture:
          rl.end_texture_mode()
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)
          src_rect = rl.Rectangle(0, 0, float(self._width), -float(self._height))
          dst_rect = rl.Rectangle(0, 0, float(self._scaled_width), float(self._scaled_height))
          rl.draw_texture_pro(self._render_texture.texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)

        if self._show_fps:
          rl.draw_fps(10, 10)

        if self._show_touches:
          self._draw_touch_points()

        rl.end_drawing()
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


gui_app = GuiApplication(2160, 1080)
