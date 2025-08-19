import atexit
import cffi
import os
import time
import pyray as rl
import threading
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple
from importlib.resources import as_file, files
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE, PC
from openpilot.common.realtime import Ratekeeper

DEFAULT_FPS = int(os.getenv("FPS", "60"))
FPS_LOG_INTERVAL = 5  # Seconds between logging FPS drops
FPS_DROP_THRESHOLD = 0.9  # FPS drop threshold for triggering a warning
FPS_CRITICAL_THRESHOLD = 0.5  # Critical threshold for triggering strict actions
MOUSE_THREAD_RATE = 140  # touch controller runs at 140Hz
MAX_TOUCH_SLOTS = 2

ENABLE_VSYNC = os.getenv("ENABLE_VSYNC", "0") == "1"
SHOW_FPS = os.getenv("SHOW_FPS") == "1"
SHOW_TOUCHES = os.getenv("SHOW_TOUCHES") == "1"
STRICT_MODE = os.getenv("STRICT_MODE") == "1"
SCALE = float(os.getenv("SCALE", "1.0"))

DEFAULT_TEXT_SIZE = 60
DEFAULT_TEXT_COLOR = rl.WHITE

ASSETS_DIR = files("openpilot.selfdrive").joinpath("assets")
FONT_DIR = ASSETS_DIR.joinpath("fonts")


class FontWeight(StrEnum):
  THIN = "Inter-Thin.ttf"
  EXTRA_LIGHT = "Inter-ExtraLight.ttf"
  LIGHT = "Inter-Light.ttf"
  NORMAL = "Inter-Regular.ttf"
  MEDIUM = "Inter-Medium.ttf"
  SEMI_BOLD = "Inter-SemiBold.ttf"
  BOLD = "Inter-Bold.ttf"
  EXTRA_BOLD = "Inter-ExtraBold.ttf"
  BLACK = "Inter-Black.ttf"


@dataclass
class ModalOverlay:
  overlay: object = None
  callback: Callable | None = None


class MousePos(NamedTuple):
  x: float
  y: float


class MouseEvent(NamedTuple):
  pos: MousePos
  slot: int
  left_pressed: bool
  left_released: bool
  left_down: bool
  t: float


class MouseState:
  def __init__(self):
    self._events: deque[MouseEvent] = deque(maxlen=MOUSE_THREAD_RATE)  # bound event list
    self._prev_mouse_event: list[MouseEvent | None] = [None] * MAX_TOUCH_SLOTS

    self._rk = Ratekeeper(MOUSE_THREAD_RATE)
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
      ev = MouseEvent(
        MousePos(mouse_pos.x, mouse_pos.y),
        slot,
        rl.is_mouse_button_pressed(slot),
        rl.is_mouse_button_released(slot),
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
    self._scale = SCALE
    self._scaled_width = int(self._width * self._scale)
    self._scaled_height = int(self._height * self._scale)
    self._render_texture: rl.RenderTexture | None = None
    self._textures: dict[str, rl.Texture] = {}
    self._target_fps: int = DEFAULT_FPS
    self._last_fps_log_time: float = time.monotonic()
    self._window_close_requested = False
    self._trace_log_callback = None
    self._modal_overlay = ModalOverlay()

    self._mouse = MouseState()
    self._mouse_events: list[MouseEvent] = []

    # Debug variables
    self._mouse_history: deque[MousePos] = deque(maxlen=MOUSE_THREAD_RATE)

  def request_close(self):
    self._window_close_requested = True

  def init_window(self, title: str, fps: int = DEFAULT_FPS):
    atexit.register(self.close)  # Automatically call close() on exit

    HARDWARE.set_display_power(True)
    HARDWARE.set_screen_brightness(65)

    self._set_log_callback()
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_ALL)

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

    if not PC:
      self._mouse.start()

  def set_modal_overlay(self, overlay, callback: Callable | None = None):
    self._modal_overlay = ModalOverlay(overlay=overlay, callback=callback)

  def texture(self, asset_path: str, width: int, height: int, alpha_premultiply=False, keep_aspect_ratio=True):
    cache_key = f"{asset_path}_{width}_{height}_{alpha_premultiply}{keep_aspect_ratio}"
    if cache_key in self._textures:
      return self._textures[cache_key]

    with as_file(ASSETS_DIR.joinpath(asset_path)) as fspath:
      texture_obj = self._load_texture_from_image(fspath.as_posix(), width, height, alpha_premultiply, keep_aspect_ratio)
    self._textures[cache_key] = texture_obj
    return texture_obj

  def _load_texture_from_image(self, image_path: str, width: int, height: int, alpha_premultiply=False, keep_aspect_ratio=True):
    """Load and resize a texture, storing it for later automatic unloading."""
    image = rl.load_image(image_path)

    if alpha_premultiply:
      rl.image_alpha_premultiply(image)

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

  def render(self):
    try:
      while not (self._window_close_requested or rl.window_should_close()):
        if PC:
          # Thread is not used on PC, need to manually add mouse events
          self._mouse._handle_mouse_event()

        # Store all mouse events for the current frame
        self._mouse_events = self._mouse.get_events()

        if self._render_texture:
          rl.begin_texture_mode(self._render_texture)
          rl.clear_background(rl.BLACK)
        else:
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)

        # Handle modal overlay rendering and input processing
        if self._modal_overlay.overlay:
          if hasattr(self._modal_overlay.overlay, 'render'):
            result = self._modal_overlay.overlay.render(rl.Rectangle(0, 0, self.width, self.height))
          elif callable(self._modal_overlay.overlay):
            result = self._modal_overlay.overlay()
          else:
            raise Exception

          if result >= 0:
            # Execute callback with the result and clear the overlay
            if self._modal_overlay.callback is not None:
              self._modal_overlay.callback(result)

            self._modal_overlay = ModalOverlay()
        else:
          yield

        if self._render_texture:
          rl.end_texture_mode()
          rl.begin_drawing()
          rl.clear_background(rl.BLACK)
          src_rect = rl.Rectangle(0, 0, float(self._width), -float(self._height))
          dst_rect = rl.Rectangle(0, 0, float(self._scaled_width), float(self._scaled_height))
          rl.draw_texture_pro(self._render_texture.texture, src_rect, dst_rect, rl.Vector2(0, 0), 0.0, rl.WHITE)

        if SHOW_FPS:
          rl.draw_fps(10, 10)

        if SHOW_TOUCHES:
          for mouse_event in self._mouse_events:
            if mouse_event.left_pressed:
              self._mouse_history.clear()
            self._mouse_history.append(mouse_event.pos)

          if self._mouse_history:
            mouse_pos = self._mouse_history[-1]
            rl.draw_circle(int(mouse_pos.x), int(mouse_pos.y), 15, rl.RED)
            for idx, mouse_pos in enumerate(self._mouse_history):
              perc = idx / len(self._mouse_history)
              color = rl.Color(min(int(255 * (1.5 - perc)), 255), int(min(255 * (perc + 0.5), 255)), 50, 255)
              rl.draw_circle(int(mouse_pos.x), int(mouse_pos.y), 5, color)

        rl.end_drawing()
        self._monitor_fps()
    except KeyboardInterrupt:
      pass

  def font(self, font_weight: FontWeight = FontWeight.NORMAL):
    return self._fonts[font_weight]

  @property
  def width(self):
    return self._width

  @property
  def height(self):
    return self._height

  def _load_fonts(self):
    # Create a character set from our keyboard layouts
    from openpilot.system.ui.widgets.keyboard import KEYBOARD_LAYOUTS

    all_chars = set()
    for layout in KEYBOARD_LAYOUTS.values():
      all_chars.update(key for row in layout for key in row)
    all_chars = "".join(all_chars)
    all_chars += "–✓×°"

    codepoint_count = rl.ffi.new("int *", 1)
    codepoints = rl.load_codepoints(all_chars, codepoint_count)

    for font_weight_file in FontWeight:
      with as_file(FONT_DIR.joinpath(font_weight_file)) as fspath:
        font = rl.load_font_ex(fspath.as_posix(), 200, codepoints, codepoint_count[0])
        rl.set_texture_filter(font.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self._fonts[font_weight_file] = font

    rl.unload_codepoints(codepoints)
    rl.gui_set_font(self._fonts[FontWeight.NORMAL])

  def _set_styles(self):
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BORDER_WIDTH, 0)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, DEFAULT_TEXT_SIZE)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.BACKGROUND_COLOR, rl.color_to_int(rl.BLACK))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(DEFAULT_TEXT_COLOR))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BASE_COLOR_NORMAL, rl.color_to_int(rl.Color(50, 50, 50, 255)))

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


gui_app = GuiApplication(2160, 1080)
