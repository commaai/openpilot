import atexit
import os
import time
import pyray as rl
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from importlib.resources import as_file, files
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE

DEFAULT_FPS = 60
FPS_LOG_INTERVAL = 5  # Seconds between logging FPS drops
FPS_DROP_THRESHOLD = 0.9  # FPS drop threshold for triggering a warning
FPS_CRITICAL_THRESHOLD = 0.5  # Critical threshold for triggering strict actions

ENABLE_VSYNC = os.getenv("ENABLE_VSYNC", "1") == "1"
SHOW_FPS = os.getenv("SHOW_FPS") == '1'
STRICT_MODE = os.getenv("STRICT_MODE") == '1'
SCALE = float(os.getenv("SCALE", "1.0"))

DEFAULT_TEXT_SIZE = 60
DEFAULT_TEXT_COLOR = rl.WHITE

ASSETS_DIR = files("openpilot.selfdrive").joinpath("assets")
FONT_DIR = ASSETS_DIR.joinpath("fonts")


class FontWeight(IntEnum):
  THIN = 0
  EXTRA_LIGHT = 1
  LIGHT = 2
  NORMAL = 3
  MEDIUM = 4
  SEMI_BOLD = 5
  BOLD = 6
  EXTRA_BOLD = 7
  BLACK = 8


@dataclass
class ModalOverlay:
  overlay: object = None
  callback: Callable | None = None


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

    rl.close_window()

  def render(self):
    try:
      while not (self._window_close_requested or rl.window_should_close()):
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
    font_files = (
      "Inter-Thin.ttf",
      "Inter-ExtraLight.ttf",
      "Inter-Light.ttf",
      "Inter-Regular.ttf",
      "Inter-Medium.ttf",
      "Inter-SemiBold.ttf",
      "Inter-Bold.ttf",
      "Inter-ExtraBold.ttf",
      "Inter-Black.ttf",
    )

    # Create a character set from our keyboard layouts
    from openpilot.system.ui.widgets.keyboard import KEYBOARD_LAYOUTS
    all_chars = set()
    for layout in KEYBOARD_LAYOUTS.values():
      all_chars.update(key for row in layout for key in row)
    all_chars = "".join(all_chars)
    all_chars += "–✓"

    codepoint_count = rl.ffi.new("int *", 1)
    codepoints = rl.load_codepoints(all_chars, codepoint_count)

    for index, font_file in enumerate(font_files):
      with as_file(FONT_DIR.joinpath(font_file)) as fspath:
        font = rl.load_font_ex(fspath.as_posix(), 200, codepoints, codepoint_count[0])
        rl.set_texture_filter(font.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self._fonts[index] = font

    rl.unload_codepoints(codepoints)
    rl.gui_set_font(self._fonts[FontWeight.NORMAL])

  def _set_styles(self):
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BORDER_WIDTH, 0)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, DEFAULT_TEXT_SIZE)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.BACKGROUND_COLOR, rl.color_to_int(rl.BLACK))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(DEFAULT_TEXT_COLOR))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BASE_COLOR_NORMAL, rl.color_to_int(rl.Color(50, 50, 50, 255)))

  def _set_log_callback(self):
    @rl.ffi.callback("void(int, char *, void *)")
    def trace_log_callback(log_level, text, args):
      try:
        text_str = rl.ffi.string(text).decode('utf-8')
      except (TypeError, UnicodeDecodeError):
        text_str = str(text)

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
