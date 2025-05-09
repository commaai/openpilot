import atexit
import os
import time
import pyray as rl
from enum import IntEnum
from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE

DEFAULT_FPS = 60
FPS_LOG_INTERVAL = 5  # Seconds between logging FPS drops
FPS_DROP_THRESHOLD = 0.9  # FPS drop threshold for triggering a warning
FPS_CRITICAL_THRESHOLD = 0.5  # Critical threshold for triggering strict actions

DEBUG_FPS = os.getenv("DEBUG_FPS") == '1'
STRICT_MODE = os.getenv("STRICT_MODE") == '1'

DEFAULT_TEXT_SIZE = 60
DEFAULT_TEXT_COLOR = rl.Color(200, 200, 200, 255)
FONT_DIR = os.path.join(BASEDIR, "selfdrive/assets/fonts")


class FontWeight(IntEnum):
  BLACK = 0
  BOLD = 1
  EXTRA_BOLD = 2
  EXTRA_LIGHT = 3
  MEDIUM = 4
  NORMAL = 5
  SEMI_BOLD = 6
  THIN = 7


class GuiApplication:
  def __init__(self, width: int, height: int):
    self._fonts: dict[FontWeight, rl.Font] = {}
    self._width = width
    self._height = height
    self._textures: list[rl.Texture] = []
    self._target_fps: int = DEFAULT_FPS
    self._last_fps_log_time: float = time.monotonic()
    self._window_close_requested = False

  def request_close(self):
    self._window_close_requested = True

  def init_window(self, title: str, fps: int = DEFAULT_FPS):
    atexit.register(self.close)  # Automatically call close() on exit

    HARDWARE.set_display_power(True)
    HARDWARE.set_screen_brightness(65)

    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT | rl.ConfigFlags.FLAG_VSYNC_HINT)
    rl.init_window(self._width, self._height, title)
    rl.set_target_fps(fps)

    self._target_fps = fps
    self._set_styles()
    self._load_fonts()

  def load_texture_from_image(self, file_name: str, width: int, height: int, alpha_premultiply = False):
    """Load and resize a texture, storing it for later automatic unloading."""
    image = rl.load_image(file_name)
    if alpha_premultiply:
      rl.image_alpha_premultiply(image)
    rl.image_resize(image, width, height)
    texture = rl.load_texture_from_image(image)
    # Set texture filtering to smooth the result
    rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

    rl.unload_image(image)

    self._textures.append(texture)
    return texture

  def close(self):
    if not rl.is_window_ready():
      return

    for texture in self._textures:
      rl.unload_texture(texture)
    self._textures = []

    for font in self._fonts.values():
      rl.unload_font(font)
    self._fonts = {}

    rl.close_window()

  def render(self):
    while not (self._window_close_requested or rl.window_should_close()):
      rl.begin_drawing()
      rl.clear_background(rl.BLACK)

      yield

      if DEBUG_FPS:
        rl.draw_fps(10, 10)

      rl.end_drawing()
      self._monitor_fps()

  def font(self, font_weight: FontWeight=FontWeight.NORMAL):
    return self._fonts[font_weight]

  @property
  def width(self):
    return self._width

  @property
  def height(self):
    return self._height

  def _load_fonts(self):
    font_files = (
      "Inter-Black.ttf",
      "Inter-Bold.ttf",
      "Inter-ExtraBold.ttf",
      "Inter-ExtraLight.ttf",
      "Inter-Medium.ttf",
      "Inter-Regular.ttf",
      "Inter-SemiBold.ttf",
      "Inter-Thin.ttf"
      )

    for index, font_file in enumerate(font_files):
      font = rl.load_font_ex(os.path.join(FONT_DIR, font_file), 120, None, 0)
      rl.set_texture_filter(font.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      self._fonts[index] = font

    rl.gui_set_font(self._fonts[FontWeight.NORMAL])

  def _set_styles(self):
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BORDER_WIDTH, 0)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, DEFAULT_TEXT_SIZE)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.BACKGROUND_COLOR, rl.color_to_int(rl.BLACK))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(DEFAULT_TEXT_COLOR))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BASE_COLOR_NORMAL, rl.color_to_int(rl.Color(50, 50, 50, 255)))

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
