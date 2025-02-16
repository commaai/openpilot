import atexit
import os
import pyray as rl
from enum import IntEnum
from openpilot.common.basedir import BASEDIR

DEFAULT_TEXT_SIZE = 60
DEFAULT_FPS = 60
FONT_DIR = os.path.join(BASEDIR, "selfdrive/assets/fonts")

DEBUG_FPS = os.getenv("DEBUG_FPS") == '1'


class FontWeight(IntEnum):
  BLACK = 0
  BOLD = 1
  EXTRA_BOLD = 2
  EXTRA_LIGHT = 3
  MEDIUM = 4
  NORMAL = 5
  SEMI_BOLD= 6
  THIN = 7


class GuiApplication:
  def __init__(self, width: int, height: int):
    self._fonts: dict[FontWeight, rl.Font] = {}
    self._width = width
    self._height = height
    self._textures: list[rl.Texture] = []

  def init_window(self, title: str, fps: int=DEFAULT_FPS):
    atexit.register(self.close)  # Automatically call close() on exit

    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT | rl.ConfigFlags.FLAG_VSYNC_HINT)
    rl.init_window(self._width, self._height, title)
    rl.set_target_fps(fps)

    self._set_styles()
    self._load_fonts()

  def load_texture_from_image(self, file_name: str, width: int, height: int):
    """Load and resize a texture, storing it for later automatic unloading."""
    image = rl.load_image(file_name)
    rl.image_resize(image, width, height)
    texture = rl.load_texture_from_image(image)
    # Set texture filtering to smooth the result
    rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

    rl.unload_image(image)

    self._textures.append(texture)
    return texture

  def close(self):
    for texture in self._textures:
      rl.unload_texture(texture)

    for font in self._fonts.values():
      rl.unload_font(font)

    rl.close_window()

  def render(self):
    while not rl.window_should_close():
      rl.begin_drawing()
      rl.clear_background(rl.BLACK)

      yield

      if DEBUG_FPS:
        rl.draw_fps(10, 10)

      rl.end_drawing()

  def font(self, font_wight: FontWeight=FontWeight.NORMAL):
    return self._fonts[font_wight]

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
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(rl.Color(200, 200, 200, 255)))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.BACKGROUND_COLOR, rl.color_to_int(rl.Color(30, 30, 30, 255)))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BASE_COLOR_NORMAL, rl.color_to_int(rl.Color(50, 50, 50, 255)))


gui_app = GuiApplication(2160, 1080)
