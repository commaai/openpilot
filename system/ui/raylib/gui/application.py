import os
import pyray as rl
from enum import IntEnum
from openpilot.common.basedir import BASEDIR

DEFAULT_APP_TEXT_SIZE = 60

class FontSize(IntEnum):
  NORMAL = 0
  BOLD = 1
  EXTRA_BOLD = 2
  EXTRA_LIGHT = 3
  MEDIUM = 4
  REGULAR = 5
  SEMI_BOLD= 6
  THIN = 7


class GuiApplication:
  def __init__(self, width: int, height: int):
    self._fonts: dict[FontSize, rl.Font] = {}
    self._width = width
    self._height = height

  def init_window(self, title: str, fps: int):
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)
    rl.init_window(self._width, self._height, title)
    rl.set_target_fps(fps)

    # Set styles
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BORDER_WIDTH, 0)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.TEXT_SIZE, DEFAULT_APP_TEXT_SIZE)
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.BACKGROUND_COLOR, rl.color_to_int(rl.BLACK))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.TEXT_COLOR_NORMAL, rl.color_to_int(rl.Color(200, 200, 200, 255)))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiDefaultProperty.BACKGROUND_COLOR, rl.color_to_int(rl.Color(30, 30, 30, 255)))
    rl.gui_set_style(rl.GuiControl.DEFAULT, rl.GuiControlProperty.BASE_COLOR_NORMAL, rl.color_to_int(rl.Color(50, 50, 50, 255)))

    self._load_fonts()

  def close(self):
    rl.close_window()

  def font(self, font_size: FontSize=FontSize.NORMAL):
    return self._fonts[font_size]

  @property
  def width(self):
    return self._width

  @property
  def height(self):
    return self._height

  def _load_fonts(self):
    font_files = (
      "selfdrive/assets/fonts/Inter-Black.ttf",
      "selfdrive/assets/fonts/Inter-Bold.ttf",
      "selfdrive/assets/fonts/Inter-ExtraBold.ttf",
      "selfdrive/assets/fonts/Inter-ExtraLight.ttf",
      "selfdrive/assets/fonts/Inter-Medium.ttf",
      "selfdrive/assets/fonts/Inter-Regular.ttf",
      "selfdrive/assets/fonts/Inter-SemiBold.ttf",
      "selfdrive/assets/fonts/Inter-Thin.ttf"
      )

    for index, font_file in enumerate(font_files):
      font = rl.load_font_ex(os.path.join(BASEDIR, font_file), 120, None, 0)
      rl.set_texture_filter(font.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      self._fonts[index] = font

    rl.gui_set_font(self._fonts[FontSize.NORMAL])

gui_app = GuiApplication(2160, 1080)
