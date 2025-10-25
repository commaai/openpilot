from importlib.resources import as_file, files
import os
import pyray as rl
from enum import StrEnum
from openpilot.system.ui.lib.multilang import TRANSLATIONS_DIR, UNIFONT_LANGUAGES, multilang
from openpilot.common.swaglog import cloudlog

ASSETS_DIR = files("openpilot.selfdrive").joinpath("assets")
FONT_DIR = ASSETS_DIR.joinpath("fonts")

DEFAULT_TEXT_SIZE = 60
DEFAULT_TEXT_COLOR = rl.WHITE

# Qt draws fonts accounting for ascent/descent differently, so compensate to match old styles
# The real scales for the fonts below range from 1.212 to 1.266
FONT_SCALE = 1.242


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
  UNIFONT = "unifont.otf"


class FontManager:
  def __init__(self):
    self._fonts: dict[FontWeight, rl.Font] = {}

  def font(self, font_weight: FontWeight) -> rl.Font:
    return self._fonts[font_weight]

  def font_fallback(self, font: rl.Font) -> rl.Font:
    """Fall back to unifont for languages that require it."""
    if multilang.requires_unifont():
      return self.font(FontWeight.UNIFONT)
    return font

  def load_fonts(self):
    # Create a character set from our keyboard layouts
    from openpilot.system.ui.widgets.keyboard import KEYBOARD_LAYOUTS

    base_chars = set()
    for layout in KEYBOARD_LAYOUTS.values():
      base_chars.update(key for row in layout for key in row)
    base_chars |= set("–‑✓×°§•")

    # Load only the characters used in translations
    unifont_chars = set(base_chars)
    for language, code in multilang.languages.items():
      unifont_chars |= set(language)
      try:
        with open(os.path.join(TRANSLATIONS_DIR, f"app_{code}.po")) as f:
          lang_chars = set(f.read())
          if code in UNIFONT_LANGUAGES:
            unifont_chars |= lang_chars
          else:
            base_chars |= lang_chars
      except FileNotFoundError:
        cloudlog.warning(f"Translation file for language '{code}' not found when loading fonts.")

    base_chars = "".join(base_chars)
    cloudlog.debug(f"Loading fonts with {len(base_chars)} glyphs.")

    unifont_chars = "".join(unifont_chars)
    cloudlog.debug(f"Loading unifont with {len(unifont_chars)} glyphs.")

    base_codepoint_count = rl.ffi.new("int *", 1)
    base_codepoints = rl.load_codepoints(base_chars, base_codepoint_count)

    unifont_codepoint_count = rl.ffi.new("int *", 1)
    unifont_codepoints = rl.load_codepoints(unifont_chars, unifont_codepoint_count)

    for font_weight_file in FontWeight:
      with as_file(FONT_DIR.joinpath(font_weight_file)) as fspath:
        if font_weight_file == FontWeight.UNIFONT:
          font = rl.load_font_ex(fspath.as_posix(), 200, unifont_codepoints, unifont_codepoint_count[0])
        else:
          font = rl.load_font_ex(fspath.as_posix(), 200, base_codepoints, base_codepoint_count[0])
        rl.set_texture_filter(font.texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self._fonts[font_weight_file] = font

    rl.unload_codepoints(base_codepoints)
    rl.unload_codepoints(unifont_codepoints)
    rl.gui_set_font(self._fonts[FontWeight.NORMAL])

  def unload_fonts(self):
    for font in self._fonts.values():
      rl.unload_font(font)
    self._fonts = {}
