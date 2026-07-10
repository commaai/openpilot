import io
import re
import functools
from importlib.resources import as_file

from fontTools.ttLib import TTFont
import pyray as rl

from openpilot.system.ui.lib.application import FONT_DIR

_cache: dict[str, rl.Texture] = {}

# codepoints that modify/join an emoji rather than selecting a glyph of their own
_EMOJI_MODIFIERS = {0x200D, 0xFE0E, 0xFE0F}

EMOJI_REGEX = re.compile(
"""[\U0001F600-\U0001F64F
\U0001F300-\U0001F5FF
\U0001F680-\U0001F6FF
\U0001F1E0-\U0001F1FF
\U00002700-\U000027BF
\U0001F900-\U0001F9FF
\U00002600-\U000026FF
\U00002300-\U000023FF
\U00002B00-\U00002BFF
\U0001FA70-\U0001FAFF
\U0001F700-\U0001F77F
\u2640-\u2642
\u2600-\u2B55
\u200d
\u23cf
\u23e9
\u231a
\ufe0f
\u3030
]+""".replace("\n", ""),
  flags=re.UNICODE
)

@functools.cache
def _load_emoji_font():
  # NotoColorEmoji stores each glyph as an embedded PNG in the CBDT/CBLC tables,
  # so pull those PNGs out directly instead of rasterizing the font with PIL.
  with as_file(FONT_DIR.joinpath("NotoColorEmoji.ttf")) as font_path:
    font = TTFont(io.BytesIO(font_path.read_bytes()))
    return font.getBestCmap(), font["CBDT"].strikeData

def find_emoji(text):
  return [(m.start(), m.end(), m.group()) for m in EMOJI_REGEX.finditer(text)]

def _emoji_png(emoji):
  cmap, strike_data = _load_emoji_font()
  for ch in emoji:
    if ord(ch) in _EMOJI_MODIFIERS:
      continue
    name = cmap.get(ord(ch))
    if name is None:
      continue
    for strike in strike_data:
      glyph = strike.get(name)
      if glyph is not None:
        return glyph.imageData
  return None

def emoji_tex(emoji):
  if emoji not in _cache:
    png = _emoji_png(emoji)
    if png is not None:
      image = rl.load_image_from_memory(".png", png, len(png))
    else:
      image = rl.gen_image_color(128, 128, rl.BLANK)
    _cache[emoji] = rl.load_texture_from_image(image)
    rl.unload_image(image)
  return _cache[emoji]
