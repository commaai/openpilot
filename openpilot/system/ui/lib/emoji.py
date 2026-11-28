import functools
from importlib.resources import as_file

import numpy as np
import raylib_nb as rl
from openpilot.system.ui.lib.application import ASSETS_DIR

# Noto Color Emoji at 72px (default skin tone) and region flags, as RGBA arrays keyed by codepoints ("1f468_200d_1f469").
# Built by selfdrive/assets/build_emoji.py. Fonts can't draw color, so labels draw these over placeholder characters instead
EMOJI_FILE = "emoji.npz"
PLACEHOLDER = 0xE000  # private use area

_textures: dict[str, rl.Texture] = {}


@functools.cache
def _archive() -> np.lib.npyio.NpzFile:
  with as_file(ASSETS_DIR.joinpath(EMOJI_FILE)) as path:
    return np.load(path)  # reads each emoji on first use


@functools.cache
def _emoji() -> dict[str, str]:
  return {"".join(chr(int(c, 16)) for c in key.split("_")): key for key in _archive().files}


def replace_emoji(text: str, font_chars: set[str]) -> tuple[str, dict[str, str]]:
  """Swaps each emoji for two placeholder characters, which take up about an emoji's width.
  Returns the new text and the emoji to draw for each placeholder."""
  if font_chars.issuperset(text):
    return text, {}

  emoji = _emoji()
  max_len = max(map(len, emoji))
  found: dict[str, str] = {}
  out = []
  i = 0
  while i < len(text):
    for n in range(min(max_len * 2, len(text) - i), 0, -1):
      seq = text[i:i + n]
      key = "".join(ch for ch in seq if ch != "️" and not 0x1F3FB <= ord(ch) <= 0x1F3FF)
      # a character we can draw as text stays text, unless it asks for emoji style
      if key in emoji and (len(key) > 1 or key not in font_chars or "️" in seq):
        placeholder = chr(PLACEHOLDER + len(found))
        found[placeholder] = emoji[key]
        out.append(placeholder * 2)
        i += n
        break
    else:
      # skip emoji style selectors and skin tones (only the default tone ships)
      if text[i] != "️" and not 0x1F3FB <= ord(text[i]) <= 0x1F3FF:
        out.append(text[i])
      i += 1
  return "".join(out), found


def emoji_texture(key: str) -> rl.Texture:
  if key not in _textures:
    rgba = np.ascontiguousarray(_archive()[key])
    texture = rl.load_texture_from_image(rl.Image(None, rgba.shape[1], rgba.shape[0], 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8))
    rl.update_texture(texture, rgba)
    rl.set_texture_filter(texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
    _textures[key] = texture
  return _textures[key]
