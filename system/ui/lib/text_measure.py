import pyray as rl
from openpilot.system.ui.lib.application import FONT_SCALE
from openpilot.system.ui.lib.emoji import find_emoji

_cache: dict[int, rl.Vector2] = {}


def measure_text_cached(font: rl.Font, text: str, font_size: int, spacing: int = 0) -> rl.Vector2:
  """Caches text measurements to avoid redundant calculations."""
  key = hash((font.texture.id, text, font_size, spacing))
  if key in _cache:
    return _cache[key]

  result = rl.measure_text_ex(font, text, font_size * FONT_SCALE, spacing)  # noqa: TID251
  for _, _, e in find_emoji(text):
    result.x += font_size - rl.measure_text_ex(font, e, font_size * FONT_SCALE, 0).x
  _cache[key] = result
  return result
