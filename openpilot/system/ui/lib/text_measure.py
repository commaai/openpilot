import pyray as rl
from openpilot.system.ui.lib.application import FONT_SCALE, font_fallback
from openpilot.system.ui.lib.emoji import find_emoji

_cache: dict[int, rl.Vector2] = {}


def measure_text_cached(font: rl.Font, text: str, font_size: int, spacing: float = 0) -> rl.Vector2:
  """Caches text measurements to avoid redundant calculations."""
  font = font_fallback(font)
  spacing = round(spacing, 4)
  key = hash((font.texture.id, text, font_size, spacing))
  if key in _cache:
    return _cache[key]

  # Measure normal characters without emojis, then add standard width for each found emoji
  emoji = find_emoji(text)
  if emoji:
    non_emoji_text = ""
    last_index = 0
    for start, end, _ in emoji:
      non_emoji_text += text[last_index:start]
      last_index = end
    non_emoji_text += text[last_index:]
  else:
    non_emoji_text = text

  result = rl.measure_text_ex(font, non_emoji_text, font_size * FONT_SCALE, spacing)  # noqa: TID251
  if emoji:
    result.x += len(emoji) * font_size * FONT_SCALE
    # If just emoji assume a single line height
    if result.y == 0:
      result.y = font_size * FONT_SCALE

  _cache[key] = result
  return result
