import pyray as rl
from openpilot.system.ui.lib.application import FONT_SCALE
from openpilot.system.ui.lib.emoji import find_emoji

_cache: dict[tuple[int, str, int, float], rl.Vector2] = {}


def measure_text(font: rl.Font, text: str, font_size: int, spacing: float = 0, emojis: bool = False) -> rl.Vector2:
  """Caches text measurements to avoid redundant calculations."""
  scaled_size = font_size * FONT_SCALE
  found_emojis = find_emoji(text) if emojis else []
  if found_emojis:
    non_emoji_parts = []
    last_idx = 0
    for start, end, _ in found_emojis:
      non_emoji_parts.append(text[last_idx:start])
      last_idx = end
    non_emoji_parts.append(text[last_idx:])

    result = rl.measure_text_ex(font, "".join(non_emoji_parts), scaled_size, spacing)  # noqa: TID251
    result.x += len(found_emojis) * scaled_size

    # Ensure a height exists if the line is purely emojis
    if result.y == 0:
      result.y = scaled_size
  else:
    result = rl.measure_text_ex(font, text, scaled_size, spacing)  # noqa: TID251

  return result


def measure_text_cached(font: rl.Font, text: str, font_size: int, spacing: float = 0, emojis: bool = False) -> rl.Vector2:
  """Caches text measurements to avoid redundant calculations."""
  spacing = round(spacing, 4)
  key = (font.texture.id, text, font_size, spacing)
  if key in _cache:
    return _cache[key]

  result = measure_text(font, text, font_size, spacing, emojis)
  _cache[key] = result
  return result
