import pyray as rl

_cache: dict[int, rl.Vector2] = {}


def measure_text_cached(font: rl.Font, text: str, font_size: int, spacing: int = 0) -> rl.Vector2:
  """Caches text measurements to avoid redundant calculations."""
  key = hash((font.texture.id, text, font_size, spacing))
  if key in _cache:
    return _cache[key]

  result = rl.measure_text_ex(font, text, font_size, spacing)  # noqa: TID251
  _cache[key] = result
  return result
