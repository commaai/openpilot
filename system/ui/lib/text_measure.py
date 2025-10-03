import pyray as rl
from openpilot.system.ui.lib.application import gui_app

_cache: dict[int, rl.Vector2] = {}


def measure_text_cached(font: rl.Font, text: str, font_size: int, spacing: int = 0) -> rl.Vector2:
  """Caches text measurements to avoid redundant calculations.

  Applies the same per-font scaling used by draw/measure wrappers so
  returned sizes match Qt across all call sites.
  """
  scale = gui_app.get_font_scale(font)
  key = hash((int(getattr(font.texture, 'id', 0)), text, font_size, spacing, scale))
  if key in _cache:
    return _cache[key]

  # Call Raylib's original measure function directly with our per-font scaling
  result = rl.measure_text_ex(font, text, float(font_size) * scale, spacing)  # noqa: TID251
  _cache[key] = result
  return result
