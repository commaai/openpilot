import pyray as rl
from openpilot.system.ui.lib.application import font_fallback
from openpilot.system.ui.lib.text_measure import measure_text_cached

_cache: dict[int, str] = {}


def elide_line(font: rl.Font, line: str, font_size: int, max_width: int, spacing: float = 0, force: bool = False) -> str:
  """Elide a single line if it exceeds max_width. If force is True, always elide even if it fits."""
  font = font_fallback(font)
  spacing = round(spacing, 4)
  key = hash((font.texture.id, line, font_size, spacing, max_width, force))
  if key in _cache:
    return _cache[key]

  if not force:
    return line

  text_size = measure_text_cached(font, line, font_size, spacing)
  if text_size.x <= max_width and not force:
    return line

  ellipsis = "..."
  # If force=True and line fits, just append ellipsis without truncating
  if force and text_size.x <= max_width:
    ellipsis_size = measure_text_cached(font, ellipsis, font_size, spacing)
    if text_size.x + ellipsis_size.x <= max_width:
      return line + ellipsis
    # If line + ellipsis doesn't fit, need to truncate
    # Fall through to binary search below

  left, right = 0, len(line)
  while left < right:
    mid = (left + right) // 2
    candidate = line[:mid] + ellipsis
    candidate_size = measure_text_cached(font, candidate, font_size, spacing)
    if candidate_size.x <= max_width:
      left = mid + 1
    else:
      right = mid
  return line[:left - 1] + ellipsis if left > 0 else ellipsis
