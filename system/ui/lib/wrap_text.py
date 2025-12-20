import pyray as rl
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.application import font_fallback


def _break_long_word(font: rl.Font, word: str, font_size: int, max_width: int, spacing: float = 0) -> list[str]:
  if not word:
    return []

  parts = []
  remaining = word

  while remaining:
    if measure_text_cached(font, remaining, font_size, spacing).x <= max_width:
      parts.append(remaining)
      break

    # Binary search for the longest substring that fits
    left, right = 1, len(remaining)
    best_fit = 1

    while left <= right:
      mid = (left + right) // 2
      substring = remaining[:mid]
      width = measure_text_cached(font, substring, font_size, spacing).x

      if width <= max_width:
        best_fit = mid
        left = mid + 1
      else:
        right = mid - 1

    # Add the part that fits
    parts.append(remaining[:best_fit])
    remaining = remaining[best_fit:]

  return parts


_cache: dict[tuple[int, str, int, int, float], list[str]] = {}


def wrap_text(font: rl.Font, text: str, font_size: int, max_width: int, spacing: float = 0, emojis: bool = False) -> list[str]:
  if not text or max_width <= 0:
    return []

  font = font_fallback(font)
  spacing = round(spacing, 4)
  key = (font.texture.id, text, font_size, max_width, spacing)
  if key in _cache:
    return _cache[key]

  space_width = measure_text_cached(font, " ", font_size, spacing, emojis).x
  all_lines = []

  for paragraph in text.split('\n'):
    words = paragraph.split()
    if not words:
      all_lines.append("")
      continue

    lines = []
    cur_line_words: list[str] = []
    cur_width = 0.0

    for word in words:
      w_width = measure_text_cached(font, word, font_size, spacing, emojis).x

      # 1. Handle words too long for a single line
      if w_width > max_width:
        if cur_line_words:
          lines.append(" ".join(cur_line_words))
          cur_line_words, cur_width = [], 0.0
        lines.extend(_break_long_word(font, word, font_size, max_width, spacing))
        continue

      # 2. Check if word fits: (current width + space + word width)
      added_width = w_width + (space_width if cur_line_words else 0)

      if cur_width + added_width <= max_width:
        cur_line_words.append(word)
        cur_width += added_width
      else:
        lines.append(" ".join(cur_line_words))
        cur_line_words = [word]
        cur_width = w_width

    if cur_line_words:
      lines.append(" ".join(cur_line_words))
    all_lines.extend(lines)

  _cache[key] = all_lines
  return all_lines
