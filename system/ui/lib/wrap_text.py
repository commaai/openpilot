import pyray as rl
import re
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.application import font_fallback

# Regex for Unicode-aware tokenization: words, whitespace, symbols
WORD_RE = re.compile(r'(\s+|[\w]+|[^\w\s])', re.UNICODE)
_cache: dict[int, list[str]] = {}

def _break_long_word(font: rl.Font, word: str, font_size: int, max_width: int) -> list[str]:
  """Break a long word into parts that fit within max_width."""
  if not word:
    return []
  parts = []
  remaining = word
  while remaining:
    if measure_text_cached(font, remaining, font_size).x <= max_width:
      parts.append(remaining)
      break
    # Binary search for the longest substring that fits
    left, right = 1, len(remaining)
    best_fit = 1
    while left <= right:
      mid = (left + right) // 2
      substring = remaining[:mid]
      width = measure_text_cached(font, substring, font_size).x
      if width <= max_width:
        best_fit = mid
        left = mid + 1
      else:
        right = mid - 1
    parts.append(remaining[:best_fit])
    remaining = remaining[best_fit:]
  return parts

def wrap_text(font: rl.Font, text: str, font_size: int, max_width: int) -> list[str]:
  """Wrap text to fit within max_width, handling Unicode and whitespace properly."""
  font = font_fallback(font)
  # Strip input text for consistent caching
  text = text.strip()
  key = hash((font.texture.id, text, font_size, max_width))
  if key in _cache:
    return _cache[key]

  if not text or max_width <= 0:
    return []

  all_lines: list[str] = []
  for paragraph in text.split('\n'):
    if not paragraph:
      all_lines.append("")
      continue

    words = WORD_RE.findall(paragraph)
    if not words:
      all_lines.append("")
      continue

    current_line = ""
    for word in words:
      # Normalize whitespace to a single space
      if word.isspace():
        word = " "
      word_width = measure_text_cached(font, word, font_size).x

      if word_width > max_width:
        # Break long word
        if current_line:
          all_lines.append(current_line.strip())
          current_line = ""
        for part in _break_long_word(font, word, font_size, max_width):
          all_lines.append(part.strip())
        continue

      test_line = current_line + word if current_line else word
      test_width = measure_text_cached(font, test_line, font_size).x

      if test_width <= max_width:
        current_line = test_line
      else:
        # Start new line, remove trailing space from previous line
        if current_line:
          all_lines.append(current_line.strip())
        current_line = word if word != " " else ""  # Don't start with space

    if current_line:
      all_lines.append(current_line.strip())

  _cache[key] = all_lines
  return all_lines
