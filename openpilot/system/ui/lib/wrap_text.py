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


_cache: dict[int, list[str]] = {}


def wrap_text(font: rl.Font, text: str, font_size: int, max_width: int, spacing: float = 0) -> list[str]:
  font = font_fallback(font)
  spacing = round(spacing, 4)
  key = hash((font.texture.id, text, font_size, max_width, spacing))
  if key in _cache:
    return _cache[key]

  if not text or max_width <= 0:
    return []

  # Split text by newlines first to preserve explicit line breaks
  paragraphs = text.split('\n')
  all_lines: list[str] = []

  for paragraph in paragraphs:
    # Handle empty paragraphs (preserve empty lines)
    if not paragraph.strip():
      all_lines.append("")
      continue

    # Process each paragraph separately
    words = paragraph.split()
    if not words:
      all_lines.append("")
      continue

    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
      word_width = measure_text_cached(font, word, font_size, spacing).x

      # Check if word alone exceeds max width (need to break the word)
      if word_width > max_width:
        # Finish current line if it has content
        if current_line:
          lines.append(" ".join(current_line))
          current_line = []

        # Break the long word into parts
        lines.extend(_break_long_word(font, word, font_size, max_width, spacing))
        continue

      # Measure the actual joined string to get accurate width (accounts for kerning, etc.)
      test_line = " ".join(current_line + [word]) if current_line else word
      test_width = measure_text_cached(font, test_line, font_size, spacing).x

      # Check if word fits on current line
      if test_width <= max_width:
        current_line.append(word)
      else:
        # Start new line with this word
        if current_line:
          lines.append(" ".join(current_line))
        current_line = [word]

    # Add remaining words
    if current_line:
      lines.append(" ".join(current_line))

    # Add all lines from this paragraph
    all_lines.extend(lines)

  _cache[key] = all_lines
  return all_lines
