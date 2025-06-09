import pyray as rl
from openpilot.system.ui.lib.text_measure import measure_text_cached


def _break_long_word(font: rl.Font, word: str, font_size: int, max_width: int) -> list[str]:
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

    # Add the part that fits
    parts.append(remaining[:best_fit])
    remaining = remaining[best_fit:]

  return parts


def wrap_text(font: rl.Font, text: str, font_size: int, max_width: int) -> list[str]:
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
    current_width = 0
    space_width = int(measure_text_cached(font, " ", font_size).x)

    for word in words:
      word_width = int(measure_text_cached(font, word, font_size).x)

      # Check if word alone exceeds max width (need to break the word)
      if word_width > max_width:
        # Finish current line if it has content
        if current_line:
          lines.append(" ".join(current_line))
          current_line = []
          current_width = 0

        # Break the long word into parts
        lines.extend(_break_long_word(font, word, font_size, max_width))
        continue

      # Calculate width if we add this word
      needed_width = current_width
      if current_line:  # Need space before word
        needed_width += space_width
      needed_width += word_width

      # Check if word fits on current line
      if needed_width <= max_width:
        current_line.append(word)
        current_width = needed_width
      else:
        # Start new line with this word
        if current_line:
          lines.append(" ".join(current_line))
        current_line = [word]
        current_width = word_width

    # Add remaining words
    if current_line:
      lines.append(" ".join(current_line))

    # Add all lines from this paragraph
    all_lines.extend(lines)

  return all_lines
