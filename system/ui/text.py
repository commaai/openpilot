#!/usr/bin/env python3
import re
import sys
import pyray as rl
from openpilot.system.hardware import HARDWARE, PC
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.button import gui_button, ButtonStyle
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.widget import Widget

MARGIN = 50
SPACING = 40
FONT_SIZE = 72
LINE_HEIGHT = 80
BUTTON_SIZE = rl.Vector2(310, 160)

DEMO_TEXT = """This is a sample text that will be wrapped and scrolled if necessary.
            The text is long enough to demonstrate scrolling and word wrapping.""" * 30


def wrap_text(text, font_size, max_width):
  lines = []
  font = gui_app.font()

  for paragraph in text.split("\n"):
    if not paragraph.strip():
      # Don't add empty lines first, ensuring wrap_text("") returns []
      if lines:
        lines.append("")
      continue
    indent = re.match(r"^\s*", paragraph).group()
    current_line = indent
    words = re.split(r"(\s+)", paragraph[len(indent):])
    while len(words):
      word = words.pop(0)
      test_line = current_line + word + (words.pop(0) if words else "")
      if measure_text_cached(font, test_line, font_size).x <= max_width:
        current_line = test_line
      else:
        lines.append(current_line)
        current_line = word + " "
    current_line = current_line.rstrip()
    if current_line:
      lines.append(current_line)

  return lines


class TextWindow(Widget):
  def __init__(self, text: str):
    super().__init__()
    self._textarea_rect = rl.Rectangle(MARGIN, MARGIN, gui_app.width - MARGIN * 2, gui_app.height - MARGIN * 2)
    self._wrapped_lines = wrap_text(text, FONT_SIZE, self._textarea_rect.width - 20)
    self._content_rect = rl.Rectangle(0, 0, self._textarea_rect.width - 20, len(self._wrapped_lines) * LINE_HEIGHT)
    self._scroll_panel = GuiScrollPanel(show_vertical_scroll_bar=True)
    self._scroll_panel._offset.y = -max(self._content_rect.height - self._textarea_rect.height, 0)

  def _render(self, rect: rl.Rectangle):
    scroll = self._scroll_panel.handle_scroll(self._textarea_rect, self._content_rect)
    rl.begin_scissor_mode(int(self._textarea_rect.x), int(self._textarea_rect.y), int(self._textarea_rect.width), int(self._textarea_rect.height))
    for i, line in enumerate(self._wrapped_lines):
      position = rl.Vector2(self._textarea_rect.x + scroll.x, self._textarea_rect.y + scroll.y + i * LINE_HEIGHT)
      if position.y + LINE_HEIGHT < self._textarea_rect.y or position.y > self._textarea_rect.y + self._textarea_rect.height:
        continue
      rl.draw_text_ex(gui_app.font(), line, position, FONT_SIZE, 0, rl.WHITE)
    rl.end_scissor_mode()

    button_bounds = rl.Rectangle(rect.width - MARGIN - BUTTON_SIZE.x - SPACING, rect.height - MARGIN - BUTTON_SIZE.y, BUTTON_SIZE.x, BUTTON_SIZE.y)
    ret = gui_button(button_bounds, "Exit" if PC else "Reboot", button_style=ButtonStyle.TRANSPARENT)
    if ret:
      if PC:
        gui_app.request_close()
      else:
        HARDWARE.reboot()
    return ret


if __name__ == "__main__":
  text = sys.argv[1] if len(sys.argv) > 1 else DEMO_TEXT
  gui_app.init_window("Text Viewer")
  text_window = TextWindow(text)
  for _ in gui_app.render():
    text_window.render(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
