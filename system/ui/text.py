#!/usr/bin/env python3
import sys
import pyray as rl

from openpilot.system.hardware import HARDWARE
from openpilot.system.ui.lib.button import gui_button, ButtonStyle
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.application import gui_app

MARGIN = 50
SPACING = 50
FONT_SIZE = 60
LINE_HEIGHT = 64
BUTTON_SIZE = rl.Vector2(310, 160)

DEMO_TEXT = """This is a sample text that will be wrapped and scrolled if necessary.
            The text is long enough to demonstrate scrolling and word wrapping.""" * 30

def wrap_text(text, font_size, max_width):
  lines = []
  current_line = ""
  font = gui_app.font()

  for word in text.split():
    test_line = current_line + word + " "
    if rl.measure_text_ex(font, test_line, font_size, 0).x <= max_width:
      current_line = test_line
    else:
      lines.append(current_line)
      current_line = word + " "
  if current_line:
    lines.append(current_line)

  return lines


def main():
  gui_app.init_window("Text")

  text_content = sys.argv[1] if len(sys.argv) > 1 else DEMO_TEXT

  textarea_rect = rl.Rectangle(MARGIN, MARGIN, gui_app.width - MARGIN * 2, gui_app.height - MARGIN * 2 - BUTTON_SIZE.y - SPACING)
  wrapped_lines = wrap_text(text_content, FONT_SIZE, textarea_rect.width - 20)
  content_rect = rl.Rectangle(0, 0, textarea_rect.width - 20, len(wrapped_lines) * LINE_HEIGHT)
  scroll_panel = GuiScrollPanel(show_vertical_scroll_bar=True)

  for _ in gui_app.render():
    scroll = scroll_panel.handle_scroll(textarea_rect, content_rect)

    rl.begin_scissor_mode(int(textarea_rect.x), int(textarea_rect.y), int(textarea_rect.width), int(textarea_rect.height))
    for i, line in enumerate(wrapped_lines):
      position = rl.Vector2(textarea_rect.x + scroll.x, textarea_rect.y + scroll.y + i * LINE_HEIGHT)
      if position.y + LINE_HEIGHT < textarea_rect.y or position.y > textarea_rect.y + textarea_rect.height:
        continue
      rl.draw_text_ex(gui_app.font(), line.strip(), position, FONT_SIZE, 0, rl.WHITE)
    rl.end_scissor_mode()

    button_bounds = rl.Rectangle(gui_app.width - MARGIN - BUTTON_SIZE.x, gui_app.height - MARGIN - BUTTON_SIZE.y, BUTTON_SIZE.x, BUTTON_SIZE.y)
    if gui_button(button_bounds, "Reboot", button_style=ButtonStyle.TRANSPARENT):
      HARDWARE.reboot()


if __name__ == "__main__":
  main()
