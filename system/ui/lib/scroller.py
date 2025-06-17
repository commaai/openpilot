import os
import pyray as rl
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
from abc import ABC, abstractmethod
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.lib.widget import Widget
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.application import gui_app, FontWeight

LINE_PADDING = 40
LINE_COLOR = rl.GRAY
ITEM_SPACING = 30
ITEM_BASE_HEIGHT = 170
ITEM_TEXT_FONT_SIZE = 50
ITEM_TEXT_COLOR = rl.WHITE
ITEM_DESC_TEXT_COLOR = rl.Color(128, 128, 128, 255)
ITEM_DESC_FONT_SIZE = 40
ITEM_DESC_V_OFFSET = 130
RIGHT_ITEM_SPACING = 20
ICON_SIZE = 80
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 100
BUTTON_BORDER_RADIUS = 50
BUTTON_FONT_SIZE = 35
BUTTON_FONT_WEIGHT = FontWeight.MEDIUM


class Text(Widget):
  def __init__(self, text: str, font_size: int = 40, color: rl.Color = rl.WHITE):
    super().__init__()
    self.text = text
    self.font_size = font_size
    self.color = color

  def _render(self, rect: rl.Rectangle):
    print('text rect', rect.x, rect.y, rect.width, rect.height)
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    font = gui_app.font(FontWeight.NORMAL)
    wrapped_text = wrap_text(font, self.text, self.font_size, int(rect.width))
    print(wrapped_text)
    y_offset = rect.y + (rect.height - len(wrapped_text) * self.font_size) // 2
    for line in wrapped_text:
      rl.draw_text_ex(font, line, rl.Vector2(rect.x + 20, y_offset), self.font_size, 0, self.color)
      y_offset += self.font_size
    rl.end_scissor_mode()


# horiz or vert enum
class ScrollMode(Enum):
  HORIZONTAL = 0
  VERTICAL = 1


class Scroller(Widget):
  def __init__(self, items: list, mode: ScrollMode = ScrollMode.HORIZONTAL):
    super().__init__()
    self._items: list = items
    self._mode = mode
    self.scroll_panel = GuiScrollPanel()

    for item in self._items:
      item.set_click_valid_callback(lambda: self.scroll_panel.is_click_valid())

  def _render(self, _):
    content_width = sum(item._rect.width for item in self._items) + ITEM_SPACING * (len(self._items))
    content_height = max(item._rect.height for item in self._items) + ITEM_SPACING * (len(self._items))

    scroll = self.scroll_panel.handle_scroll(self._rect, rl.Rectangle(0, 0, content_width, content_height))
    print(scroll.x, scroll.y)

    cur_width = 0
    for idx, item in enumerate(self._items):
      item_x = self._rect.x + cur_width + ITEM_SPACING * (idx != 0)
      cur_width += item.width + ITEM_SPACING * (idx != 0)
      print(cur_width, item.width, ITEM_SPACING)
      item_y = self._rect.y + (self._rect.height - item.height) / 2

      # Consider scroll
      item_x += scroll.x
      item_y += scroll.y
      item.set_position(item_x, item_y)

      # Update item state
      item.render()
