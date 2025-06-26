import os
import pyray as rl
from dataclasses import dataclass
from collections.abc import Callable
from abc import ABC, abstractmethod
from openpilot.system.ui.lib.widget import Widget
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.application import gui_app, FontWeight

LINE_COLOR = rl.GRAY
ITEM_SPACING = 40


class Scroller(Widget):
  def __init__(self, items: list, pad_end: bool = True):
    super().__init__()
    self._items: list = items
    self._pad_end = pad_end
    self.scroll_panel = GuiScrollPanel()

    for item in self._items:
      item.set_click_valid_callback(self.scroll_panel.is_click_valid,
                                    self.scroll_panel.is_touch_valid)

  def add_widget(self, item: Widget) -> None:
    self._items.append(item)
    item.set_click_valid_callback(self.scroll_panel.is_click_valid,
                                  self.scroll_panel.is_touch_valid)

  def _render(self, _):
    # TODO: don't draw items that are not in the viewport
    visible_items = [item for item in self._items if item.is_visible]
    content_height = sum(item.rect.height for item in visible_items) + ITEM_SPACING * (len(visible_items))
    print('content height', content_height, 'rect height', self._rect.height)
    if not self._pad_end:
      content_height -= ITEM_SPACING
    scroll = self.scroll_panel.handle_scroll(self._rect, rl.Rectangle(0, 0, self._rect.width, content_height))
    print(scroll.x, scroll.y)
    rl.draw_rectangle_lines_ex(self._rect, 5, LINE_COLOR)

    rl.begin_scissor_mode(int(self._rect.x), int(self._rect.y),
                          int(self._rect.width), int(self._rect.height))

    cur_height = 0
    for idx, item in enumerate(visible_items):
      if not item.is_visible:
        continue
      # print(f"Rendering item {idx} at position {cur_height}")
      # print('item height', item.rect.height)

      # Nicely lay out items vertically
      # x = self._rect.x + cur_height + ITEM_SPACING * (idx != 0)
      y = self._rect.y + cur_height + ITEM_SPACING * (idx != 0)
      cur_height += item.rect.height + ITEM_SPACING * (idx != 0)
      x = self._rect.x# + ITEM_SPACING
      # y = self._rect.y + (self._rect.height - item.rect.height) / 2

      # Consider scroll
      x += scroll.x
      y += scroll.y

      # Update item state
      item.set_position(x, y)
      item.set_parent_rect(self._rect)
      # print('setting item position', item.rect.x, item.rect.y)
      item.render()

    rl.end_scissor_mode()
