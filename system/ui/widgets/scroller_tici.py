import pyray as rl
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.widgets import Widget

ITEM_SPACING = 40
LINE_COLOR = rl.GRAY
LINE_PADDING = 40


class Scroller(Widget):
  def __init__(self, items: list[Widget], spacing: int = ITEM_SPACING, line_separator: bool = False):
    super().__init__()
    self._items: list[Widget] = []
    self._spacing = spacing
    self._line_separator = line_separator
    self.scroll_panel = GuiScrollPanel()

    for item in items:
      self.add_widget(item)

  def add_widget(self, item: Widget) -> None:
    self._items.append(item)
    item.set_touch_valid_callback(self.scroll_panel.is_touch_valid)

  def _render(self, _):
    items = [item for item in self._items if item.is_visible]
    if not items:
      return

    # 1. Geometry Setup
    line_h = 1 if self._line_separator else 0
    item_gap = self._spacing + line_h
    content_height = sum(i.rect.height for i in items) + item_gap * (len(items) - 1)

    scroll_y = self.scroll_panel.update(self._rect, rl.Rectangle(0, 0, self._rect.width, content_height))
    rect_x, rect_y = int(self._rect.x), int(self._rect.y)
    rect_w, rect_h = int(self._rect.width), int(self._rect.height)

    rl.begin_scissor_mode(rect_x, rect_y, rect_w, rect_h)

    cur_y = self._rect.y + scroll_y
    view_bottom = self._rect.y + self._rect.height

    for i, item in enumerate(items):
      if cur_y > view_bottom:
        break

      item_h = item.rect.height
      if cur_y + item_h > self._rect.y:
        item.set_position(self._rect.x, cur_y)
        item.set_parent_rect(self._rect)
        item.render()

        if self._line_separator and i < len(items) - 1:
          line_y = int(cur_y + item_h + self._spacing // 2)
          rl.draw_line(rect_x + LINE_PADDING, line_y, rect_x + rect_w - LINE_PADDING, line_y, LINE_COLOR)

      cur_y += item_h + item_gap

    rl.end_scissor_mode()

  def show_event(self):
    super().show_event()
    # Reset to top
    self.scroll_panel.set_offset(0)
    for item in self._items:
      item.show_event()

  def hide_event(self):
    super().hide_event()
    for item in self._items:
      item.hide_event()
