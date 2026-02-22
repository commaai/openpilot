import abc
from openpilot.system.ui.widgets import Widget


class Layout(Widget, abc.ABC):
  """
  A Widget that lays out child Widgets.
  """

  def __init__(self):
    super().__init__()
    self._children: list[Widget] = []
    self._visible_widgets: list[Widget] = []  # tracks offscreen widgets for performance

  @property
  def children(self) -> list[Widget]:
    return self._children

  def add_widget(self, widget: Widget) -> None:
    self._children.append(widget)


class HBoxLayout(Layout):
  """
  A Layout that arranges child Widgets horizontally.
  """

  def _layout(self) -> None:
    self._visible_items = [item for item in self._items if item.is_visible]

    # Add line separator between items
    if self._line_separator is not None:
      l = len(self._visible_items)
      for i in range(1, len(self._visible_items)):
        self._visible_items.insert(l - i, self._line_separator)

    self._content_size = sum(item.rect.width if self._horizontal else item.rect.height for item in self._visible_items)
    self._content_size += self._spacing * (len(self._visible_items) - 1)
    self._content_size += self._pad_start + self._pad_end

    self._scroll_offset = self._get_scroll(self._visible_items, self._content_size)

    self._item_pos_filter.update(self._scroll_offset)

    cur_pos = 0
    for idx, item in enumerate(self._visible_items):
      spacing = self._spacing if (idx > 0) else self._pad_start
      # Nicely lay out items horizontally/vertically
      if self._horizontal:
        x = self._rect.x + cur_pos + spacing
        y = self._rect.y + (self._rect.height - item.rect.height) / 2
        cur_pos += item.rect.width + spacing
      else:
        x = self._rect.x + (self._rect.width - item.rect.width) / 2
        y = self._rect.y + cur_pos + spacing
        cur_pos += item.rect.height + spacing

      # Consider scroll
      if self._horizontal:
        x += self._scroll_offset
      else:
        y += self._scroll_offset

      # Add some jello effect when scrolling
      if DO_JELLO:
        if self._horizontal:
          cx = self._rect.x + self._rect.width / 2
          jello_offset = self._scroll_offset - np.interp(x + item.rect.width / 2,
                                                         [self._rect.x, cx, self._rect.x + self._rect.width],
                                                         [self._item_pos_filter.x, self._scroll_offset, self._item_pos_filter.x])
          x -= np.clip(jello_offset, -20, 20)
        else:
          cy = self._rect.y + self._rect.height / 2
          jello_offset = self._scroll_offset - np.interp(y + item.rect.height / 2,
                                                         [self._rect.y, cy, self._rect.y + self._rect.height],
                                                         [self._item_pos_filter.x, self._scroll_offset, self._item_pos_filter.x])
          y -= np.clip(jello_offset, -20, 20)

      # Update item state
      item.set_position(round(x), round(y))  # round to prevent jumping when settling
      item.set_parent_rect(self._rect)
