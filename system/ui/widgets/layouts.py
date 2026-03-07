from enum import IntFlag
from openpilot.system.ui.widgets import Widget


class Alignment(IntFlag):
  LEFT = 0
  # TODO: implement
  # H_CENTER = 2
  # RIGHT = 4

  TOP = 8
  V_CENTER = 16
  BOTTOM = 32


class _BoxLayout(Widget):
  def __init__(self, horizontal: bool, widgets: list[Widget] | None = None, spacing: int = 0,
               alignment: Alignment = Alignment.LEFT | Alignment.V_CENTER):
    super().__init__()
    self._horizontal = horizontal
    self._spacing = spacing
    self._alignment = alignment

    if widgets is not None:
      for widget in widgets:
        self.add_widget(widget)

  @property
  def widgets(self) -> list[Widget]:
    return self._children

  def add_widget(self, widget: Widget) -> None:
    self._child(widget)

  def _render(self, _):
    visible_widgets = [w for w in self._children if w.is_visible]
    cur_offset = 0

    for idx, widget in enumerate(visible_widgets):
      spacing = self._spacing if (idx > 0) else 0

      if self._horizontal:
        main = self._rect.x + cur_offset + spacing
        cur_offset += widget.rect.width + spacing

        if self._alignment & Alignment.TOP:
          cross = self._rect.y
        elif self._alignment & Alignment.BOTTOM:
          cross = self._rect.y + self._rect.height - widget.rect.height
        else:
          cross = self._rect.y + (self._rect.height - widget.rect.height) / 2

        x, y = main, cross
      else:
        main = self._rect.y + cur_offset + spacing
        cur_offset += widget.rect.height + spacing
        x, y = self._rect.x, main

      widget.set_position(round(x), round(y))
      widget.set_parent_rect(self._rect)
      widget.render()


class HBoxLayout(_BoxLayout):
  """A Widget that lays out child Widgets horizontally."""

  def __init__(self, widgets: list[Widget] | None = None, spacing: int = 0,
               alignment: Alignment = Alignment.LEFT | Alignment.V_CENTER):
    super().__init__(True, widgets, spacing, alignment)


class VBoxLayout(_BoxLayout):
  """A Widget that lays out child Widgets vertically."""

  def __init__(self, widgets: list[Widget] | None = None, spacing: int = 0):
    super().__init__(False, widgets, spacing)
