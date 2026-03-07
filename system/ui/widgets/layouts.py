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


class HBoxLayout(Widget):
  """
  A Widget that lays out child Widgets horizontally.
  """

  def __init__(self, widgets: list[Widget] | None = None, spacing: int = 0,
               alignment: Alignment = Alignment.LEFT | Alignment.V_CENTER):
    super().__init__()
    self._widgets: list[Widget] = []
    self._spacing = spacing
    self._alignment = alignment

    if widgets is not None:
      for widget in widgets:
        self.add_widget(widget)

  @property
  def widgets(self) -> list[Widget]:
    return self._widgets

  def add_widget(self, widget: Widget) -> None:
    self._widgets.append(widget)

  def _render(self, _):
    visible_widgets = [w for w in self._widgets if w.is_visible]

    cur_offset_x = 0

    for idx, widget in enumerate(visible_widgets):
      spacing = self._spacing if (idx > 0) else 0

      x = self._rect.x + cur_offset_x + spacing
      cur_offset_x += widget.rect.width + spacing

      if self._alignment & Alignment.TOP:
        y = self._rect.y
      elif self._alignment & Alignment.BOTTOM:
        y = self._rect.y + self._rect.height - widget.rect.height
      else:  # center
        y = self._rect.y + (self._rect.height - widget.rect.height) / 2

      # Update widget position and render
      widget.set_position(round(x), round(y))
      widget.set_parent_rect(self._rect)
      widget.render()
