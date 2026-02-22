from enum import IntEnum
from openpilot.system.ui.widgets import Widget


class Alignment(IntEnum):
  LEFT = 0
  H_CENTER = 1
  RIGHT = 2

  TOP = 3
  V_CENTER = 4
  BOTTOM = 5


class HBoxLayout(Widget):
  """
  A Widget that lays out child Widgets.
  """

  def __init__(self, widgets: list[Widget] | None = None, spacing: int = 0,
               alignment: Alignment = Alignment.LEFT):
    super().__init__()
    self._widgets: list[Widget] = []
    self._spacing = spacing
    self._alignment = alignment

    self._visible_widgets: list[Widget] = []  # tracks offscreen widgets for performance

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

    cur_x = 0  #self._rect.x
    # cur_y = self._rect.y

    for idx, widget in enumerate(visible_widgets):
      spacing = self._spacing if (idx > 0) else 0  # self._pad_start

      x = self._rect.x + cur_x + spacing
      y = self._rect.y + (self._rect.height - widget.rect.height) / 2
      cur_x += widget.rect.width + spacing

      widget.set_position(round(x), round(y))
      widget.set_parent_rect(self._rect)

      widget.render()

