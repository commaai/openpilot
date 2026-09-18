import pyray as rl

from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel


class InfoLayoutMici(Widget):
  def __init__(self, titles: tuple[str, str], values: tuple[str, str] = ("", ""), *, width: int = 360,
               scroll: tuple[bool, bool] = (False, False)):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, width, 180))

    subheader_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))
    max_width = int(self._rect.width - 20)
    self._title_labels = [
      UnifiedLabel(title, 48, max_width=max_width, font_weight=FontWeight.DISPLAY, wrap_text=False)
      for title in titles
    ]
    self._value_labels = [
      UnifiedLabel(value, 36, max_width=max_width, text_color=subheader_color,
                   font_weight=FontWeight.ROMAN, wrap_text=False, scroll=scroll_value)
      for value, scroll_value in zip(values, scroll, strict=True)
    ]

  def set_value(self, index: int, value: str):
    self._value_labels[index].set_text(value)

  def _render(self, _):
    for title, value, title_y, value_y in zip(self._title_labels, self._value_labels, (-10, 114 - 30), (68 - 25, 161 - 25), strict=True):
      title.set_position(self._rect.x + 20, self._rect.y + title_y)
      title.render()
      value.set_position(self._rect.x + 20, self._rect.y + value_y)
      value.render()
