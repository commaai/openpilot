import pyray as rl

from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel


class InfoLayoutMici(Widget):
  def __init__(self, title1: str, subtext1: str, title2: str, subtext2: str, *, width: int = 360):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, width, 180))

    max_width = int(self._rect.width - 20)
    self._subheader_color = rl.Color(255, 255, 255, int(255 * 0.9 * 0.65))

    self.title1 = UnifiedLabel(title1, 48, max_width=max_width, font_weight=FontWeight.DISPLAY, wrap_text=False)
    self.subtext1 = UnifiedLabel(subtext1, 36, max_width=max_width, text_color=self._subheader_color,
                                font_weight=FontWeight.ROMAN, wrap_text=False, scroll=True)
    self.title2 = UnifiedLabel(title2, 48, max_width=max_width, font_weight=FontWeight.DISPLAY, wrap_text=False)
    self.subtext2 = UnifiedLabel(subtext2, 36, max_width=max_width, text_color=self._subheader_color,
                                font_weight=FontWeight.ROMAN, wrap_text=False, scroll=True)

  def _layout(self):
    self.title1.set_position(self._rect.x + 20, self._rect.y - 10)
    self.subtext1.set_position(self._rect.x + 20, self._rect.y + 68 - 25)
    self.title2.set_position(self._rect.x + 20, self._rect.y + 114 - 30)
    self.subtext2.set_position(self._rect.x + 20, self._rect.y + 161 - 25)

  def _render(self, rect: rl.Rectangle):
    for widget in [self.title1, self.subtext1, self.title2, self.subtext2]:
      widget.render()
