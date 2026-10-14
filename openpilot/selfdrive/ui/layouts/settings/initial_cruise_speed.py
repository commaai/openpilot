from collections.abc import Callable
import pyray as rl

from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.selfdrive.car.cruise import V_CRUISE_INITIAL, V_CRUISE_MAX
from openpilot.system.ui.lib.application import FontWeight, TextAlignment
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.button import Button
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import ItemAction


class InitialCruiseSpeedControl(ItemAction):
  def __init__(self):
    super().__init__(width=580)
    self._params = Params()
    self._button_size = 100
    self._font_size = 50
    self.refresh()

    self._decrease = self._child(Button("-", lambda: self._adjust_speed(-5), font_size=self._font_size,
                                       border_radius=16, text_padding=0))
    self._increase = self._child(Button("+", lambda: self._adjust_speed(5), font_size=self._font_size,
                                       border_radius=16, text_padding=0))
    self._decrease.set_enabled(lambda: self.enabled and self._value - 5 >= self._minimum)
    self._increase.set_enabled(lambda: self.enabled and self._value + 5 <= self._maximum)

  def refresh(self):
    self._metric = self._params.get_bool("IsMetric")
    self._conversion = 1 if self._metric else CV.KPH_TO_MPH
    self._minimum = round(V_CRUISE_INITIAL * self._conversion)
    self._maximum = round(V_CRUISE_MAX * self._conversion)
    self._value = max(self._minimum, round(self._params.get("InitialCruiseSpeed", return_default=True) * self._conversion))

  def set_touch_valid_callback(self, touch_callback: Callable[[], bool]) -> None:
    super().set_touch_valid_callback(touch_callback)
    self._decrease.set_touch_valid_callback(touch_callback)
    self._increase.set_touch_valid_callback(touch_callback)

  def _adjust_speed(self, delta: int):
    self._value += delta
    self._params.put("InitialCruiseSpeed", round(self._value / self._conversion), block=True)

  def _render(self, rect):
    size = self._button_size
    y = rect.y + (rect.height - size) / 2
    self._decrease.render(rl.Rectangle(rect.x, y, size, size))
    self._increase.render(rl.Rectangle(rect.x + rect.width - size, y, size, size))
    text = f"{self._value} " + (tr("km/h") if self._metric else tr("mph"))
    gui_label(rl.Rectangle(rect.x + size, y, rect.width - 2 * size, size), text, self._font_size,
              color=rl.WHITE if self.enabled else rl.GRAY, font_weight=FontWeight.MEDIUM, alignment=TextAlignment.CENTER)
