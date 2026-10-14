import math
import time
import pyray as rl

from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.selfdrive.car.cruise import V_CRUISE_INITIAL, V_CRUISE_MAX
from openpilot.system.ui.lib.application import FontWeight, TextAlignment
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.button import Button
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import ItemAction


class RepeatButton(Button):
  def __init__(self, text, callback, font_size):
    super().__init__(text, callback, font_size=font_size, border_radius=16, text_padding=0)
    self._repeat_at: float | None = None
    self._repeated = False

  def cancel_repeat(self):
    self._repeat_at = None
    self._repeated = True

  def _handle_mouse_press(self, mouse_pos):
    self._repeat_at = time.monotonic() + 0.5
    self._repeated = False

  def _handle_mouse_release(self, mouse_pos):
    if not self._repeated:
      super()._handle_mouse_release(mouse_pos)

  def _process_mouse_events(self):
    super()._process_mouse_events()
    if not self.is_pressed or not self._touch_valid():
      self.cancel_repeat()
    elif self._repeat_at is not None and time.monotonic() >= self._repeat_at:
      self._repeated = True
      self._repeat_at = time.monotonic() + 0.2
      if self._click_callback:
        self._click_callback()


class InitialCruiseSpeedControl(ItemAction):
  def __init__(self, compact: bool = False):
    super().__init__(width=322 if compact else 580)
    self._params = Params()
    self._button_size = 64 if compact else 100
    self._font_size = 36 if compact else 50

    self._decrease = self._child(RepeatButton("-", lambda: self._adjust_speed(-5), self._font_size))
    self._increase = self._child(RepeatButton("+", lambda: self._adjust_speed(5), self._font_size))
    self._decrease.set_enabled(lambda: self.enabled and self._value > self._minimum)
    self._increase.set_enabled(lambda: self.enabled and self._value < self._maximum)
    self._decrease.set_touch_valid_callback(self._touch_valid)
    self._increase.set_touch_valid_callback(self._touch_valid)
    self.refresh()

  def refresh(self, is_metric: bool | None = None):
    for button in (self._decrease, self._increase):
      button.cancel_repeat()
    self._metric = self._params.get_bool("IsMetric") if is_metric is None else is_metric
    self._conversion = 1 if self._metric else CV.KPH_TO_MPH
    self._minimum = round(V_CRUISE_INITIAL * self._conversion)
    self._maximum = round(V_CRUISE_MAX * self._conversion)
    self._value = max(self._minimum, round(self._params.get("InitialCruiseSpeedKph", return_default=True) * self._conversion))

  def _adjust_speed(self, delta: int):
    round_func = math.floor if delta > 0 else math.ceil
    self._value = round_func(self._value / 5) * 5 + delta
    self._params.put("InitialCruiseSpeedKph", round(self._value / self._conversion), block=True)

  def _render(self, rect):
    size = self._button_size
    y = rect.y + (rect.height - size) / 2
    self._decrease.render(rl.Rectangle(rect.x, y, size, size))
    self._increase.render(rl.Rectangle(rect.x + rect.width - size, y, size, size))
    text = f"{self._value} " + (tr("km/h") if self._metric else tr("mph"))
    gui_label(rl.Rectangle(rect.x + size, y, rect.width - 2 * size, size), text, self._font_size,
              color=rl.WHITE if self.enabled else rl.GRAY, font_weight=FontWeight.MEDIUM, alignment=TextAlignment.CENTER)
