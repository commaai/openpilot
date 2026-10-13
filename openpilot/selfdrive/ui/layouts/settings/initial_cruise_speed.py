from collections.abc import Callable
import pyray as rl

from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.selfdrive.car.cruise import V_CRUISE_MIN, V_CRUISE_MAX
from openpilot.system.ui.lib.application import gui_app, FontWeight, TextAlignment
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.nav_widget import NavWidget


def initial_cruise_speed_text(params: Params) -> str:
  speed = params.get("InitialCruiseSpeed", return_default=True)
  is_metric = params.get_bool("IsMetric")
  value = round(speed if is_metric else speed * CV.KPH_TO_MPH)
  return f"{value} " + (tr("km/h") if is_metric else tr("mph"))


class InitialCruiseSpeedDialog(NavWidget):
  BACK_TOUCH_AREA_PERCENTAGE = 0.1

  def __init__(self, callback: Callable[[], None]):
    super().__init__()
    self._params = Params()
    self._callback = callback
    self._metric = self._params.get_bool("IsMetric")
    self._conversion = 1 if self._metric else CV.KPH_TO_MPH
    self._minimum = round(V_CRUISE_MIN * self._conversion)
    self._maximum = round(V_CRUISE_MAX * self._conversion)
    self._value = round(self._params.get("InitialCruiseSpeed", return_default=True) * self._conversion)
    self._scale = min(gui_app.width / 536, gui_app.height / 240)

    self._decrease = self._child(Button("-", lambda: self._adjust_speed(-5), font_size=int(48 * self._scale),
                                       border_radius=int(16 * self._scale)))
    self._increase = self._child(Button("+", lambda: self._adjust_speed(5), font_size=int(48 * self._scale),
                                       border_radius=int(16 * self._scale)))
    self._done = self._child(Button(lambda: tr("Done"), self.dismiss, font_size=int(19 * self._scale),
                                   border_radius=int(12 * self._scale), button_style=ButtonStyle.PRIMARY))
    self._decrease.set_enabled(lambda: self.enabled and not self.is_dismissing and self._value - 5 >= self._minimum)
    self._increase.set_enabled(lambda: self.enabled and not self.is_dismissing and self._value + 5 <= self._maximum)
    self._done.set_enabled(lambda: self.enabled and not self.is_dismissing)

  def _adjust_speed(self, delta: int):
    self._value += delta
    self._params.put("InitialCruiseSpeed", round(self._value / self._conversion), block=True)
    self._callback()

  def _render(self, rect):
    scale = self._scale
    x = rect.x + (rect.width - 536 * scale) / 2

    def area(left, top, width, height):
      return rl.Rectangle(x + left * scale, rect.y + (rect.height - 240 * scale) / 2 + top * scale, width * scale, height * scale)

    gui_label(area(24, 22, 488, 30), tr("Initial max speed"), int(26 * scale),
              font_weight=FontWeight.BOLD, alignment=TextAlignment.CENTER)
    gui_label(area(136, 66, 264, 76), str(self._value), int(72 * scale), font_weight=FontWeight.BOLD, alignment=TextAlignment.CENTER)
    gui_label(area(136, 145, 264, 26), tr("km/h") if self._metric else tr("mph"), int(23 * scale),
              color=rl.GRAY, alignment=TextAlignment.CENTER)

    self._decrease.render(area(24, 76, 96, 96))
    self._increase.render(area(416, 76, 96, 96))
    self._done.render(area(24, 188, 488, 40))
