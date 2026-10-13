from collections.abc import Callable
import pyray as rl

from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.selfdrive.car.cruise import V_CRUISE_INITIAL, V_CRUISE_MIN, V_CRUISE_MAX
from openpilot.system.ui.lib.application import gui_app, FontWeight
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
    self._speed_kph = self._params.get("InitialCruiseSpeed", return_default=True)
    self._text = str(round(self._speed_kph * self._conversion))
    self._replace = True
    self._edited = False
    self._scale = min(gui_app.width / 536, gui_app.height / 240)

    self._keys = []
    for key in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "default", "0", "backspace"]:
      text = tr("Default") if key == "default" else ("" if key == "backspace" else key)
      icon = gui_app.texture("icons/backspace.png", int(28 * self._scale), int(28 * self._scale)) if key == "backspace" else None
      button = self._child(Button(text, lambda k=key: self._key_press(k),
                                 font_size=int((15 if key == "default" else 28) * self._scale),
                                 border_radius=int(12 * self._scale), text_padding=int(4 * self._scale), icon=icon))
      button.set_enabled(lambda: self.enabled and not self.is_dismissing)
      self._keys.append(button)

    self._cancel = self._child(Button(lambda: tr("Cancel"), self.dismiss, font_size=int(19 * self._scale),
                                     border_radius=int(12 * self._scale)))
    self._save = self._child(Button(lambda: tr("Save"), self._save_speed, font_size=int(19 * self._scale),
                                   border_radius=int(12 * self._scale), button_style=ButtonStyle.PRIMARY))
    self._cancel.set_enabled(lambda: self.enabled and not self.is_dismissing)
    self._save.set_enabled(lambda: self.enabled and not self.is_dismissing and self._valid())

  def _valid(self):
    return bool(self._text) and self._minimum <= int(self._text) <= self._maximum

  def _key_press(self, key: str):
    if key == "default":
      self._speed_kph = V_CRUISE_INITIAL
      self._text = str(round(V_CRUISE_INITIAL * self._conversion))
      self._replace = True
      self._edited = False
      return
    if key == "backspace":
      self._text = self._text[:-1]
    elif len(self._text) < 3 or self._replace:
      self._text = key if self._replace or self._text == "0" else self._text + key
    self._replace = False
    self._edited = True

  def _save_speed(self):
    if self._valid():
      speed = round(int(self._text) / self._conversion) if self._edited else self._speed_kph
      self._params.put("InitialCruiseSpeed", speed, block=True)
      self.dismiss(self._callback)

  def _render(self, rect):
    scale = self._scale
    x = rect.x + (rect.width - 536 * scale) / 2

    def area(left, top, width, height):
      return rl.Rectangle(x + left * scale, rect.y + (rect.height - 240 * scale) / 2 + top * scale, width * scale, height * scale)

    gui_label(area(16, 19, 238, 27), tr("Initial cruise speed"), int(23 * scale), font_weight=FontWeight.BOLD)
    gui_label(area(16, 52, 168, 76), self._text or "—", int(72 * scale), font_weight=FontWeight.BOLD)
    gui_label(area(184, 91, 64, 30), tr("km/h") if self._metric else tr("mph"), int(23 * scale), color=rl.GRAY)
    range_color = rl.GRAY if self._valid() else rl.Color(255, 130, 110, 255)
    gui_label(area(16, 130, 238, 22), tr("Enter {min}–{max}").format(min=self._minimum, max=self._maximum), int(16 * scale), color=range_color)
    gui_label(area(16, 155, 238, 20), tr("Minimum speed when engaging."), int(15 * scale), color=rl.GRAY)

    self._cancel.render(area(16, 188, 112, 40))
    self._save.render(area(136, 188, 112, 40))

    for i, button in enumerate(self._keys):
      row, col = divmod(i, 3)
      button.render(area(272 + col * 84, 20 + row * 54, 76, 46))
