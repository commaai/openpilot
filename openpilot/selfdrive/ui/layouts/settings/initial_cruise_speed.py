import time
import pyray as rl

from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_INITIAL, V_CRUISE_MAX
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, TextAlignment
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import ItemAction, BUTTON_HEIGHT, BUTTON_BORDER_RADIUS

SPEED_STEP = 5
STEP_BUTTON_WIDTH = 160
VALUE_FONT_SIZE = 50
REPEAT_DELAY = 0.5  # seconds held before repeating
REPEAT_INTERVAL = 0.2  # seconds


def get_speed_conversion() -> float:
  return 1 if ui_state.is_metric else CV.KPH_TO_MPH


def get_speed_unit() -> str:
  return tr("km/h") if ui_state.is_metric else tr("mph")


def get_speed_options() -> list[int]:
  conversion = get_speed_conversion()
  return list(range(round(V_CRUISE_INITIAL * conversion), round(V_CRUISE_MAX * conversion) + 1, SPEED_STEP))


def get_selected_speed_index(options: list[int]) -> int:
  speed = ui_state.initial_cruise_speed * get_speed_conversion()
  return min(range(len(options)), key=lambda idx: abs(options[idx] - speed))


def set_initial_cruise_speed(speed: int) -> None:
  """Save an initial cruise speed in display units"""
  ui_state.params.put("InitialCruiseSpeedKph", round(speed / get_speed_conversion()), block=True)
  ui_state.update_params()


class RepeatButton(Button):
  def __init__(self, text, callback, font_size):
    super().__init__(text, callback, font_size=font_size, button_style=ButtonStyle.PILL, border_radius=BUTTON_BORDER_RADIUS, text_padding=0)
    self._repeat_at: float | None = None
    self._repeated = False

  def _handle_mouse_press(self, mouse_pos):
    super()._handle_mouse_press(mouse_pos)
    self._repeat_at = time.monotonic() + REPEAT_DELAY
    self._repeated = False

  def _handle_mouse_release(self, mouse_pos):
    if not self._repeated:
      super()._handle_mouse_release(mouse_pos)

  def _process_mouse_events(self):
    super()._process_mouse_events()
    if not self.is_pressed or not self._touch_valid():
      # released, scrolled away, or repeated: the release shouldn't step again
      self._repeat_at = None
      self._repeated = True
    elif self._repeat_at is not None and time.monotonic() >= self._repeat_at:
      self._repeated = True
      self._repeat_at = time.monotonic() + REPEAT_INTERVAL
      if self._click_callback:
        self._click_callback()


class InitialCruiseSpeedControl(ItemAction):
  def __init__(self):
    super().__init__(width=580)
    self._options: list[int] = []
    self._selected_idx = 0

    self._decrease = self._child(RepeatButton("-", lambda: self._adjust_speed(-1), VALUE_FONT_SIZE))
    self._increase = self._child(RepeatButton("+", lambda: self._adjust_speed(1), VALUE_FONT_SIZE))
    self._decrease.set_enabled(lambda: self.enabled and self._selected_idx > 0)
    self._increase.set_enabled(lambda: self.enabled and self._selected_idx < len(self._options) - 1)
    self._decrease.set_touch_valid_callback(self._touch_valid)
    self._increase.set_touch_valid_callback(self._touch_valid)
    self.refresh()

  def refresh(self):
    self._options = get_speed_options()
    self._selected_idx = get_selected_speed_index(self._options)

  def _adjust_speed(self, step: int):
    self._selected_idx += step
    set_initial_cruise_speed(self._options[self._selected_idx])

  def _render(self, rect):
    y = rect.y + (rect.height - BUTTON_HEIGHT) / 2
    self._decrease.render(rl.Rectangle(rect.x, y, STEP_BUTTON_WIDTH, BUTTON_HEIGHT))
    self._increase.render(rl.Rectangle(rect.x + rect.width - STEP_BUTTON_WIDTH, y, STEP_BUTTON_WIDTH, BUTTON_HEIGHT))
    text = f"{self._options[self._selected_idx]} {get_speed_unit()}"
    gui_label(rl.Rectangle(rect.x + STEP_BUTTON_WIDTH, y, rect.width - 2 * STEP_BUTTON_WIDTH, BUTTON_HEIGHT), text, VALUE_FONT_SIZE,
              color=rl.WHITE if self.enabled else rl.GRAY, font_weight=FontWeight.MEDIUM, alignment=TextAlignment.CENTER)
