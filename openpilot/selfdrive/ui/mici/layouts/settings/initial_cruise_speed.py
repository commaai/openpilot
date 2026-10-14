import math
import pyray as rl
from collections.abc import Callable

from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.selfdrive.car.cruise import V_CRUISE_INITIAL, V_CRUISE_MAX
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, COMPLICATION_GREY, LABEL_COLOR
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.scroll_panel2 import AUTO_SCROLL_TC, MIN_DRAG_PIXELS, GuiScrollPanel2, ScrollState
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets.nav_widget import NavWidget

SPEED_STEP = 5
ITEM_DRAG_DISTANCE = 120  # px of drag per value
ITEM_GAP = 22  # px between neighboring values
VALUE_FONT_SIZE = 133
VALUE_LETTER_SPACING = -0.05 * VALUE_FONT_SIZE * FONT_SCALE  # -5% of the font size, between digits
UNIT_FONT_SIZE = 42
VALUE_CENTER_Y = 102
UNIT_CENTER_Y = 183
# values shrink and darken with distance from the center, in items
MIN_SCALE = 0.43
SCALE_DECAY = 1.14
COLOR_DECAY = 1.4
EDGE_COLOR = rl.Color(0x3A, 0x3A, 0x3A, 255)


def get_conversion() -> float:
  return 1 if ui_state.is_metric else CV.KPH_TO_MPH


def get_unit() -> str:
  return tr("km/h") if ui_state.is_metric else tr("mph")


def get_speed_options(conversion: float) -> list[int]:
  return list(range(round(V_CRUISE_INITIAL * conversion), round(V_CRUISE_MAX * conversion) + 1, SPEED_STEP))


def get_selected_index(params: Params, options: list[int], conversion: float) -> int:
  speed = params.get("InitialCruiseSpeed", return_default=True) * conversion
  return min(range(len(options)), key=lambda idx: abs(options[idx] - speed))


def lerp_color(start: rl.Color, end: rl.Color, t: float) -> rl.Color:
  return rl.Color(int(start.r + (end.r - start.r) * t), int(start.g + (end.g - start.g) * t),
                  int(start.b + (end.b - start.b) * t), int(start.a + (end.a - start.a) * t))


class InitialCruiseSpeedPicker(NavWidget):
  def __init__(self, change_callback: Callable[[], None]):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    self._params = Params()
    self._change_callback = change_callback

    self._value_font = gui_app.font(FontWeight.ROMAN)
    self._center_value_font = gui_app.font(FontWeight.SEMI_BOLD)
    self._unit_font = gui_app.font(FontWeight.MEDIUM)

    self._scroll_panel = GuiScrollPanel2(horizontal=True)
    self._scroll_panel.set_enabled(lambda: self.enabled and not self.is_dismissing)

    self._conversion = 1.0
    self._options: list[int] = []
    self._text_widths: list[float] = []
    self._item_centers: list[float] = []
    self._selected_idx = 0
    self._locked_idx: int | None = None  # value the picker is easing onto, None while the user drags
    self._press_pos: MousePos | None = None

  def show_event(self):
    super().show_event()
    self._conversion = get_conversion()
    self._options = get_speed_options(self._conversion)
    self._text_widths = [measure_text_cached(self._value_font, str(option), VALUE_FONT_SIZE, VALUE_LETTER_SPACING).x for option in self._options]
    self._item_centers = [0.0] * len(self._options)
    self._selected_idx = get_selected_index(self._params, self._options, self._conversion)
    self._locked_idx = self._selected_idx
    self._press_pos = None

    # open at the lowest value and glide to the selected one
    self._scroll_panel.set_offset(0.0)

  def _clamp_index(self, idx: int) -> int:
    return max(0, min(len(self._options) - 1, idx))

  def _select(self, idx: int):
    self._locked_idx = idx
    if idx != self._selected_idx:
      self._selected_idx = idx
      self._params.put("InitialCruiseSpeed", round(self._options[idx] / self._conversion), block=True)
      self._change_callback()

  def _update_state(self):
    super()._update_state()

    # Once released, lock onto the value the fling would coast to, like an iOS picker
    if self._scroll_panel.state in (ScrollState.PRESSED, ScrollState.MANUAL_SCROLL):
      self._locked_idx = None
    elif self._locked_idx is None:
      projected_offset = self._scroll_panel.get_offset() + self._scroll_panel.velocity * AUTO_SCROLL_TC
      self._select(self._clamp_index(round(-projected_offset / ITEM_DRAG_DISTANCE)))

    snap_target = -self._locked_idx * ITEM_DRAG_DISTANCE if self._locked_idx is not None else None
    content_size = (len(self._options) - 1) * ITEM_DRAG_DISTANCE + self._rect.width
    self._scroll_panel.update(self._rect, content_size, snap_target=snap_target)

  def _handle_mouse_press(self, mouse_pos: MousePos):
    super()._handle_mouse_press(mouse_pos)
    # a press that stops a fling isn't a tap
    self._press_pos = mouse_pos if self._scroll_panel.state == ScrollState.PRESSED else None

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    if self._press_pos is None or self.is_dismissing:
      return

    # tap a neighboring value to select it
    if abs(mouse_pos.x - self._press_pos.x) <= MIN_DRAG_PIXELS and abs(mouse_pos.y - self._press_pos.y) <= MIN_DRAG_PIXELS:
      self._select(min(range(len(self._options)), key=lambda idx: abs(self._item_centers[idx] - mouse_pos.x)))
    self._press_pos = None

  def _render(self, _):
    center_x = self._rect.x + self._rect.width / 2
    current = -self._scroll_panel.get_offset() / ITEM_DRAG_DISTANCE

    scales = [MIN_SCALE + (1.0 - MIN_SCALE) * math.exp(-SCALE_DECAY * abs(idx - current)) for idx in range(len(self._options))]
    widths = [width * scale for width, scale in zip(self._text_widths, scales, strict=True)]

    # lay values out from the pair straddling the center with a constant gap, so smaller values pack closer like a wheel
    left_idx = max(0, min(len(self._options) - 2, math.floor(current)))
    step = widths[left_idx] / 2 + ITEM_GAP + widths[left_idx + 1] / 2
    self._item_centers[left_idx] = center_x - (current - left_idx) * step
    for idx in range(left_idx + 1, len(self._options)):
      self._item_centers[idx] = self._item_centers[idx - 1] + widths[idx - 1] / 2 + ITEM_GAP + widths[idx] / 2
    for idx in range(left_idx - 1, -1, -1):
      self._item_centers[idx] = self._item_centers[idx + 1] - widths[idx + 1] / 2 - ITEM_GAP - widths[idx] / 2

    # the value closest to the center snaps to a heavier weight
    center_idx = max(0, min(len(self._options) - 1, round(current)))

    for idx, option in enumerate(self._options):
      if abs(self._item_centers[idx] - center_x) > self._rect.width / 2 + widths[idx] / 2:
        continue

      font = self._center_value_font if idx == center_idx else self._value_font
      text_size = measure_text_cached(font, str(option), VALUE_FONT_SIZE, VALUE_LETTER_SPACING)
      position = rl.Vector2(self._item_centers[idx] - text_size.x * scales[idx] / 2, self._rect.y + VALUE_CENTER_Y - text_size.y * scales[idx] / 2)
      color = lerp_color(EDGE_COLOR, LABEL_COLOR, math.exp(-COLOR_DECAY * abs(idx - current)))
      rl.draw_text_ex(font, str(option), position, VALUE_FONT_SIZE * scales[idx], VALUE_LETTER_SPACING * scales[idx], color)

    unit = get_unit()
    unit_size = measure_text_cached(self._unit_font, unit, UNIT_FONT_SIZE)
    unit_position = rl.Vector2(center_x - unit_size.x / 2, self._rect.y + UNIT_CENTER_Y - unit_size.y / 2)
    rl.draw_text_ex(self._unit_font, unit, unit_position, UNIT_FONT_SIZE, 0, COMPLICATION_GREY)


class InitialCruiseSpeedButton(BigButton):
  def __init__(self):
    super().__init__("initial max speed", " ")
    self._params = Params()
    self._picker = InitialCruiseSpeedPicker(self.refresh)
    self.set_click_callback(lambda: gui_app.push_widget(self._picker))
    self.refresh()

  def refresh(self):
    conversion = get_conversion()
    options = get_speed_options(conversion)
    self.set_value(f"{options[get_selected_index(self._params, options, conversion)]} {get_unit()}")
