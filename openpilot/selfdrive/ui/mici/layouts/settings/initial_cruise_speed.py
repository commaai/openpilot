import math
import pyray as rl
from collections.abc import Callable

from openpilot.selfdrive.ui.layouts.settings.initial_cruise_speed import get_speed_options, set_initial_cruise_speed
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, LABEL_COLOR
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialogBase
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight, TextAlignment, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.scroll_panel2 import AUTO_SCROLL_TC, GuiScrollPanel2, ScrollState
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel

ITEM_DRAG_DISTANCE = 120  # px of drag per value
ITEM_GAP = 20  # px between adjacent values
VALUE_CENTER_Y = 102
VISIBLE_VALUES = 3  # on each side of center value

# value styles
CENTER_FONT_SIZE = 120
CENTER_LETTER_SPACING = -0.03
SIDE_FONT_SIZE = 96
SIDE_LETTER_SPACING = -0.01


def lerp(start: float, end: float, t: float) -> float:
  return start + (end - start) * t


class SpeedScroller(Widget):
  def __init__(self, change_callback: Callable[[], None]):
    super().__init__()
    self._change_callback = change_callback

    self._center_font = gui_app.font(FontWeight.BOLD)
    self._side_font = gui_app.font(FontWeight.ROMAN)

    self._scroll_panel = GuiScrollPanel2(horizontal=True)
    self._scroll_panel.set_enabled(lambda: self.enabled)
    self.set_touch_valid_callback(self._scroll_panel.is_touch_valid)
    self.set_click_callback(self._on_tap)

    self._options: list[int] = []
    self._center_widths: list[float] = []
    self._side_widths: list[float] = []
    self._locked_idx: int | None = None  # value the scroller is easing onto, None while the user drags

  def show_event(self):
    super().show_event()
    self._options, self._locked_idx = get_speed_options()
    self._center_widths = [measure_text_cached(self._center_font, str(option), CENTER_FONT_SIZE,
                                               CENTER_LETTER_SPACING * CENTER_FONT_SIZE * FONT_SCALE).x for option in self._options]
    self._side_widths = [measure_text_cached(self._side_font, str(option), SIDE_FONT_SIZE,
                                             SIDE_LETTER_SPACING * SIDE_FONT_SIZE * FONT_SCALE).x for option in self._options]

    self._scroll_panel.set_offset(-self._locked_idx * ITEM_DRAG_DISTANCE)

  def _clamp_index(self, idx: int) -> int:
    return max(0, min(len(self._options) - 1, idx))

  def _lock(self, idx: int):
    self._locked_idx = self._clamp_index(idx)
    set_initial_cruise_speed(self._options[self._locked_idx])
    self._change_callback()

  def _update_state(self):
    super()._update_state()

    # when the finger releases, lock the value where the scroll will stop
    if self._scroll_panel.state == ScrollState.MANUAL_SCROLL:
      self._locked_idx = None
    elif self._locked_idx is None and self._scroll_panel.state != ScrollState.PRESSED:
      projected_offset = self._scroll_panel.get_offset() + self._scroll_panel.velocity * AUTO_SCROLL_TC
      self._lock(round(-projected_offset / ITEM_DRAG_DISTANCE))

    snap_target = -self._locked_idx * ITEM_DRAG_DISTANCE if self._locked_idx is not None else None
    content_size = (len(self._options) - 1) * ITEM_DRAG_DISTANCE + self._rect.width
    self._scroll_panel.update(self._rect, content_size, snap_target=snap_target)

  def _on_tap(self):
    if self._locked_idx is None:
      return

    # a tap on the left or the right moves one value
    tap_x = gui_app.last_mouse_event.pos.x
    center_x = self._rect.x + self._rect.width / 2
    tap_zone = self._center_widths[self._locked_idx] / 2 + ITEM_GAP / 2
    if tap_x > center_x + tap_zone:
      self._lock(self._locked_idx + 1)
    elif tap_x < center_x - tap_zone:
      self._lock(self._locked_idx - 1)

  def _render(self, _):
    center_x = self._rect.x + self._rect.width / 2
    current = -self._scroll_panel.get_offset() / ITEM_DRAG_DISTANCE

    # the style of each value changes from the center style to the side style over a distance of one value
    blends = [min(abs(idx - current), 1.0) for idx in range(len(self._options))]
    widths = [lerp(center, side, t) for center, side, t in zip(self._center_widths, self._side_widths, blends, strict=True)]

    # lay values out with a constant gap, then shift so the scroll position sits at the center
    centers = [0.0]
    for idx in range(1, len(widths)):
      centers.append(centers[-1] + widths[idx - 1] / 2 + ITEM_GAP + widths[idx] / 2)
    left_idx = max(0, min(len(widths) - 2, math.floor(current)))
    shift = center_x - lerp(centers[left_idx], centers[left_idx + 1], current - left_idx)

    # only the values near the center can be on the screen. the nearest value uses the bold font
    closest_idx = self._clamp_index(round(current))
    for idx in range(max(0, closest_idx - VISIBLE_VALUES), min(len(self._options), closest_idx + VISIBLE_VALUES + 1)):
      t = blends[idx]
      font_size = lerp(CENTER_FONT_SIZE, SIDE_FONT_SIZE, t)
      spacing = lerp(CENTER_LETTER_SPACING, SIDE_LETTER_SPACING, t) * font_size * FONT_SCALE
      color = rl.Color(255, 255, 255, int(255 * lerp(0.9, 0.45, t)))
      # use the width of the font that draws the value. this keeps the value at its center when the font changes
      if idx == closest_idx:
        font, width = self._center_font, self._center_widths[idx] * font_size / CENTER_FONT_SIZE
      else:
        font, width = self._side_font, self._side_widths[idx] * font_size / SIDE_FONT_SIZE
      position = rl.Vector2(centers[idx] + shift - width / 2, self._rect.y + VALUE_CENTER_Y - font_size * FONT_SCALE / 2)
      rl.draw_text_ex(font, str(self._options[idx]), position, font_size, spacing, color)

    # gradient on left and right
    rl.draw_rectangle_gradient_h(int(self._rect.x), int(self._rect.y), 100, int(self._rect.height), rl.BLACK, rl.BLANK)
    rl.draw_rectangle_gradient_h(int(self._rect.x + self._rect.width - 100), int(self._rect.y), 100, int(self._rect.height), rl.BLANK, rl.BLACK)


class InitialCruiseSpeedPicker(BigDialogBase):
  def __init__(self, change_callback: Callable[[], None]):
    super().__init__()
    self._unit_label = self._child(UnifiedLabel("", 36, FontWeight.BOLD, LABEL_COLOR, alignment=TextAlignment.CENTER, letter_spacing=-0.03))
    self._speed_scroller = self._child(SpeedScroller(change_callback))
    self._speed_scroller.set_enabled(lambda: self.enabled and not self.is_dismissing)

  def show_event(self):
    super().show_event()
    self._unit_label.set_text(tr("KPH") if ui_state.is_metric else tr("MPH"))

  def _render(self, _):
    self._speed_scroller.render(self._rect)
    self._unit_label.render(rl.Rectangle(self._rect.x, self._rect.y + 166, self._rect.width, self._rect.height - 166))


class InitialCruiseSpeedButton(BigButton):
  def __init__(self):
    super().__init__("initial max speed", " ", description="Max speed when engaging openpilot.\n" +
                                                           "If you are already driving faster, your current speed is used.\n" +
                                                           "Resume restores the previous max speed.")
    self._picker = InitialCruiseSpeedPicker(self.refresh)
    self.set_click_callback(lambda: gui_app.push_widget(self._picker))
    self.refresh()

  def _get_label_font_size(self):
    return 42

  def refresh(self):
    options, idx = get_speed_options()
    self.set_value(f"{options[idx]} " + (tr("kph") if ui_state.is_metric else tr("mph")))
