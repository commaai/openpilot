import math
import pyray as rl
from collections.abc import Callable

from openpilot.selfdrive.ui.layouts.settings.initial_cruise_speed import get_speed_options, set_initial_cruise_speed
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, LABEL_COLOR
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos, FONT_SCALE
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.scroll_panel2 import AUTO_SCROLL_TC, MIN_DRAG_PIXELS, GuiScrollPanel2, ScrollState
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets.nav_widget import NavWidget

ITEM_DRAG_DISTANCE = 120  # px of drag per value
ITEM_GAP = 20  # px between neighboring values
VALUE_CENTER_Y = 102
UNIT_FONT_SIZE = 36
UNIT_LETTER_SPACING = -0.03
VISIBLE_VALUES = 3  # on each side of the closest value

# the value nearest the center, and every other value
CENTER_FONT_SIZE = 120
CENTER_LETTER_SPACING = -0.03  # of the font size
SIDE_FONT_SIZE = 96
SIDE_LETTER_SPACING = -0.01


def lerp(start: float, end: float, t: float) -> float:
  return start + (end - start) * t


def letter_spacing(font_size: float, spacing: float) -> float:
  return spacing * font_size * FONT_SCALE


class InitialCruiseSpeedPicker(NavWidget):
  def __init__(self, change_callback: Callable[[], None]):
    super().__init__()
    self.set_rect(rl.Rectangle(0, 0, gui_app.width, gui_app.height))
    self._change_callback = change_callback

    self._center_font = gui_app.font(FontWeight.BOLD)
    self._side_font = gui_app.font(FontWeight.ROMAN)
    self._unit_font = gui_app.font(FontWeight.BOLD)

    self._scroll_panel = GuiScrollPanel2(horizontal=True)
    self._scroll_panel.set_enabled(lambda: self.enabled and not self.is_dismissing)

    self._unit = ""
    self._options: list[int] = []
    self._center_widths: list[float] = []
    self._side_widths: list[float] = []
    self._locked_idx: int | None = None  # value the picker is easing onto, None while the user drags
    self._tap_start: MousePos | None = None

  def show_event(self):
    super().show_event()
    self._unit = tr("KPH") if ui_state.is_metric else tr("MPH")
    self._options, self._locked_idx = get_speed_options()
    self._center_widths = [measure_text_cached(self._center_font, str(option), CENTER_FONT_SIZE,
                                               letter_spacing(CENTER_FONT_SIZE, CENTER_LETTER_SPACING)).x for option in self._options]
    self._side_widths = [measure_text_cached(self._side_font, str(option), SIDE_FONT_SIZE,
                                             letter_spacing(SIDE_FONT_SIZE, SIDE_LETTER_SPACING)).x for option in self._options]

    self._scroll_panel.set_offset(-self._locked_idx * ITEM_DRAG_DISTANCE)

  def _clamp_index(self, idx: int) -> int:
    return max(0, min(len(self._options) - 1, idx))

  def _lock(self, idx: int):
    self._locked_idx = self._clamp_index(idx)
    set_initial_cruise_speed(self._options[self._locked_idx])
    self._change_callback()

  def _update_state(self):
    super()._update_state()

    # Once released, lock onto the value the fling would coast to, like an iOS picker
    if self._scroll_panel.state == ScrollState.MANUAL_SCROLL:
      self._locked_idx = None
    elif self._locked_idx is None and self._scroll_panel.state != ScrollState.PRESSED:
      projected_offset = self._scroll_panel.get_offset() + self._scroll_panel.velocity * AUTO_SCROLL_TC
      self._lock(round(-projected_offset / ITEM_DRAG_DISTANCE))

    snap_target = -self._locked_idx * ITEM_DRAG_DISTANCE if self._locked_idx is not None else None
    content_size = (len(self._options) - 1) * ITEM_DRAG_DISTANCE + self._rect.width
    self._scroll_panel.update(self._rect, content_size, snap_target=snap_target)

  def _handle_mouse_press(self, mouse_pos: MousePos):
    super()._handle_mouse_press(mouse_pos)
    # a press that stops a fling isn't a tap
    self._tap_start = mouse_pos if self._scroll_panel.state == ScrollState.PRESSED else None

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    tap_start, self._tap_start = self._tap_start, None
    if tap_start is None or self.is_dismissing:
      return
    if abs(mouse_pos.x - tap_start.x) > MIN_DRAG_PIXELS or abs(mouse_pos.y - tap_start.y) > MIN_DRAG_PIXELS:
      return

    # tap either side of the center value to step one value, from where the picker is heading
    idx = self._locked_idx if self._locked_idx is not None else self._clamp_index(round(-self._scroll_panel.get_offset() / ITEM_DRAG_DISTANCE))
    center_x = self._rect.x + self._rect.width / 2
    tap_zone = self._center_widths[idx] / 2 + ITEM_GAP / 2
    if mouse_pos.x > center_x + tap_zone:
      self._lock(idx + 1)
    elif mouse_pos.x < center_x - tap_zone:
      self._lock(idx - 1)

  def _render(self, _):
    center_x = self._rect.x + self._rect.width / 2
    current = -self._scroll_panel.get_offset() / ITEM_DRAG_DISTANCE

    # values blend from the center style to the side style over one value from the center
    blends = [min(abs(idx - current), 1.0) for idx in range(len(self._options))]
    widths = [lerp(center, side, t) for center, side, t in zip(self._center_widths, self._side_widths, blends, strict=True)]

    # lay values out with a constant gap, then shift so the scroll position sits at the center
    centers = [0.0]
    for idx in range(1, len(widths)):
      centers.append(centers[-1] + widths[idx - 1] / 2 + ITEM_GAP + widths[idx] / 2)
    left_idx = max(0, min(len(widths) - 2, math.floor(current)))
    shift = center_x - lerp(centers[left_idx], centers[left_idx + 1], current - left_idx)

    # only values near the center can be on screen. the closest one snaps to bold
    closest_idx = self._clamp_index(round(current))
    for idx in range(max(0, closest_idx - VISIBLE_VALUES), min(len(self._options), closest_idx + VISIBLE_VALUES + 1)):
      t = blends[idx]
      font_size = lerp(CENTER_FONT_SIZE, SIDE_FONT_SIZE, t)
      spacing = letter_spacing(font_size, lerp(CENTER_LETTER_SPACING, SIDE_LETTER_SPACING, t))
      color = rl.Color(255, 255, 255, int(255 * lerp(0.9, 0.9 * 0.65, t)))
      # center on the width in the font it's drawn with, scaled to its size, so switching fonts grows it evenly
      if idx == closest_idx:
        font, width = self._center_font, self._center_widths[idx] * font_size / CENTER_FONT_SIZE
      else:
        font, width = self._side_font, self._side_widths[idx] * font_size / SIDE_FONT_SIZE
      position = rl.Vector2(centers[idx] + shift - width / 2, self._rect.y + VALUE_CENTER_Y - font_size * FONT_SCALE / 2)
      rl.draw_text_ex(font, str(self._options[idx]), position, font_size, spacing, color)

    # fade values out into the left and right edges
    rl.draw_rectangle_gradient_h(int(self._rect.x), int(self._rect.y), 120, int(self._rect.height), rl.BLACK, rl.BLANK)
    rl.draw_rectangle_gradient_h(int(self._rect.x + self._rect.width - 120), int(self._rect.y), 120, int(self._rect.height), rl.BLANK, rl.BLACK)

    unit_spacing = letter_spacing(UNIT_FONT_SIZE, UNIT_LETTER_SPACING)
    unit_size = measure_text_cached(self._unit_font, self._unit, UNIT_FONT_SIZE, unit_spacing)
    unit_position = rl.Vector2(center_x - unit_size.x / 2, self._rect.y + 166)
    rl.draw_text_ex(self._unit_font, self._unit, unit_position, UNIT_FONT_SIZE, unit_spacing, LABEL_COLOR)


class InitialCruiseSpeedButton(BigButton):
  def __init__(self):
    super().__init__("initial max speed", " ", description="Max speed when engaging openpilot.\n" +
                                                          "If you are already driving faster, your current speed is used.\n" +
                                                          "Resume restores the previous max speed.")
    self._picker = InitialCruiseSpeedPicker(self.refresh)
    self.set_click_callback(lambda: gui_app.push_widget(self._picker))
    self.refresh()

  def _get_label_font_size(self):
    # size as a long title so the wrapped title doesn't crowd the speed
    return 42

  def refresh(self):
    options, idx = get_speed_options()
    self.set_value(f"{options[idx]} " + (tr("kph") if ui_state.is_metric else tr("mph")))
