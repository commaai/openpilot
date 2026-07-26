import time
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.body.animations import FaceAnimator, ASLEEP, INQUISITIVE, NORMAL, SLEEPY

GRID_COLS = 16
GRID_ROWS = 8
DOT_RADIUS = 50 if gui_app.big_ui() else 10

MOVING_SPEED_THRESH = 0.01  # m/s
INQUISITIVE_TIMEOUT = 30.0  # seconds without movement

CHARGING_COLOR = rl.Color(49, 208, 103, 255)


# This class is used both in BIG (tizi) and small (mici) UIs
class BodyLayout(Widget):
  def __init__(self):
    super().__init__()
    self._animator = FaceAnimator(ASLEEP)
    self._turning_left = False
    self._turning_right = False
    self._last_movement_time = time.monotonic()
    self._was_onroad = False
    self._is_charging = False
    self._charge_percent = 0
    self._offroad_label = UnifiedLabel("turn on ignition to use", 95 if gui_app.big_ui() else 45, FontWeight.DISPLAY,
                                       alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
                                       alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE)

  def draw_dot_grid(self, rect: rl.Rectangle, dots: list[tuple[int, int]], color: rl.Color):
    spacing = min(rect.height / GRID_ROWS, rect.width / GRID_COLS)

    grid_w = (GRID_COLS - 1) * spacing
    grid_h = (GRID_ROWS - 1) * spacing

    offset_x = rect.x + (rect.width - grid_w) / 2
    offset_y = rect.y + (rect.height - grid_h) / 2

    for row, col in dots:
      x = int(offset_x + col * spacing)
      y = int(offset_y + row * spacing)
      rl.draw_circle(x, y, DOT_RADIUS, color)

  def _update_state(self):
    super()._update_state()

    sm = ui_state.sm

    if ui_state.is_onroad():
      if not self._was_onroad:
        self._last_movement_time = time.monotonic()
        self._was_onroad = True

      is_moving = abs(sm['carState'].vEgo) > MOVING_SPEED_THRESH
      is_livestreaming = ui_state.params.get_bool("IsLiveStreaming")
      if is_moving:
        self._last_movement_time = time.monotonic()

      if is_moving or is_livestreaming:
        animation = INQUISITIVE if time.monotonic() - self._last_movement_time > INQUISITIVE_TIMEOUT else NORMAL
        self._animator.set_animation(animation)
      else:
        self._animator.set_animation(ASLEEP)
    else:
      self._was_onroad = False
      self._animator.set_animation(ASLEEP)

    steer = sm['testJoystick'].axes[1] if len(sm['testJoystick'].axes) > 1 else 0
    self._turning_left = steer <= -0.05
    self._turning_right = steer >= 0.05

    car_state = sm['carState']
    self._is_charging = ui_state.is_onroad() and car_state.charging
    self._charge_percent = round(max(0.0, min(1.0, car_state.fuelGauge)) * 100)

  # play animation on screen tap
  def _handle_mouse_release(self, mouse_pos):
    super()._handle_mouse_release(mouse_pos)
    if ui_state.is_offroad():
      self._animator.set_animation(SLEEPY)

  def _draw_charging_indicator(self, rect: rl.Rectangle):
    scale = 1.0 if gui_app.big_ui() else 0.5
    margin = 50 * scale
    battery_width = 150 * scale
    battery_height = 74 * scale
    terminal_width = 12 * scale
    terminal_height = 30 * scale
    line_width = 6 * scale

    battery_x = rect.x + (rect.width - battery_width) / 2
    battery_y = rect.y + margin
    battery_rect = rl.Rectangle(battery_x, battery_y, battery_width, battery_height)
    terminal_rect = rl.Rectangle(battery_x + battery_width, battery_y + (battery_height - terminal_height) / 2,
                                 terminal_width, terminal_height)

    rl.draw_rectangle_rounded_lines_ex(battery_rect, 0.2, 8, line_width, rl.WHITE)
    rl.draw_rectangle_rounded(terminal_rect, 0.3, 6, rl.WHITE)

    # Fill the battery to match the reported charge level.
    inset = 10 * scale
    fill_rect = rl.Rectangle(battery_x + inset, battery_y + inset,
                             (battery_width - 2 * inset) * self._charge_percent / 100,
                             battery_height - 2 * inset)
    if fill_rect.width > 0:
      rl.draw_rectangle_rounded(fill_rect, 0.16, 6, CHARGING_COLOR)

    # Lightning bolt centered in the battery.
    center_x = battery_x + battery_width / 2
    center_y = battery_y + battery_height / 2
    bolt = [
      rl.Vector2(center_x + 4 * scale, center_y - 27 * scale),
      rl.Vector2(center_x - 21 * scale, center_y + 5 * scale),
      rl.Vector2(center_x - 3 * scale, center_y + 5 * scale),
      rl.Vector2(center_x - 8 * scale, center_y + 28 * scale),
      rl.Vector2(center_x + 22 * scale, center_y - 7 * scale),
      rl.Vector2(center_x + 3 * scale, center_y - 7 * scale),
    ]
    triangles = [
      (bolt[0], bolt[1], bolt[2]),
      (bolt[0], bolt[2], bolt[5]),
      (bolt[2], bolt[3], bolt[4]),
      (bolt[2], bolt[4], bolt[5]),
    ]
    for point_a, point_b, point_c in triangles:
      cross = ((point_b.x - point_a.x) * (point_c.y - point_a.y) -
               (point_b.y - point_a.y) * (point_c.x - point_a.x))
      if cross > 0:
        point_b, point_c = point_c, point_b
      rl.draw_triangle(point_a, point_b, point_c, rl.WHITE)

  def _render(self, rect: rl.Rectangle):
    dots = self._animator.get_dots()
    animation = self._animator._animation
    if self._turning_left and animation.left_turn_remove:
      remove_set = set(animation.left_turn_remove)
      dots = [d for d in dots if d not in remove_set]
    elif self._turning_right and animation.right_turn_remove:
      remove_set = set(animation.right_turn_remove)
      dots = [d for d in dots if d not in remove_set]
    self.draw_dot_grid(rect, dots, rl.WHITE)

    if self._is_charging:
      self._draw_charging_indicator(rect)

    if ui_state.is_offroad():
      rl.draw_rectangle(int(self.rect.x), int(self.rect.y), int(self.rect.width), int(self.rect.height), rl.Color(0, 0, 0, 175))
      upper_half = rl.Rectangle(rect.x, rect.y, rect.width, rect.height / 2)
      self._offroad_label.render(upper_half)
