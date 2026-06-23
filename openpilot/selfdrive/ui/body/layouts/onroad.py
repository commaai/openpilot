import time
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.body.animations import FaceAnimator, ASLEEP, INQUISITIVE, NORMAL, SLEEPY

GRID_COLS = 16
GRID_ROWS = 8
DOT_RADIUS = 50 if gui_app.big_ui() else 10

IDLE_TIMEOUT = 30.0        # seconds of no joystick input before playing INQUISITIVE
IDLE_STEER_THRESH = 0.5    # degrees — below this counts as no input
IDLE_SPEED_THRESH = 0.01   # m/s — below this counts as no input


# This class is used both in BIG (tizi) and small (mici) UIs
class BodyLayout(Widget):
  def __init__(self):
    super().__init__()
    self._animator = FaceAnimator(ASLEEP)
    self._turning_left = False
    self._turning_right = False
    self._last_input_time = time.monotonic()
    self._was_active = False
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
      if not self._was_active:
        self._last_input_time = time.monotonic()
        self._was_active = True

      cs = sm['carState']
      has_input = abs(cs.steeringAngleDeg) > IDLE_STEER_THRESH or abs(cs.vEgo) > IDLE_SPEED_THRESH
      if has_input:
        self._last_input_time = time.monotonic()

      if time.monotonic() - self._last_input_time > IDLE_TIMEOUT:
        self._animator.set_animation(INQUISITIVE)
      else:
        self._animator.set_animation(NORMAL)
    else:
      self._was_active = False
      self._animator.set_animation(ASLEEP)

    steer = sm['testJoystick'].axes[1] if len(sm['testJoystick'].axes) > 1 else 0
    self._turning_left = steer <= -0.05
    self._turning_right = steer >= 0.05

  # play animation on screen tap
  def _handle_mouse_release(self, mouse_pos):
    super()._handle_mouse_release(mouse_pos)
    if not self._was_active:
      self._animator.set_animation(SLEEPY)

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

    if ui_state.is_offroad():
      rl.draw_rectangle(int(self.rect.x), int(self.rect.y), int(self.rect.width), int(self.rect.height), rl.Color(0, 0, 0, 175))
      upper_half = rl.Rectangle(rect.x, rect.y, rect.width, rect.height / 2)
      self._offroad_label.render(upper_half)
