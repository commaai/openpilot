import pyray as rl
from openpilot.system.ui.lib.widget import Widget

ON_COLOR = rl.Color(51, 171, 76, 255)
OFF_COLOR = rl.Color(0x39, 0x39, 0x39, 255)
KNOB_COLOR = rl.WHITE
DISABLED_ON_COLOR = rl.Color(0x22, 0x77, 0x22, 255)  # Dark green when disabled + on
DISABLED_OFF_COLOR = rl.Color(0x39, 0x39, 0x39, 255)
DISABLED_KNOB_COLOR = rl.Color(0x88, 0x88, 0x88, 255)
WIDTH, HEIGHT = 160, 80
BG_HEIGHT = 60
ANIMATION_SPEED = 8.0


class Toggle(Widget):
  def __init__(self, initial_state=False):
    super().__init__()
    self._state = initial_state
    self._enabled = True
    self._progress = 1.0 if initial_state else 0.0
    self._target = self._progress

  def set_rect(self, rect: rl.Rectangle):
    self._rect = rl.Rectangle(rect.x, rect.y, WIDTH, HEIGHT)

  def handle_input(self):
    if not self._enabled:
      return 0

    if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
      if rl.check_collision_point_rec(rl.get_mouse_position(), self._rect):
        self._state = not self._state
        self._target = 1.0 if self._state else 0.0
        return 1
    return 0

  def get_state(self):
    return self._state

  def set_state(self, state: bool):
    self._state = state
    self._target = 1.0 if state else 0.0

  def set_enabled(self, enabled: bool):
    self._enabled = enabled

  def is_enabled(self):
    return self._enabled

  def update(self):
    if abs(self._progress - self._target) > 0.01:
      delta = rl.get_frame_time() * ANIMATION_SPEED
      self._progress += delta if self._progress < self._target else -delta
      self._progress = max(0.0, min(1.0, self._progress))

  def _render(self, rect: rl.Rectangle):
    self.update()

    if self._enabled:
      bg_color = self._blend_color(OFF_COLOR, ON_COLOR, self._progress)
      knob_color = KNOB_COLOR
    else:
      bg_color = self._blend_color(DISABLED_OFF_COLOR, DISABLED_ON_COLOR, self._progress)
      knob_color = DISABLED_KNOB_COLOR

    # Draw background
    bg_rect = rl.Rectangle(self._rect.x + 5, self._rect.y + 10, WIDTH - 10, BG_HEIGHT)
    rl.draw_rectangle_rounded(bg_rect, 1.0, 10, bg_color)

    # Draw knob
    knob_x = self._rect.x + HEIGHT / 2 + (WIDTH - HEIGHT) * self._progress
    knob_y = self._rect.y + HEIGHT / 2
    rl.draw_circle(int(knob_x), int(knob_y), HEIGHT / 2, knob_color)

    return self.handle_input()

  def _blend_color(self, c1, c2, t):
    return rl.Color(int(c1.r + (c2.r - c1.r) * t), int(c1.g + (c2.g - c1.g) * t), int(c1.b + (c2.b - c1.b) * t), 255)
